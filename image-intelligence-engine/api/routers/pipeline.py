"""Pipeline runs, stage state and live progress.

**Deviation from ARCHITECTURE §4.1, recorded deliberately.** The design called
for progress events published to a Redis channel and relayed over SSE. This
streams from the `pipeline_stages` table instead, and that is the better choice
here for a reason beyond simplicity: pub/sub is lossy for late subscribers. A
browser that connects mid-run, or reconnects after a laptop sleeps, misses every
event already published and would render an empty progress view for a run that is
80% done. Reading the stage rows always yields *current state*, so a reconnecting
client is immediately correct.

Redis stays in the stack for the job queue. Should multi-instance fan-out ever
matter, pub/sub can be layered on top of this without changing the contract.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request, status
from fastapi.responses import StreamingResponse

from shared import pipeline as domain
from shared.enums import RunStatus
from shared.errors import NotFoundError
from shared.logging import get_logger

from ..dependencies import (
    AuditRepoDep,
    CurrentUser,
    InvestigationRepoDep,
    PipelineRepoDep,
    SessionFactoryDep,
    SettingsDep,
    client_label,
)
from ..schemas import (
    ResumePlanResponse,
    RunDetailResponse,
    RunResponse,
    RunStartRequest,
    RunStartResponse,
    StageResponse,
)

logger = get_logger(__name__)
router = APIRouter(tags=["pipeline"])

SSE_POLL_SECONDS = 1.0
SSE_KEEPALIVE_SECONDS = 15.0


def _plan_response(plan: domain.ResumePlan) -> ResumePlanResponse:
    return ResumePlanResponse(
        next_stage=plan.next_stage,
        completed=list(plan.completed),
        remaining=list(plan.remaining),
        is_complete=plan.is_complete,
        stages_skipped=plan.skipped_work,
    )


async def _assert_owner(investigations, investigation_id: uuid.UUID, user) -> None:  # noqa: ANN001
    investigation = await investigations.get(investigation_id)
    if investigation.owner_id != user.id:
        # NotFound rather than Forbidden: confirming a case id exists is itself
        # a disclosure.
        raise NotFoundError(f"Investigation {investigation_id} not found.")


@router.post(
    "/investigations/{investigation_id}/runs",
    response_model=RunStartResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_or_resume_run(
    investigation_id: uuid.UUID,
    payload: RunStartRequest,
    request: Request,
    user: CurrentUser,
    settings: SettingsDep,
    investigations: InvestigationRepoDep,
    pipeline: PipelineRepoDep,
    audit: AuditRepoDep,
) -> RunStartResponse:
    """Start a run, or continue the latest unfinished one.

    Resuming reuses the existing run rather than starting over: the stage rows
    *are* the record of completed work, and re-running an early stage would
    repeat a paid discovery call and produce a second set of observations for
    the same source — which independence scoring would then count twice.
    """
    investigation = await investigations.get(investigation_id)
    if investigation.owner_id != user.id:
        raise NotFoundError(f"Investigation {investigation_id} not found.")

    run, plan, resumed = await pipeline.start_or_resume(
        investigation, settings=settings, trigger=payload.trigger
    )
    states = await pipeline.stage_states(run.id)

    await audit.record(
        action="pipeline.run",
        outcome="resumed" if resumed else "started",
        investigation_id=investigation_id,
        actor_id=user.id,
        actor_label=client_label(request, user),
        lawful_basis=investigation.lawful_basis,
        resource_type="pipeline_run",
        resource_id=str(run.id),
        detail={
            "trigger": str(payload.trigger),
            "resumed": resumed,
            "next_stage": plan.next_stage,
            "stages_skipped": plan.skipped_work,
        },
    )
    return RunStartResponse(
        run=RunResponse.model_validate(run),
        resumed=resumed,
        plan=_plan_response(plan),
        progress=domain.progress_fraction(states),
    )


@router.get("/investigations/{investigation_id}/runs", response_model=list[RunResponse])
async def list_runs(
    investigation_id: uuid.UUID,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    pipeline: PipelineRepoDep,
    limit: int = 50,
) -> list[RunResponse]:
    await _assert_owner(investigations, investigation_id, user)
    runs = await pipeline.list_runs(investigation_id, limit=max(1, min(limit, 200)))
    return [RunResponse.model_validate(run) for run in runs]


@router.get("/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(
    run_id: uuid.UUID,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    pipeline: PipelineRepoDep,
) -> RunDetailResponse:
    run = await pipeline.get_run(run_id)
    await _assert_owner(investigations, run.investigation_id, user)

    rows = await pipeline.stages(run_id)
    states = await pipeline.stage_states(run_id)
    return RunDetailResponse(
        run=RunResponse.model_validate(run),
        stages=[StageResponse.model_validate(row) for row in rows],
        plan=_plan_response(domain.plan_resume(states)),
        progress=domain.progress_fraction(states),
    )


@router.get("/runs/{run_id}/stages", response_model=list[StageResponse])
async def list_stages(
    run_id: uuid.UUID,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    pipeline: PipelineRepoDep,
) -> list[StageResponse]:
    run = await pipeline.get_run(run_id)
    await _assert_owner(investigations, run.investigation_id, user)
    return [StageResponse.model_validate(row) for row in await pipeline.stages(run_id)]


@router.post("/runs/{run_id}/pause", response_model=RunResponse)
async def pause_run(
    run_id: uuid.UUID,
    request: Request,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    pipeline: PipelineRepoDep,
    audit: AuditRepoDep,
) -> RunResponse:
    run = await pipeline.get_run(run_id)
    await _assert_owner(investigations, run.investigation_id, user)
    updated = await pipeline.transition_run(run, RunStatus.PAUSED)
    await audit.record(
        action="pipeline.pause", outcome="paused",
        investigation_id=run.investigation_id, actor_id=user.id,
        actor_label=client_label(request, user),
        resource_type="pipeline_run", resource_id=str(run_id), detail={},
    )
    return RunResponse.model_validate(updated)


@router.post("/runs/{run_id}/resume", response_model=RunDetailResponse)
async def resume_run(
    run_id: uuid.UUID,
    request: Request,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    pipeline: PipelineRepoDep,
    audit: AuditRepoDep,
) -> RunDetailResponse:
    run = await pipeline.get_run(run_id)
    await _assert_owner(investigations, run.investigation_id, user)

    # Resuming a run that is already RUNNING is idempotent: a retry after a
    # dropped response must not fail a request that already succeeded.
    updated = (
        run
        if run.status == RunStatus.RUNNING
        else await pipeline.transition_run(run, RunStatus.RUNNING)
    )
    states = await pipeline.stage_states(run_id)
    plan = domain.plan_resume(states)

    await audit.record(
        action="pipeline.resume", outcome="running",
        investigation_id=run.investigation_id, actor_id=user.id,
        actor_label=client_label(request, user),
        resource_type="pipeline_run", resource_id=str(run_id),
        detail={"next_stage": plan.next_stage, "stages_skipped": plan.skipped_work},
    )
    return RunDetailResponse(
        run=RunResponse.model_validate(updated),
        stages=[StageResponse.model_validate(row) for row in await pipeline.stages(run_id)],
        plan=_plan_response(plan),
        progress=domain.progress_fraction(states),
    )


# ------------------------------------------------------------ live progress --


@router.get("/runs/{run_id}/events", include_in_schema=True)
async def stream_progress(
    run_id: uuid.UUID,
    request: Request,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    pipeline: PipelineRepoDep,
    session_factory: SessionFactoryDep,
) -> StreamingResponse:
    """Server-Sent Events carrying stage progress.

    SSE rather than WebSocket: the traffic is one-directional, browsers
    reconnect automatically, and it survives proxies that mishandle upgrade
    headers.

    The first event is always a full snapshot, so a client that joins late — or
    reconnects — is immediately correct rather than waiting for the next change.
    """
    run = await pipeline.get_run(run_id)
    await _assert_owner(investigations, run.investigation_id, user)

    async def events() -> AsyncIterator[bytes]:
        last_payload: str | None = None
        idle = 0.0

        # A dedicated session: the request-scoped one closes when this handler
        # returns, and the generator outlives it by design.
        while True:
            if await request.is_disconnected():
                break

            from database.repositories import PipelineRepository

            async with session_factory() as session:
                repo = PipelineRepository(session)
                current = await repo.get_run(run_id)
                rows = await repo.stages(run_id)
                states = await repo.stage_states(run_id)

            snapshot = {
                "run_id": str(run_id),
                "status": current.status,
                "progress": round(domain.progress_fraction(states), 4),
                "stages": [
                    {
                        "stage": row.stage,
                        "status": row.status,
                        "items_total": row.items_total,
                        "items_done": row.items_done,
                        "items_failed": row.items_failed,
                    }
                    for row in rows
                ],
            }
            payload = json.dumps(snapshot, separators=(",", ":"))

            if payload != last_payload:
                yield f"event: progress\ndata: {payload}\n\n".encode()
                last_payload = payload
                idle = 0.0

            # Terminal states end the stream; a client watching a finished run
            # should not hold a connection open forever.
            if current.status in (RunStatus.COMPLETED, RunStatus.FAILED):
                yield f"event: end\ndata: {json.dumps({'status': current.status})}\n\n".encode()
                break

            await asyncio.sleep(SSE_POLL_SECONDS)
            idle += SSE_POLL_SECONDS
            if idle >= SSE_KEEPALIVE_SECONDS:
                # Comment frame: keeps intermediaries from reaping an idle
                # connection during a long, quiet stage.
                yield b": keepalive\n\n"
                idle = 0.0

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # nginx must not buffer an event stream
        },
    )


__all__ = ["router"]
