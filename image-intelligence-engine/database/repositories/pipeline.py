"""Pipeline runs, stages, retention holds and config snapshots.

The persistence side of the resume machinery in `shared/pipeline.py`. Nothing
here decides *policy* — where to resume, whether a transition is legal — it
applies decisions the domain layer already made and records them.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared import pipeline as domain
from shared.clock import Clock, SystemClock
from shared.config import Settings
from shared.enums import (
    PIPELINE_ORDER,
    RunStatus,
    RunTrigger,
    StageStatus,
)
from shared.errors import (
    ConflictError,
    NotFoundError,
    RetentionHoldActive,
    StateTransitionError,
    ValidationError,
)
from shared.logging import get_logger

from ..models import (
    ConfigSnapshot,
    Investigation,
    PipelineRun,
    PipelineStageRow,
    RetentionHold,
)

logger = get_logger(__name__)


class PipelineRepository:
    """Runs and their per-stage state."""

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    # ------------------------------------------------------------- reading --

    async def get_run(self, run_id: uuid.UUID) -> PipelineRun:
        run = (
            await self.session.execute(select(PipelineRun).where(PipelineRun.id == run_id))
        ).scalar_one_or_none()
        if run is None:
            raise NotFoundError(f"Pipeline run {run_id} not found.")
        return run

    async def latest_run(self, investigation_id: uuid.UUID) -> PipelineRun | None:
        result = await self.session.execute(
            select(PipelineRun)
            .where(PipelineRun.investigation_id == investigation_id)
            .order_by(PipelineRun.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_runs(
        self, investigation_id: uuid.UUID, limit: int = 50
    ) -> Sequence[PipelineRun]:
        result = await self.session.execute(
            select(PipelineRun)
            .where(PipelineRun.investigation_id == investigation_id)
            .order_by(PipelineRun.id.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def stages(self, run_id: uuid.UUID) -> Sequence[PipelineStageRow]:
        """Stage rows in pipeline order, not insertion order.

        The UI renders these as a sequence, and a stage that failed and was
        retried would otherwise appear out of place.
        """
        result = await self.session.execute(
            select(PipelineStageRow).where(PipelineStageRow.run_id == run_id)
        )
        rows = list(result.scalars().all())
        order = {str(stage): index for index, stage in enumerate(PIPELINE_ORDER)}
        rows.sort(key=lambda row: order.get(row.stage, len(order)))
        return rows

    async def stage_states(self, run_id: uuid.UUID) -> list[domain.StageState]:
        return [
            domain.StageState(
                stage=row.stage,
                status=row.status,
                items_total=row.items_total,
                items_done=row.items_done,
                items_failed=row.items_failed,
            )
            for row in await self.stages(run_id)
        ]

    async def resume_plan(self, run_id: uuid.UUID) -> domain.ResumePlan:
        return domain.plan_resume(await self.stage_states(run_id))

    # ------------------------------------------------------------- writing --

    async def start_or_resume(
        self,
        investigation: Investigation,
        *,
        settings: Settings,
        trigger: RunTrigger | str = RunTrigger.MANUAL,
    ) -> tuple[PipelineRun, domain.ResumePlan, bool]:
        """Continue the latest unfinished run, or begin a new one.

        Returns ``(run, plan, resumed)``. Resuming reuses the existing run rather
        than creating a fresh one, because the stage rows *are* the record of
        completed work — starting over would discard them and re-run paid
        discovery calls.
        """
        now = self.clock.now()
        latest = await self.latest_run(investigation.id)

        if latest is not None and domain.is_resumable(latest.status):
            plan = await self.resume_plan(latest.id)
            if plan.is_complete:
                # Every stage settled but the run never closed — finish it
                # rather than leaving a permanently "resumable" run behind.
                await self.finish_run(latest, RunStatus.COMPLETED)
                raise ConflictError(
                    "The latest run has already completed every stage; "
                    "nothing to resume. Start a new run instead."
                )

            # Already RUNNING is a no-op, not an error. This endpoint is
            # "start or continue", so a client retrying after a dropped
            # response must get the current state back rather than a 409 for a
            # request that already succeeded.
            if latest.status != RunStatus.RUNNING:
                if not domain.can_transition_run(latest.status, RunStatus.RUNNING):
                    raise StateTransitionError(
                        f"A run in state {latest.status} cannot be resumed."
                    )
                latest.status = RunStatus.RUNNING
            latest.started_at = latest.started_at or now
            latest.error = ""
            await self._capture_config(investigation, latest, settings)
            await self.session.flush()
            logger.info(
                "pipeline.resumed", run_id=str(latest.id),
                next_stage=plan.next_stage, skipped=plan.skipped_work,
            )
            return latest, plan, True

        run = PipelineRun(
            investigation_id=investigation.id,
            trigger=str(trigger),
            status=RunStatus.QUEUED,
            started_at=now,
        )
        self.session.add(run)
        await self.session.flush()

        # Materialise every stage up front so progress is reportable as a
        # complete list from the first moment, not a list that grows.
        for stage in PIPELINE_ORDER:
            self.session.add(
                PipelineStageRow(
                    run_id=run.id, stage=str(stage), status=StageStatus.PENDING
                )
            )
        await self._capture_config(investigation, run, settings)
        await self.session.flush()

        plan = await self.resume_plan(run.id)
        logger.info("pipeline.started", run_id=str(run.id), stages=len(PIPELINE_ORDER))
        return run, plan, False

    async def _capture_config(
        self, investigation: Investigation, run: PipelineRun, settings: Settings
    ) -> ConfigSnapshot:
        """Record the configuration this attempt executes under.

        Captured per *attempt*, not per run: a run paused before a model upgrade
        and resumed after it contains findings from two different regimes, and
        the report has to be able to say which came from which.
        """
        snapshot = ConfigSnapshot(
            investigation_id=investigation.id,
            pipeline_run_id=run.id,
            app_version=settings.app_version,
            ruleset_version=settings.ruleset_version,
            scorer_version="",
            classifier_version="",
            parser_versions={},
            extractor_versions={},
            provider_versions={},
            prompt_versions={},
            thresholds={},
            captured_at=self.clock.now(),
        )
        self.session.add(snapshot)
        await self.session.flush()
        return snapshot

    async def transition_run(self, run: PipelineRun, to_status: RunStatus | str) -> PipelineRun:
        target = str(to_status)
        if not domain.can_transition_run(run.status, target):
            permitted = ", ".join(sorted(domain.RUN_TRANSITIONS.get(run.status, set())))
            raise StateTransitionError(
                f"A run in state {run.status} cannot move to {target}. "
                f"Permitted: {permitted or '(terminal)'}"
            )
        run.status = target
        if target in (RunStatus.COMPLETED, RunStatus.FAILED):
            run.finished_at = self.clock.now()
        await self.session.flush()
        return run

    async def finish_run(
        self, run: PipelineRun, status: RunStatus | str, error: str = ""
    ) -> PipelineRun:
        run.status = str(status)
        run.finished_at = self.clock.now()
        run.error = error
        await self.session.flush()
        return run

    async def begin_stage(self, run_id: uuid.UUID, stage: str, items_total: int = 0):
        row = await self._stage_row(run_id, stage)
        row.status = StageStatus.RUNNING
        row.started_at = self.clock.now()
        row.items_total = items_total
        row.items_done = 0
        row.items_failed = 0
        row.error = ""
        await self.session.flush()
        return row

    async def complete_stage(
        self,
        run_id: uuid.UUID,
        stage: str,
        *,
        status: StageStatus | str = StageStatus.OK,
        items_done: int | None = None,
        items_failed: int | None = None,
        error: str = "",
    ):
        row = await self._stage_row(run_id, stage)
        row.status = str(status)
        row.finished_at = self.clock.now()
        if items_done is not None:
            row.items_done = items_done
        if items_failed is not None:
            row.items_failed = items_failed
        row.error = error
        await self.session.flush()
        return row

    async def invalidate_downstream(self, run_id: uuid.UUID, stage: str) -> int:
        """Reset stages that depend on ``stage`` back to PENDING.

        Output derived from a stage is stale once that stage re-runs. Leaving it
        OK would mix findings computed from two different inputs into one report.
        """
        downstream = set(domain.stages_after(stage))
        reset = 0
        for row in await self.stages(run_id):
            if row.stage in downstream and row.status != StageStatus.PENDING:
                row.status = StageStatus.PENDING
                row.started_at = None
                row.finished_at = None
                row.error = ""
                reset += 1
        await self.session.flush()
        return reset

    async def _stage_row(self, run_id: uuid.UUID, stage: str) -> PipelineStageRow:
        row = (
            await self.session.execute(
                select(PipelineStageRow).where(
                    PipelineStageRow.run_id == run_id,
                    PipelineStageRow.stage == str(stage),
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise NotFoundError(f"Stage {stage} not found on run {run_id}.")
        return row


class RetentionRepository:
    """Preservation locks and the purge precondition.

    Holds win over policy, always. A legal hold that a scheduler could override
    is not a hold.
    """

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    async def place_hold(
        self,
        *,
        investigation_id: uuid.UUID,
        reason: str,
        placed_by: uuid.UUID,
        artifact_type: str = "INVESTIGATION",
        artifact_id: uuid.UUID | None = None,
    ) -> RetentionHold:
        if not reason.strip():
            raise ValidationError(
                "A retention hold must state its reason. An unexplained hold "
                "cannot be reviewed, and therefore cannot be released with confidence."
            )
        hold = RetentionHold(
            investigation_id=investigation_id,
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            reason=reason.strip(),
            placed_by=placed_by,
            placed_at=self.clock.now(),
        )
        self.session.add(hold)
        await self.session.flush()
        logger.info("retention.hold_placed", investigation_id=str(investigation_id))
        return hold

    async def release_hold(self, hold: RetentionHold, *, released_by: uuid.UUID) -> RetentionHold:
        if hold.released_at is not None:
            raise ConflictError("That hold has already been released.")
        hold.released_by = released_by
        hold.released_at = self.clock.now()
        await self.session.flush()
        return hold

    async def get_hold(self, hold_id: uuid.UUID) -> RetentionHold:
        hold = (
            await self.session.execute(
                select(RetentionHold).where(RetentionHold.id == hold_id)
            )
        ).scalar_one_or_none()
        if hold is None:
            raise NotFoundError(f"Retention hold {hold_id} not found.")
        return hold

    async def list_holds(
        self, investigation_id: uuid.UUID, *, active_only: bool = False
    ) -> Sequence[RetentionHold]:
        statement = select(RetentionHold).where(
            RetentionHold.investigation_id == investigation_id
        )
        if active_only:
            statement = statement.where(RetentionHold.released_at.is_(None))
        result = await self.session.execute(statement.order_by(RetentionHold.placed_at))
        return result.scalars().all()

    async def active_hold_count(self, investigation_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(RetentionHold)
            .where(
                RetentionHold.investigation_id == investigation_id,
                RetentionHold.released_at.is_(None),
            )
        )
        return int(result.scalar_one() or 0)

    async def assert_purgeable(self, investigation_id: uuid.UUID) -> None:
        """Raise unless every precondition for purging is satisfied."""
        active = await self.active_hold_count(investigation_id)
        if active:
            raise RetentionHoldActive(
                f"{active} unreleased retention hold(s) block purging this investigation. "
                "Holds take precedence over retention policy without exception.",
                active_holds=active,
            )

    async def snapshot(self, investigation_id: uuid.UUID) -> dict[str, Any]:
        """Everything the UI needs to explain the retention position."""
        holds = await self.list_holds(investigation_id)
        active = [h for h in holds if h.released_at is None]
        return {
            "holds_total": len(holds),
            "holds_active": len(active),
            "purge_blocked": bool(active),
            "block_reasons": [h.reason for h in active],
        }


__all__ = ["PipelineRepository", "RetentionRepository"]
