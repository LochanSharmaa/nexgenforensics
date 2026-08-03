"""Phase 4 acceptance: pipeline runs, live progress, retention holds, review gate.

The headline requirement is that an investigation can be created, paused,
resumed and archived through the API alone, with the audit chain intact
throughout.
"""

from __future__ import annotations

import json
import uuid

from shared.enums import PIPELINE_ORDER, StageStatus

CASE = {
    "case_id": "OSINT-2026-400",
    "title": "Phase 4 workspace",
    "lawful_basis": "Client engagement 2026/400",
    "purpose": "Exercise the run lifecycle",
}


async def _case(auth_client, **overrides) -> str:
    payload = {**CASE, **overrides}
    response = await auth_client.post("/api/v1/investigations", json=payload)
    assert response.status_code == 201, response.text
    return response.json()["id"]


# ------------------------------------------------------------ run lifecycle --


async def test_starting_a_run_materialises_every_stage(auth_client):
    """The full stage list exists from the first moment, so progress is a
    complete picture rather than a list that grows."""
    case_id = await _case(auth_client)
    response = await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["resumed"] is False
    assert body["plan"]["next_stage"] == "INGEST"
    assert body["plan"]["stages_skipped"] == 0
    assert body["progress"] == 0.0

    stages = (await auth_client.get(f"/api/v1/runs/{body['run']['id']}/stages")).json()
    assert len(stages) == len(PIPELINE_ORDER)
    assert {s["status"] for s in stages} == {StageStatus.PENDING}


async def test_stages_are_returned_in_pipeline_order(auth_client):
    case_id = await _case(auth_client)
    run = (await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})).json()
    stages = (await auth_client.get(f"/api/v1/runs/{run['run']['id']}/stages")).json()
    assert [s["stage"] for s in stages] == [str(s) for s in PIPELINE_ORDER]


async def test_resume_reuses_the_run_and_skips_completed_stages(
    auth_client, session_factory, clock
):
    """Resuming must not restart. Re-running an early stage would repeat a paid
    discovery call and duplicate observations for the same source.
    """
    from database.repositories import PipelineRepository

    case_id = await _case(auth_client)
    started = (
        await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})
    ).json()
    run_id = started["run"]["id"]

    # Simulate the worker finishing the first two stages.
    async with session_factory() as session:
        repo = PipelineRepository(session, clock)
        for stage in ("INGEST", "DISCOVER"):
            await repo.complete_stage(uuid.UUID(run_id), stage, status=StageStatus.OK, items_done=5)
        await session.commit()

    resumed = (
        await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})
    ).json()

    assert resumed["resumed"] is True
    assert resumed["run"]["id"] == run_id, "resume must continue the same run"
    assert resumed["plan"]["next_stage"] == "VERIFY"
    assert resumed["plan"]["stages_skipped"] == 2
    assert resumed["progress"] > 0


async def test_pause_then_resume_round_trips(auth_client):
    case_id = await _case(auth_client)
    run_id = (
        await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})
    ).json()["run"]["id"]

    # QUEUED must reach RUNNING before it can pause.
    await auth_client.post(f"/api/v1/runs/{run_id}/resume")
    paused = await auth_client.post(f"/api/v1/runs/{run_id}/pause")
    assert paused.status_code == 200
    assert paused.json()["status"] == "PAUSED"

    resumed = await auth_client.post(f"/api/v1/runs/{run_id}/resume")
    assert resumed.status_code == 200
    assert resumed.json()["run"]["status"] == "RUNNING"


async def test_illegal_run_transition_is_refused(auth_client, session_factory, clock):
    """A completed run is terminal."""
    from database.repositories import PipelineRepository
    from shared.enums import RunStatus

    case_id = await _case(auth_client)
    run_id = (
        await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})
    ).json()["run"]["id"]

    async with session_factory() as session:
        repo = PipelineRepository(session, clock)
        await repo.finish_run(await repo.get_run(uuid.UUID(run_id)), RunStatus.COMPLETED)
        await session.commit()

    response = await auth_client.post(f"/api/v1/runs/{run_id}/resume")
    assert response.status_code == 409
    assert "illegal-state-transition" in response.json()["type"]


async def test_run_start_is_audited_with_the_saving(auth_client):
    case_id = await _case(auth_client)
    await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})
    entries = (await auth_client.get(f"/api/v1/audit?investigation_id={case_id}")).json()
    starts = [e for e in entries if e["action"] == "pipeline.run"]
    assert starts and starts[-1]["outcome"] == "started"
    assert starts[-1]["detail"]["next_stage"] == "INGEST"


async def test_config_snapshot_captured_per_attempt(auth_client, session_factory):
    """A run paused before a model upgrade and resumed after contains two
    regimes; the report has to be able to say which findings came from which."""
    from sqlalchemy import select

    from database.models import ConfigSnapshot

    case_id = await _case(auth_client)
    await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})

    async with session_factory() as session:
        rows = (await session.execute(select(ConfigSnapshot))).scalars().all()
    assert len(rows) == 1
    assert rows[0].ruleset_version


# --------------------------------------------------------------- live SSE --


async def test_progress_stream_opens_with_a_full_snapshot(auth_client, session_factory, clock):
    """A client joining late must be immediately correct, not wait for the next
    change — which is why this streams state rather than events."""
    from database.repositories import PipelineRepository
    from shared.enums import RunStatus

    case_id = await _case(auth_client)
    run_id = (
        await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})
    ).json()["run"]["id"]

    # Terminal state so the stream closes deterministically.
    async with session_factory() as session:
        repo = PipelineRepository(session, clock)
        await repo.complete_stage(uuid.UUID(run_id), "INGEST", status=StageStatus.OK, items_done=3)
        await repo.finish_run(await repo.get_run(uuid.UUID(run_id)), RunStatus.COMPLETED)
        await session.commit()

    async with auth_client.stream("GET", f"/api/v1/runs/{run_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = ""
        async for chunk in response.aiter_text():
            body += chunk
            if "event: end" in body:
                break

    assert "event: progress" in body
    payload = json.loads(body.split("event: progress\ndata: ", 1)[1].split("\n\n", 1)[0])
    assert payload["run_id"] == run_id
    assert len(payload["stages"]) == len(PIPELINE_ORDER)
    assert payload["progress"] > 0
    assert "event: end" in body


# ------------------------------------------------------- retention holds --


async def test_hold_requires_a_reason(auth_client):
    case_id = await _case(auth_client)
    response = await auth_client.post(
        f"/api/v1/investigations/{case_id}/holds", json={"reason": ""}
    )
    assert response.status_code == 422


async def test_hold_blocks_purge_and_release_restores_it(auth_client):
    case_id = await _case(auth_client)

    clear = (await auth_client.post(f"/api/v1/investigations/{case_id}/purge-check")).json()
    assert clear["purge_blocked"] is False

    hold = (
        await auth_client.post(
            f"/api/v1/investigations/{case_id}/holds",
            json={"reason": "Litigation hold, matter 2026/44"},
        )
    ).json()

    blocked = (await auth_client.post(f"/api/v1/investigations/{case_id}/purge-check")).json()
    assert blocked["purge_blocked"] is True
    assert blocked["holds_active"] == 1
    assert "Litigation hold" in blocked["block_reasons"][0]

    released = await auth_client.delete(f"/api/v1/holds/{hold['id']}")
    assert released.status_code == 200
    assert released.json()["released_at"] is not None

    after = (await auth_client.post(f"/api/v1/investigations/{case_id}/purge-check")).json()
    assert after["purge_blocked"] is False
    assert after["holds_total"] == 1, "released holds stay on the record"


async def test_double_release_is_a_conflict(auth_client):
    case_id = await _case(auth_client)
    hold = (
        await auth_client.post(
            f"/api/v1/investigations/{case_id}/holds", json={"reason": "hold"}
        )
    ).json()
    await auth_client.delete(f"/api/v1/holds/{hold['id']}")
    again = await auth_client.delete(f"/api/v1/holds/{hold['id']}")
    assert again.status_code == 409


async def test_hold_blocks_marking_for_deletion(auth_client):
    """Holds take precedence over retention policy without exception."""
    case_id = await _case(auth_client)
    await auth_client.post(
        f"/api/v1/investigations/{case_id}/holds", json={"reason": "Preserve for counsel"}
    )
    await auth_client.post(
        f"/api/v1/investigations/{case_id}/status", json={"to_status": "ARCHIVED"}
    )
    response = await auth_client.post(
        f"/api/v1/investigations/{case_id}/status",
        json={"to_status": "DELETED_PENDING_RETENTION"},
    )
    assert response.status_code == 422
    assert "retention hold" in response.json()["detail"].lower()


# ---------------------------------------------------------- review gating --


async def test_review_requires_a_completed_run(auth_client):
    """Nothing to review until the pipeline has produced something."""
    case_id = await _case(auth_client)
    await auth_client.post(f"/api/v1/investigations/{case_id}/status", json={"to_status": "ACTIVE"})
    response = await auth_client.post(
        f"/api/v1/investigations/{case_id}/status", json={"to_status": "UNDER_REVIEW"}
    )
    assert response.status_code == 422
    assert "no pipeline run has completed" in response.json()["detail"]


async def _reach_under_review(auth_client, session_factory, clock, case_id: str) -> None:
    from database.repositories import PipelineRepository
    from shared.enums import RunStatus

    run_id = (
        await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})
    ).json()["run"]["id"]
    async with session_factory() as session:
        repo = PipelineRepository(session, clock)
        await repo.finish_run(await repo.get_run(uuid.UUID(run_id)), RunStatus.COMPLETED)
        await session.commit()

    await auth_client.post(f"/api/v1/investigations/{case_id}/status", json={"to_status": "ACTIVE"})
    response = await auth_client.post(
        f"/api/v1/investigations/{case_id}/status", json={"to_status": "UNDER_REVIEW"}
    )
    assert response.status_code == 200, response.text


async def test_pending_review_blocks_completion(auth_client, session_factory, clock):
    """A case must not be signed off with extraction output nobody has read."""
    from database.repositories import ReviewRepository
    from shared.enums import ReviewKind

    case_id = await _case(auth_client)
    await _reach_under_review(auth_client, session_factory, clock, case_id)

    async with session_factory() as session:
        review = ReviewRepository(session, clock)
        await review.propose(
            investigation_id=uuid.UUID(case_id), kind=ReviewKind.ENTITY_CANDIDATE,
            subject_type="entity", subject_id=uuid.uuid4(),
            proposal={"name": "Meridian"}, rationale={"observations": []},
        )
        await session.commit()

    summary = (
        await auth_client.get(f"/api/v1/investigations/{case_id}/review/summary")
    ).json()
    assert summary == {"pending": 1, "blocks_completion": True}

    blocked = await auth_client.post(
        f"/api/v1/investigations/{case_id}/status", json={"to_status": "COMPLETED"}
    )
    assert blocked.status_code == 422
    assert "awaiting a human decision" in blocked.json()["detail"]

    items = (await auth_client.get(f"/api/v1/investigations/{case_id}/review")).json()
    decided = await auth_client.post(
        f"/api/v1/review/{items[0]['id']}/decide",
        json={"status": "REJECTED", "note": "Not the subject of this case"},
    )
    assert decided.status_code == 200

    now_allowed = await auth_client.post(
        f"/api/v1/investigations/{case_id}/status", json={"to_status": "COMPLETED"}
    )
    assert now_allowed.status_code == 200
    assert now_allowed.json()["status"] == "COMPLETED"


async def test_blocked_transition_is_audited_with_the_failed_rule(
    auth_client, session_factory, clock
):
    case_id = await _case(auth_client)
    await auth_client.post(f"/api/v1/investigations/{case_id}/status", json={"to_status": "ACTIVE"})
    await auth_client.post(
        f"/api/v1/investigations/{case_id}/status", json={"to_status": "UNDER_REVIEW"}
    )
    entries = (await auth_client.get(f"/api/v1/audit?investigation_id={case_id}")).json()
    refusals = [e for e in entries if e["outcome"] == "refused"]
    assert refusals
    assert refusals[-1]["detail"]["failed_preconditions"][0]["rule"] == (
        "review_requires_a_completed_run"
    )


async def test_review_decision_cannot_revert_to_pending(auth_client, session_factory, clock):
    from database.repositories import ReviewRepository
    from shared.enums import ReviewKind

    case_id = await _case(auth_client)
    async with session_factory() as session:
        review = ReviewRepository(session, clock)
        item = await review.propose(
            investigation_id=uuid.UUID(case_id), kind=ReviewKind.CONFLICT,
            subject_type="fact", subject_id=uuid.uuid4(),
            proposal={}, rationale={"observations": []},
        )
        await session.commit()
        item_id = str(item.id)

    response = await auth_client.post(
        f"/api/v1/review/{item_id}/decide", json={"status": "PENDING"}
    )
    assert response.status_code == 422


# ------------------------------------------------- full lifecycle + audit --


async def test_case_runs_the_whole_lifecycle_through_the_api(
    auth_client, session_factory, clock
):
    """Phase 4 acceptance: created, paused, resumed and archived via API alone,
    with the audit chain intact at the end."""
    case_id = await _case(auth_client, case_id="LIFECYCLE-1")

    run_id = (
        await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})
    ).json()["run"]["id"]
    await auth_client.post(f"/api/v1/runs/{run_id}/resume")
    assert (await auth_client.post(f"/api/v1/runs/{run_id}/pause")).status_code == 200
    assert (await auth_client.post(f"/api/v1/runs/{run_id}/resume")).status_code == 200

    from database.repositories import PipelineRepository
    from shared.enums import RunStatus

    async with session_factory() as session:
        repo = PipelineRepository(session, clock)
        await repo.finish_run(await repo.get_run(uuid.UUID(run_id)), RunStatus.COMPLETED)
        await session.commit()

    for target in ("ACTIVE", "UNDER_REVIEW", "COMPLETED", "ARCHIVED"):
        response = await auth_client.post(
            f"/api/v1/investigations/{case_id}/status", json={"to_status": target}
        )
        assert response.status_code == 200, f"{target}: {response.text}"

    final = (await auth_client.get(f"/api/v1/investigations/{case_id}")).json()
    assert final["status"] == "ARCHIVED"

    verification = (await auth_client.get("/api/v1/audit/verify")).json()
    assert verification["valid"] is True
    assert verification["broken_at"] is None


# ---------------------------------------------------------------- idempotency --


async def test_starting_an_already_running_run_is_idempotent(auth_client):
    """A retry after a dropped response must not 409 a request that succeeded.

    Found by a live acceptance run: RUNNING is a resumable state but has no
    self-transition, so the naive path rejected the retry.
    """
    case_id = await _case(auth_client)
    first = (
        await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})
    ).json()
    run_id = first["run"]["id"]
    await auth_client.post(f"/api/v1/runs/{run_id}/resume")

    again = await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})
    assert again.status_code == 201, again.text
    body = again.json()
    assert body["run"]["id"] == run_id
    assert body["resumed"] is True
    assert body["run"]["status"] == "RUNNING"


async def test_resuming_a_running_run_is_idempotent(auth_client):
    case_id = await _case(auth_client)
    run_id = (
        await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})
    ).json()["run"]["id"]

    first = await auth_client.post(f"/api/v1/runs/{run_id}/resume")
    second = await auth_client.post(f"/api/v1/runs/{run_id}/resume")
    assert first.status_code == second.status_code == 200
    assert second.json()["run"]["status"] == "RUNNING"
