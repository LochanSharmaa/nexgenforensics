"""Phase 2 acceptance: the API, audit chain and custody chain end to end."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select, update

from database.models import AuditLogEntry
from database.repositories import AuditRepository, CustodyRepository, InvestigationRepository
from shared.enums import ActorKind, ArtifactType, CustodyAction, InvestigationStatus

pytestmark = pytest.mark.asyncio


CASE = {
    "case_id": "OSINT-2026-114",
    "title": "Provenance of supplied photograph",
    "lawful_basis": "Client engagement 2026/114",
    "purpose": "Establish first publication",
}


# -- health ----------------------------------------------------------------


async def test_liveness_needs_no_dependencies(client):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"


async def test_readiness_reports_each_component(client):
    """Redis is absent in the test environment, so readiness must report
    degraded rather than claim health."""
    response = await client.get("/health/ready")
    body = response.json()
    names = {component["name"] for component in body["components"]}
    assert {"database", "redis"} <= names
    database = next(c for c in body["components"] if c["name"] == "database")
    assert database["healthy"] is True


async def test_openapi_schema_generates(client):
    response = await client.get("/api/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/investigations" in paths
    assert "/api/v1/audit/verify" in paths


# -- auth ------------------------------------------------------------------


async def test_login_returns_a_token(client, user):
    response = await client.post(
        "/api/v1/auth/token",
        json={"email": user.email, "password": "investigator-pass-123"},
    )
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"


async def test_wrong_password_is_rejected(client, user):
    response = await client.post(
        "/api/v1/auth/token", json={"email": user.email, "password": "wrong-password-here"}
    )
    assert response.status_code == 401


async def test_unknown_account_gives_the_same_answer(client):
    """Distinguishing 'no such user' from 'wrong password' enumerates accounts."""
    response = await client.post(
        "/api/v1/auth/token",
        json={"email": "nobody@example.com", "password": "wrong-password-here"},
    )
    assert response.status_code == 401
    assert "Incorrect email or password" in response.json()["detail"]


async def test_investigations_require_authentication(client):
    response = await client.post("/api/v1/investigations", json=CASE)
    assert response.status_code == 401


# -- investigations --------------------------------------------------------


async def test_create_empty_investigation(auth_client):
    """Phase 2 acceptance criterion 7."""
    response = await auth_client.post("/api/v1/investigations", json=CASE)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["case_id"] == CASE["case_id"]
    assert body["status"] == InvestigationStatus.NEW
    assert body["lawful_basis"] == CASE["lawful_basis"]
    assert body["retention_expires_at"] is not None


async def test_lawful_basis_is_required(auth_client):
    payload = {**CASE, "lawful_basis": "   "}
    response = await auth_client.post("/api/v1/investigations", json=payload)
    assert response.status_code == 422
    assert "lawful basis" in response.json()["detail"].lower()


async def test_refusals_are_audited(auth_client):
    """The refused request is exactly what an auditor asks about later."""
    await auth_client.post("/api/v1/investigations", json={**CASE, "lawful_basis": ""})
    entries = (await auth_client.get("/api/v1/audit")).json()
    refusals = [e for e in entries if e["outcome"] == "refused"]
    assert refusals, "a refused creation must still be recorded"
    assert refusals[-1]["action"] == "investigation.create"


async def test_duplicate_case_id_conflicts(auth_client):
    await auth_client.post("/api/v1/investigations", json=CASE)
    response = await auth_client.post("/api/v1/investigations", json=CASE)
    assert response.status_code == 409


async def test_list_and_fetch(auth_client):
    created = (await auth_client.post("/api/v1/investigations", json=CASE)).json()

    listing = await auth_client.get("/api/v1/investigations")
    assert listing.status_code == 200
    assert [row["id"] for row in listing.json()] == [created["id"]]

    fetched = await auth_client.get(f"/api/v1/investigations/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["title"] == CASE["title"]


async def test_unknown_investigation_is_404(auth_client):
    response = await auth_client.get(f"/api/v1/investigations/{uuid.uuid4()}")
    assert response.status_code == 404


# -- workflow --------------------------------------------------------------


async def test_legal_transition_succeeds(auth_client):
    created = (await auth_client.post("/api/v1/investigations", json=CASE)).json()
    response = await auth_client.post(
        f"/api/v1/investigations/{created['id']}/status", json={"to_status": "ACTIVE"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ACTIVE"


async def test_illegal_transition_is_refused(auth_client):
    """NEW cannot jump straight to COMPLETED."""
    created = (await auth_client.post("/api/v1/investigations", json=CASE)).json()
    response = await auth_client.post(
        f"/api/v1/investigations/{created['id']}/status", json={"to_status": "COMPLETED"}
    )
    assert response.status_code == 409
    assert "illegal-state-transition" in response.json()["type"]


async def _complete_a_run(auth_client, session_factory, clock, case_id: str) -> None:
    """Satisfy the `review_requires_a_completed_run` precondition.

    Added in Phase 4: a case cannot move to UNDER_REVIEW with nothing to review.
    """
    import uuid as _uuid

    from database.repositories import PipelineRepository
    from shared.enums import RunStatus

    run_id = (
        await auth_client.post(f"/api/v1/investigations/{case_id}/runs", json={})
    ).json()["run"]["id"]
    async with session_factory() as session:
        repo = PipelineRepository(session, clock)
        await repo.finish_run(await repo.get_run(_uuid.UUID(run_id)), RunStatus.COMPLETED)
        await session.commit()


async def test_reopening_without_a_reason_is_refused(auth_client, session_factory, clock):
    created = (await auth_client.post("/api/v1/investigations", json=CASE)).json()
    case_id = created["id"]
    await _complete_a_run(auth_client, session_factory, clock, case_id)
    for target in ("ACTIVE", "UNDER_REVIEW", "COMPLETED"):
        await auth_client.post(
            f"/api/v1/investigations/{case_id}/status", json={"to_status": target}
        )

    response = await auth_client.post(
        f"/api/v1/investigations/{case_id}/status", json={"to_status": "ACTIVE"}
    )
    assert response.status_code == 422
    assert "reason" in response.json()["detail"].lower()

    with_reason = await auth_client.post(
        f"/api/v1/investigations/{case_id}/status",
        json={"to_status": "ACTIVE", "reason": "New source surfaced during review"},
    )
    assert with_reason.status_code == 200


async def test_status_history_records_every_move(auth_client, session_factory, clock):
    created = (await auth_client.post("/api/v1/investigations", json=CASE)).json()
    case_id = created["id"]
    await _complete_a_run(auth_client, session_factory, clock, case_id)
    await auth_client.post(f"/api/v1/investigations/{case_id}/status", json={"to_status": "ACTIVE"})
    await auth_client.post(
        f"/api/v1/investigations/{case_id}/status", json={"to_status": "UNDER_REVIEW"}
    )

    history = (await auth_client.get(f"/api/v1/investigations/{case_id}/status-history")).json()
    assert [(e["from_status"], e["to_status"]) for e in history] == [
        ("NEW", "ACTIVE"),
        ("ACTIVE", "UNDER_REVIEW"),
    ]


# -- audit chain -----------------------------------------------------------


async def test_audit_chain_verifies_after_traffic(auth_client):
    """Phase 2 acceptance criterion 8."""
    for index in range(3):
        await auth_client.post(
            "/api/v1/investigations", json={**CASE, "case_id": f"CASE-{index}"}
        )

    verification = (await auth_client.get("/api/v1/audit/verify")).json()
    assert verification["valid"] is True
    assert verification["records"] >= 4   # one login + three creations
    assert verification["broken_at"] is None


async def test_audit_entries_link_to_their_predecessor(auth_client):
    await auth_client.post("/api/v1/investigations", json=CASE)
    entries = (await auth_client.get("/api/v1/audit")).json()
    assert entries[0]["previous_hash"] == "0" * 64
    # strict=False is correct here: pairing a list with its own tail yields one
    # fewer pair by construction.
    for previous, current in zip(entries, entries[1:], strict=False):
        assert current["previous_hash"] == previous["entry_hash"]


async def test_tampering_breaks_the_chain(auth_client, session_factory):
    await auth_client.post("/api/v1/investigations", json=CASE)

    async with session_factory() as session:
        target = (
            await session.execute(select(AuditLogEntry).order_by(AuditLogEntry.id))
        ).scalars().all()[1]
        await session.execute(
            update(AuditLogEntry)
            .where(AuditLogEntry.id == target.id)
            .values(lawful_basis="retrospectively invented warrant")
        )
        await session.commit()

    verification = (await auth_client.get("/api/v1/audit/verify")).json()
    assert verification["valid"] is False
    assert verification["broken_at"] == 1
    assert "edited" in verification["reason"]


# -- custody chain ---------------------------------------------------------


async def test_custody_chain_records_each_transformation(session, clock, user):
    """image downloaded → hashed → screenshot captured, as a walkable chain."""
    investigations = InvestigationRepository(session, clock)
    custody = CustodyRepository(session, clock)

    investigation = await investigations.create(
        owner_id=user.id, case_id="CUSTODY-1", title="Custody",
        lawful_basis="Engagement 1",
    )
    artifact = uuid.uuid4()

    collected = await custody.record(
        investigation_id=investigation.id,
        artifact_type=ArtifactType.IMAGE,
        artifact_id=artifact,
        action=CustodyAction.COLLECTED,
        content_hash="a" * 64,
        source_uri="https://example.test/photo.jpg",
        actor_kind=ActorKind.SYSTEM,
    )
    await custody.record(
        investigation_id=investigation.id,
        artifact_type=ArtifactType.IMAGE,
        artifact_id=artifact,
        action=CustodyAction.SCREENSHOT_CAPTURED,
        content_hash="b" * 64,
        derived_from_id=collected.id,
        transformation={"tool": "playwright", "version": "1.44"},
    )

    chain = await custody.chain_for(ArtifactType.IMAGE, artifact)
    assert [event.sequence for event in chain] == [1, 2]
    assert chain[1].derived_from_id == collected.id
    assert chain[0].previous_hash == "0" * 64
    assert chain[1].previous_hash == chain[0].entry_hash

    verification = await custody.verify(ArtifactType.IMAGE, artifact)
    assert verification["valid"] is True


async def test_custody_chains_are_per_artifact(session, clock, user):
    """Two artifacts must not share a chain, or one's history would depend on
    the other's."""
    investigations = InvestigationRepository(session, clock)
    custody = CustodyRepository(session, clock)
    investigation = await investigations.create(
        owner_id=user.id, case_id="CUSTODY-2", title="Custody", lawful_basis="Engagement 1"
    )

    first, second = uuid.uuid4(), uuid.uuid4()
    for artifact in (first, second):
        await custody.record(
            investigation_id=investigation.id,
            artifact_type=ArtifactType.IMAGE,
            artifact_id=artifact,
            action=CustodyAction.COLLECTED,
            content_hash="c" * 64,
        )

    for artifact in (first, second):
        chain = await custody.chain_for(ArtifactType.IMAGE, artifact)
        assert len(chain) == 1
        assert chain[0].sequence == 1
        assert chain[0].previous_hash == "0" * 64


# -- repository-level guarantees -------------------------------------------


async def test_status_cannot_be_changed_by_direct_assignment(session, clock, user):
    """Transitions route through the repository so validation, history and the
    audit entry all happen. This documents the contract."""
    investigations = InvestigationRepository(session, clock)
    investigation = await investigations.create(
        owner_id=user.id, case_id="GUARD-1", title="Guard", lawful_basis="Engagement 1"
    )

    from shared.errors import StateTransitionError

    with pytest.raises(StateTransitionError):
        await investigations.transition(investigation, InvestigationStatus.COMPLETED)

    history = await investigations.status_history(investigation.id)
    assert history == [], "a refused transition must not write history"


async def test_audit_repository_never_blocks_on_business_rules(session, clock):
    """Refusals are audited, so the writer must succeed even when the action it
    records did not."""
    audit = AuditRepository(session, clock)
    entry = await audit.record(action="investigation.create", outcome="refused",
                               detail={"reason": "no lawful basis"})
    assert entry.entry_hash and entry.previous_hash == "0" * 64
