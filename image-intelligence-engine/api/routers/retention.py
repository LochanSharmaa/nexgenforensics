"""Retention holds and the purge precondition.

A hold is a preservation lock. While one is unreleased, purging is refused
absolutely — policy does not override it, the scheduler does not override it,
and there is no force flag. A legal hold a scheduler could bypass is not a hold.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request, status

from shared.errors import NotFoundError
from shared.logging import get_logger

from ..dependencies import (
    AuditRepoDep,
    CurrentUser,
    InvestigationRepoDep,
    RetentionRepoDep,
    client_label,
)
from ..schemas import HoldCreateRequest, HoldResponse, RetentionStatusResponse

logger = get_logger(__name__)
router = APIRouter(tags=["retention"])


async def _owned(investigations, investigation_id: uuid.UUID, user):  # noqa: ANN001
    investigation = await investigations.get(investigation_id)
    if investigation.owner_id != user.id:
        raise NotFoundError(f"Investigation {investigation_id} not found.")
    return investigation


@router.post(
    "/investigations/{investigation_id}/holds",
    response_model=HoldResponse,
    status_code=status.HTTP_201_CREATED,
)
async def place_hold(
    investigation_id: uuid.UUID,
    payload: HoldCreateRequest,
    request: Request,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    retention: RetentionRepoDep,
    audit: AuditRepoDep,
) -> HoldResponse:
    """Place a preservation lock."""
    investigation = await _owned(investigations, investigation_id, user)

    hold = await retention.place_hold(
        investigation_id=investigation_id,
        reason=payload.reason,
        placed_by=user.id,
        artifact_type=str(payload.artifact_type),
        artifact_id=payload.artifact_id,
    )
    await audit.record(
        action="retention.hold", outcome="placed",
        investigation_id=investigation_id, actor_id=user.id,
        actor_label=client_label(request, user),
        lawful_basis=investigation.lawful_basis,
        resource_type="retention_hold", resource_id=str(hold.id),
        detail={"reason": hold.reason, "artifact_type": hold.artifact_type},
    )
    return HoldResponse.model_validate(hold)


@router.get("/investigations/{investigation_id}/holds", response_model=list[HoldResponse])
async def list_holds(
    investigation_id: uuid.UUID,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    retention: RetentionRepoDep,
    active_only: bool = False,
) -> list[HoldResponse]:
    await _owned(investigations, investigation_id, user)
    holds = await retention.list_holds(investigation_id, active_only=active_only)
    return [HoldResponse.model_validate(hold) for hold in holds]


@router.delete("/holds/{hold_id}", response_model=HoldResponse)
async def release_hold(
    hold_id: uuid.UUID,
    request: Request,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    retention: RetentionRepoDep,
    audit: AuditRepoDep,
) -> HoldResponse:
    """Release a hold.

    Releasing does not delete the row: the record that a hold existed, who
    placed it and who lifted it is part of the investigation's history.
    """
    hold = await retention.get_hold(hold_id)
    await _owned(investigations, hold.investigation_id, user)

    released = await retention.release_hold(hold, released_by=user.id)
    await audit.record(
        action="retention.hold", outcome="released",
        investigation_id=hold.investigation_id, actor_id=user.id,
        actor_label=client_label(request, user),
        resource_type="retention_hold", resource_id=str(hold_id),
        detail={"reason": hold.reason},
    )
    return HoldResponse.model_validate(released)


@router.get(
    "/investigations/{investigation_id}/retention", response_model=RetentionStatusResponse
)
async def retention_status(
    investigation_id: uuid.UUID,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    retention: RetentionRepoDep,
) -> RetentionStatusResponse:
    """Whether this case can currently be purged, and if not, why not."""
    investigation = await _owned(investigations, investigation_id, user)
    snapshot = await retention.snapshot(investigation_id)
    return RetentionStatusResponse(
        **snapshot, retention_expires_at=investigation.retention_expires_at
    )


@router.post(
    "/investigations/{investigation_id}/purge-check",
    response_model=RetentionStatusResponse,
)
async def purge_check(
    investigation_id: uuid.UUID,
    request: Request,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    retention: RetentionRepoDep,
    audit: AuditRepoDep,
) -> RetentionStatusResponse:
    """Dry-run the purge preconditions without deleting anything.

    Deliberately separate from purging itself, which arrives with the retention
    engine in Phase 12b. An operator must be able to ask "would this be allowed?"
    without the answer being "it already happened".
    """
    investigation = await _owned(investigations, investigation_id, user)
    try:
        await retention.assert_purgeable(investigation_id)
        blocked = False
    except Exception as exc:  # noqa: BLE001 - reported, not raised: this is a check
        blocked = True
        await audit.record(
            action="retention.purge_check", outcome="blocked",
            investigation_id=investigation_id, actor_id=user.id,
            actor_label=client_label(request, user),
            resource_type="investigation", resource_id=str(investigation_id),
            detail={"reason": str(exc)},
        )

    snapshot = await retention.snapshot(investigation_id)
    if not blocked:
        await audit.record(
            action="retention.purge_check", outcome="permitted",
            investigation_id=investigation_id, actor_id=user.id,
            actor_label=client_label(request, user),
            resource_type="investigation", resource_id=str(investigation_id),
            detail={},
        )
    return RetentionStatusResponse(
        **snapshot, retention_expires_at=investigation.retention_expires_at
    )


__all__ = ["router"]
