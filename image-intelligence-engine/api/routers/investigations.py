"""Investigation workspace endpoints.

Every mutating action writes an audit entry — including refusals. A request
rejected for a missing lawful basis is exactly the event an auditor asks about
later, so recording only successes would leave the hole where it matters most.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request, status

from shared import workflow
from shared.errors import (
    LawfulBasisRequired,
    PolicyViolation,
    StateTransitionError,
    ValidationError,
)
from shared.logging import get_logger

from ..dependencies import (
    AuditRepoDep,
    CurrentUser,
    InvestigationRepoDep,
    PipelineRepoDep,
    RetentionRepoDep,
    ReviewRepoDep,
    SettingsDep,
    client_label,
)
from ..schemas import (
    InvestigationCreate,
    InvestigationResponse,
    StatusEventResponse,
    StatusTransitionRequest,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/investigations", tags=["investigations"])


@router.post("", response_model=InvestigationResponse, status_code=status.HTTP_201_CREATED)
async def create_investigation(
    payload: InvestigationCreate,
    request: Request,
    user: CurrentUser,
    settings: SettingsDep,
    investigations: InvestigationRepoDep,
    audit: AuditRepoDep,
) -> InvestigationResponse:
    """Open a new case."""
    actor = client_label(request, user)

    try:
        investigation = await investigations.create(
            owner_id=user.id,
            case_id=payload.case_id,
            title=payload.title,
            lawful_basis=payload.lawful_basis,
            purpose=payload.purpose,
            description=payload.description,
            jurisdiction=payload.jurisdiction,
            retention_days=payload.retention_days or settings.default_retention_days,
            require_lawful_basis=settings.require_lawful_basis,
        )
    except (LawfulBasisRequired, ValidationError) as exc:
        # Record the refusal and commit it *before* re-raising. The session
        # dependency rolls back on an exception, which would otherwise discard
        # the very entry an auditor later asks for. Committing here is safe
        # because the repository validates before it writes anything, so there
        # is no partial investigation to leak.
        await audit.record(
            action="investigation.create",
            outcome="refused",
            actor_id=user.id,
            actor_label=actor,
            lawful_basis=payload.lawful_basis,
            detail={"reason": str(exc), "case_id": payload.case_id},
        )
        await audit.session.commit()
        raise

    await audit.record(
        action="investigation.create",
        outcome="created",
        investigation_id=investigation.id,
        actor_id=user.id,
        actor_label=actor,
        lawful_basis=investigation.lawful_basis,
        resource_type="investigation",
        resource_id=str(investigation.id),
        detail={
            "case_id": investigation.case_id,
            "title": investigation.title,
            "jurisdiction": investigation.jurisdiction,
            "retention_expires_at": (
                investigation.retention_expires_at.isoformat()
                if investigation.retention_expires_at
                else None
            ),
        },
    )
    logger.info(
        "investigation.created", investigation_id=str(investigation.id),
        case_id=investigation.case_id,
    )
    return InvestigationResponse.model_validate(investigation)


@router.get("", response_model=list[InvestigationResponse])
async def list_investigations(
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    status_filter: str | None = None,
    limit: int = 50,
) -> list[InvestigationResponse]:
    rows = await investigations.list(
        user.id, status=status_filter, limit=max(1, min(limit, 200))
    )
    return [InvestigationResponse.model_validate(row) for row in rows]


@router.get("/{investigation_id}", response_model=InvestigationResponse)
async def get_investigation(
    investigation_id: uuid.UUID,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
) -> InvestigationResponse:
    investigation = await investigations.get(investigation_id)
    _assert_owner(investigation, user)
    return InvestigationResponse.model_validate(investigation)


@router.post("/{investigation_id}/status", response_model=InvestigationResponse)
async def transition_status(
    investigation_id: uuid.UUID,
    payload: StatusTransitionRequest,
    request: Request,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    review: ReviewRepoDep,
    pipeline: PipelineRepoDep,
    retention: RetentionRepoDep,
    audit: AuditRepoDep,
) -> InvestigationResponse:
    """Move a case through its workflow.

    The only supported way to change status. Three layers apply, in order: the
    transition table decides whether the move is *shaped* correctly, the
    preconditions decide whether it is permitted *right now* given live case
    state, and backward moves additionally require a reason.
    """
    investigation = await investigations.get(investigation_id)
    _assert_owner(investigation, user)
    previous = investigation.status
    actor = client_label(request, user)

    state = workflow.CaseState(
        pending_reviews=await review.pending_count(investigation_id),
        completed_runs=sum(
            1 for run in await pipeline.list_runs(investigation_id) if run.status == "COMPLETED"
        ),
        active_holds=await retention.active_hold_count(investigation_id),
    )
    failures = workflow.check_transition(previous, str(payload.to_status), state)
    if failures:
        await audit.record(
            action="investigation.status",
            outcome="refused",
            investigation_id=investigation_id,
            actor_id=user.id,
            actor_label=actor,
            resource_type="investigation",
            resource_id=str(investigation_id),
            detail={
                "from": previous,
                "requested": str(payload.to_status),
                "failed_preconditions": [f.as_dict() for f in failures],
            },
        )
        await audit.session.commit()
        raise PolicyViolation(
            " ".join(f.message for f in failures),
            failed_preconditions=[f.rule for f in failures],
        )

    try:
        updated = await investigations.transition(
            investigation, payload.to_status, actor_id=user.id, reason=payload.reason
        )
    except (StateTransitionError, ValidationError) as exc:
        await audit.record(
            action="investigation.status",
            outcome="refused",
            investigation_id=investigation_id,
            actor_id=user.id,
            actor_label=actor,
            resource_type="investigation",
            resource_id=str(investigation_id),
            detail={
                "reason": str(exc),
                "from": previous,
                "requested": str(payload.to_status),
            },
        )
        # As above: commit the refusal so the rollback cannot erase it. The
        # transition validates before mutating, so nothing partial is committed.
        await audit.session.commit()
        raise

    await audit.record(
        action="investigation.status",
        outcome=str(payload.to_status),
        investigation_id=investigation_id,
        actor_id=user.id,
        actor_label=actor,
        lawful_basis=investigation.lawful_basis,
        resource_type="investigation",
        resource_id=str(investigation_id),
        detail={"from": previous, "to": str(payload.to_status), "reason": payload.reason},
    )
    return InvestigationResponse.model_validate(updated)


@router.get("/{investigation_id}/status-history", response_model=list[StatusEventResponse])
async def status_history(
    investigation_id: uuid.UUID,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
) -> list[StatusEventResponse]:
    investigation = await investigations.get(investigation_id)
    _assert_owner(investigation, user)
    events = await investigations.status_history(investigation_id)
    return [StatusEventResponse.model_validate(event) for event in events]


def _assert_owner(investigation, user) -> None:  # noqa: ANN001
    """Ownership check.

    Raises NotFound rather than Forbidden: telling an unauthorised caller that a
    case id exists is itself a disclosure.
    """
    from shared.errors import NotFoundError

    if investigation.owner_id != user.id:
        raise NotFoundError(f"Investigation {investigation.id} not found.")


__all__ = ["router"]
