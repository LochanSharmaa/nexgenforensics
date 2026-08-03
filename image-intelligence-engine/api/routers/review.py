"""Human review queue and evidence reads.

The review queue is the layer that separates machine observation from
human-confirmed interpretation. Extraction never writes a confirmed entity; it
writes an observation plus a review item, and a human decision promotes it.

A rejection never deletes the underlying observation. "The machine saw this and
a human disagreed" is itself a finding, and discarding it would erase the reason
a later reader might disagree back.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request

from shared.errors import NotFoundError
from shared.logging import get_logger

from ..dependencies import (
    AuditRepoDep,
    CurrentUser,
    FactRepoDep,
    InvestigationRepoDep,
    ReviewRepoDep,
    client_label,
)
from ..schemas import (
    EvidenceChainResponse,
    FactClassifyRequest,
    FactResponse,
    ObservationResponse,
    ReviewDecisionRequest,
    ReviewItemResponse,
    ReviewQueueSummary,
)

logger = get_logger(__name__)
router = APIRouter(tags=["review"])


async def _owned(investigations, investigation_id: uuid.UUID, user):  # noqa: ANN001
    investigation = await investigations.get(investigation_id)
    if investigation.owner_id != user.id:
        raise NotFoundError(f"Investigation {investigation_id} not found.")
    return investigation


# ------------------------------------------------------------ review queue --


@router.get(
    "/investigations/{investigation_id}/review", response_model=list[ReviewItemResponse]
)
async def pending_reviews(
    investigation_id: uuid.UUID,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    review: ReviewRepoDep,
    limit: int = 50,
) -> list[ReviewItemResponse]:
    await _owned(investigations, investigation_id, user)
    items = await review.pending(investigation_id, limit=max(1, min(limit, 200)))
    return [ReviewItemResponse.model_validate(item) for item in items]


@router.get(
    "/investigations/{investigation_id}/review/summary", response_model=ReviewQueueSummary
)
async def review_summary(
    investigation_id: uuid.UUID,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    review: ReviewRepoDep,
) -> ReviewQueueSummary:
    """Queue depth, and whether it currently blocks completion.

    `UNDER_REVIEW → COMPLETED` is gated on this being zero: a case must not be
    signed off with machine output nobody has looked at.
    """
    await _owned(investigations, investigation_id, user)
    pending = await review.pending_count(investigation_id)
    return ReviewQueueSummary(pending=pending, blocks_completion=pending > 0)


@router.post("/review/{item_id}/decide", response_model=ReviewItemResponse)
async def decide_review(
    item_id: uuid.UUID,
    payload: ReviewDecisionRequest,
    request: Request,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    review: ReviewRepoDep,
    audit: AuditRepoDep,
) -> ReviewItemResponse:
    """Record a human ruling on a machine proposal."""
    item = await review.get(item_id)
    await _owned(investigations, item.investigation_id, user)

    decided = await review.decide(
        item, payload.status, user_id=user.id, note=payload.note
    )
    await audit.record(
        action="review.decide", outcome=str(payload.status),
        investigation_id=item.investigation_id, actor_id=user.id,
        actor_label=client_label(request, user),
        resource_type="review_item", resource_id=str(item_id),
        detail={"kind": item.kind, "subject_type": item.subject_type, "note": payload.note},
    )
    return ReviewItemResponse.model_validate(decided)


# ---------------------------------------------------------------- evidence --


@router.get("/facts/{fact_id}/evidence-chain", response_model=EvidenceChainResponse)
async def evidence_chain(
    fact_id: uuid.UUID,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    facts: FactRepoDep,
) -> EvidenceChainResponse:
    """The traceability guarantee: fact → observations → page.

    An API contract rather than a UI feature, so every consumer inherits it and
    no client has to know how to assemble provenance itself. `confidence` and
    `confidence_explanation` always travel together — there is no projection
    that returns a number without its explanation.
    """
    fact = await facts.get(fact_id)
    await _owned(investigations, fact.investigation_id, user)

    chain = await facts.evidence_chain(fact_id)

    # Competing values are surfaced with the fact, not hidden behind a separate
    # call — a reader must see the disagreement at the same moment as the claim.
    conflicts = []
    if fact.conflict_group_id is not None:
        conflicts = [
            other
            for other in await facts.conflicting(fact.conflict_group_id)
            if other.id != fact.id
        ]

    return EvidenceChainResponse(
        fact=FactResponse.model_validate(fact),
        chain=[ObservationResponse.model_validate(o) for o in chain],
        conflicts=[FactResponse.model_validate(c) for c in conflicts],
    )


@router.post("/facts/{fact_id}/classify", response_model=FactResponse)
async def classify_fact(
    fact_id: uuid.UUID,
    payload: FactClassifyRequest,
    request: Request,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    facts: FactRepoDep,
    audit: AuditRepoDep,
) -> FactResponse:
    """Record an investigator's judgement on a fact.

    Touches the investigative axis only. `status` — the evidential axis — is
    machine-derived and is never overwritten here: a human doubting a fact does
    not change how many independent sources asserted it.
    """
    fact = await facts.get(fact_id)
    await _owned(investigations, fact.investigation_id, user)

    previous = fact.classification
    updated = await facts.classify(
        fact, payload.classification, user_id=user.id, note=payload.note
    )
    await audit.record(
        action="fact.classify", outcome=str(payload.classification),
        investigation_id=fact.investigation_id, actor_id=user.id,
        actor_label=client_label(request, user),
        resource_type="fact", resource_id=str(fact_id),
        detail={
            "from": previous,
            "to": str(payload.classification),
            "evidential_status": fact.status,
            "note": payload.note,
        },
    )
    return FactResponse.model_validate(updated)


__all__ = ["router"]
