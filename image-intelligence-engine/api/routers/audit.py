"""Audit trail and integrity verification.

Read-only by construction. There is no endpoint that writes, edits or deletes an
audit entry — entries are written as a side effect of the actions they record,
and the table is insert-only at the privilege level besides.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from ..dependencies import AuditRepoDep, CurrentUser, InvestigationRepoDep
from ..schemas import AuditEntryResponse, ChainVerifyResponse

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/verify", response_model=ChainVerifyResponse)
async def verify_chain(user: CurrentUser, audit: AuditRepoDep) -> ChainVerifyResponse:
    """Re-walk the hash chain and report the first divergence.

    `broken_at` is the index of the earliest failing record, which is where an
    edit or deletion occurred — everything after it fails as a consequence.
    """
    return ChainVerifyResponse(**await audit.verify())


@router.get("", response_model=list[AuditEntryResponse])
async def list_entries(
    user: CurrentUser,
    audit: AuditRepoDep,
    investigations: InvestigationRepoDep,
    investigation_id: uuid.UUID | None = None,
    limit: int = 100,
) -> list[AuditEntryResponse]:
    if investigation_id is not None:
        # Ownership check before disclosing any entry.
        investigation = await investigations.get(investigation_id)
        if investigation.owner_id != user.id:
            from shared.errors import NotFoundError

            raise NotFoundError(f"Investigation {investigation_id} not found.")

    entries = await audit.list(investigation_id, limit=max(1, min(limit, 500)))
    return [AuditEntryResponse.model_validate(entry) for entry in entries]


__all__ = ["router"]
