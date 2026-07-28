from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from ...core.config import Settings, get_settings
from ...core.dependencies import Principal, get_current_principal, require_admin
from ...db.models import AuditRecord
from ...db.session import get_session
from ...services.audit_service import AuditService
from ..schemas import AuditRecordResponse, ChainVerificationResponse

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("", response_model=list[AuditRecordResponse])
def list_audit_records(
    action: str | None = None,
    resource_id: str | None = None,
    since: datetime | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> list[AuditRecordResponse]:
    """Read the caller's tenant audit trail.

    Any authenticated user can read it, including their own entries. An audit
    log that only administrators can see is much easier to quietly misuse, and
    the operators being logged have a legitimate interest in what was recorded
    about them.
    """
    statement = select(AuditRecord).where(AuditRecord.tenant_id == principal.tenant_id)
    if action:
        statement = statement.where(AuditRecord.action == action)
    if resource_id:
        statement = statement.where(AuditRecord.resource_id == resource_id)
    if since:
        statement = statement.where(AuditRecord.created_at >= since)

    records = session.exec(statement.order_by(AuditRecord.sequence.desc()).limit(limit)).all()
    return [AuditRecordResponse.model_validate(record) for record in records]


@router.get("/verify", response_model=ChainVerificationResponse)
def verify_chain(
    principal: Principal = Depends(require_admin),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> ChainVerificationResponse:
    """Recompute the hash chain and report the first break, if any.

    A ``valid: false`` result means records were altered or removed after being
    written, which is a security incident rather than a data-quality issue.
    """
    verification = AuditService(settings.audit_path).verify_chain(session, principal.tenant_id)
    return ChainVerificationResponse(**verification.as_dict())


__all__ = ["router"]
