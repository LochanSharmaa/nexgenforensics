from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlmodel import Session

from ...core.dependencies import Principal, client_context, get_current_principal
from ...db.models import Case, Role
from ...db.session import get_session
from ...services.audit_service import ACTION_EXPORT, AuditService
from ...services.report_service import ReportService
from .auth import get_audit_service

router = APIRouter(prefix="/api/cases", tags=["reports"])


@router.get("/{case_id}/report")
def export_report(
    case_id: str,
    request: Request,
    fmt: str = "json",
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
    audit: AuditService = Depends(get_audit_service),
) -> Any:
    """Export a case report as JSON or Markdown.

    Exports are themselves audited: a report is a copy of biometric findings
    leaving the system, which is exactly the event a later review will ask about.
    """
    if fmt not in {"json", "markdown"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "fmt must be json or markdown.")

    case = session.get(Case, case_id)
    if case is None or case.tenant_id != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found.")
    if not principal.has_role(Role.SUPERVISOR) and case.owner_id != principal.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found.")

    report = ReportService().build(session, principal.tenant_id, case_id, principal.label)

    ip_address, user_agent = client_context(request)
    audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_EXPORT,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="case",
        resource_id=case_id,
        detail={"format": fmt, "searches": report["summary"]["searches_run"]},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()

    if fmt == "markdown":
        return Response(
            content=ReportService().to_markdown(report),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="case-{case.reference}.md"'},
        )
    return report


__all__ = ["router"]
