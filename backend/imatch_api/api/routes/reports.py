from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlmodel import Session

from ...core.config import Settings, get_settings
from ...core.dependencies import Principal, client_context, get_current_principal
from ...db.models import Case, Role
from ...db.session import get_session
from ...services.audit_service import ACTION_EXPORT, ACTION_NARRATIVE, AuditService
from ...services.narrative_service import NarrativeService
from ...services.report_service import ReportService
from ...services.storage_service import StorageService
from .auth import get_audit_service
from .search import get_storage

router = APIRouter(prefix="/api/cases", tags=["reports"])


@router.get("/{case_id}/report")
def export_report(
    case_id: str,
    request: Request,
    fmt: str = "json",
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    storage: StorageService = Depends(get_storage),
    audit: AuditService = Depends(get_audit_service),
) -> Any:
    """Export a case report as JSON, Markdown or PDF.

    Exports are themselves audited: a report is a copy of biometric findings
    leaving the system, which is exactly the event a later review will ask about.

    The narrative layer is attached here rather than inside ReportService.build()
    so that the factual report is complete before any third party is involved,
    and so a failure in that layer cannot prevent an export. Every format gets
    the same attached narrative, preserving the property that JSON, Markdown and
    PDF render from one dict and cannot disagree.
    """
    if fmt not in {"json", "markdown", "pdf"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "fmt must be json, markdown or pdf.")

    case = session.get(Case, case_id)
    if case is None or case.tenant_id != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found.")
    if not principal.has_role(Role.SUPERVISOR) and case.owner_id != principal.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found.")

    report = ReportService().build(session, principal.tenant_id, case_id, principal.label)

    narrative = NarrativeService(settings).attach(
        session,
        report,
        tenant_id=principal.tenant_id,
        case_id=case_id,
        generated_by=principal.label,
    )

    ip_address, user_agent = client_context(request)
    # Recorded only when findings actually left the system. A reused narrative
    # is read from our own database and sends nothing, so logging it as a
    # transfer would make the audit trail overstate what happened.
    if narrative.available and not narrative.reused:
        audit.record(
            session,
            tenant_id=principal.tenant_id,
            action=ACTION_NARRATIVE,
            actor_id=principal.id,
            actor_label=principal.label,
            resource_type="case",
            resource_id=case_id,
            outcome="success" if narrative.validator_status == "passed" else "rejected",
            detail={
                "provider": "google-gemini",
                "model": narrative.model,
                "evidence_digest": narrative.evidence_digest,
                "attempts": narrative.attempts,
                "validator_status": narrative.validator_status,
                "pseudonymised": True,
            },
            ip_address=ip_address,
            user_agent=user_agent,
        )

    audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_EXPORT,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="case",
        resource_id=case_id,
        detail={
            "format": fmt,
            "searches": report["summary"]["searches_run"],
            "narrative": narrative.validator_status if narrative.available else "absent",
        },
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
    if fmt == "pdf":
        # Rendered from the SAME report dict as the JSON and Markdown exports,
        # so the three formats cannot disagree about a finding.
        from ...services.report_pdf import render_case_report_pdf

        return Response(
            # The loader is scoped to this tenant's storage root and reads only
            # paths already present in this principal's own report. A path that
            # has aged out of the retention window returns None and the plate
            # prints as a stated absence rather than failing the export.
            content=render_case_report_pdf(report, image_loader=storage.read),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="case-{case.reference}.pdf"'},
        )
    return report


__all__ = ["router"]
