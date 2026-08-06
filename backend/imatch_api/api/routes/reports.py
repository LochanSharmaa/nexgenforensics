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
from ...services.engine_service import get_engine_service
from ...services.storage_service import StorageService
from .auth import get_audit_service
from .search import get_storage

router = APIRouter(prefix="/api/cases", tags=["reports"])


def _export_examination(
    fmt: str,
    case: Case,
    case_id: str,
    request: Request,
    principal: Principal,
    session: Session,
    storage: StorageService,
    audit: AuditService,
    engine: Any,
) -> Any:
    """Build and return the photograph examination report.

    Audited as an export like any other: a report carrying the imagery itself,
    rather than hashes of it, is the strongest form of biometric findings
    leaving the system.
    """
    from ...services.examination_pdf import render_examination_pdf
    from ...services.examination_service import ExaminationService

    report = ExaminationService(engine, storage.read).build(
        session, principal.tenant_id, case_id, principal.label
    )

    exhibit_count = sum(
        len(photo["marks"])
        for role in ("questioned", "specimen")
        for photo in report["exhibits"][role]
    )
    ip_address, user_agent = client_context(request)
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
            "report": "photograph_examination",
            "exhibits_marked": exhibit_count,
            "sections_unavailable": len(report["unavailable"]),
        },
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()

    if fmt == "examination-json":
        # The PIL images are for the renderer only; they are not serialisable
        # and a JSON export must carry hashes, not pixels.
        return {key: value for key, value in report.items() if not key.startswith("_")}

    return Response(
        content=render_examination_pdf(report),
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                f'attachment; filename="examination-{case.reference}.pdf"'
        },
    )


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
    engine: Any = Depends(get_engine_service),
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
    if fmt not in {"json", "markdown", "pdf", "examination", "examination-json"}:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "fmt must be json, markdown, pdf, examination or examination-json.",
        )

    case = session.get(Case, case_id)
    if case is None or case.tenant_id != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found.")
    if not principal.has_role(Role.SUPERVISOR) and case.owner_id != principal.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found.")

    # The photograph examination report is a different document from the case
    # log, built from the imagery rather than from the search record, so it
    # takes its own path and does not carry the narrative layer -- none of its
    # prose is generated.
    if fmt in {"examination", "examination-json"}:
        return _export_examination(
            fmt, case, case_id, request, principal, session, storage, audit, engine
        )

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
