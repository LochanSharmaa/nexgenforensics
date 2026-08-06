"""The examiner's own text in a photograph examination report.

The report generator can inventory the exhibits, measure the faces and render
the plates. It cannot state the question that was put, describe a
correspondence, or reach a result -- those are opinion, and this is the path by
which a person supplies them.

Everything written here is attributed and timestamped, and every write is
audited. A forensic opinion that cannot be traced to who formed it and when is
not worth much when it is challenged.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session, select

from ...core.dependencies import Principal, client_context, get_current_principal
from ...db.models import (
    Case,
    ExaminationNote,
    ExaminationSection,
    Role,
    User,
    utcnow,
)
from ...db.session import get_session
from ...services.audit_service import AuditService
from ..schemas import ExaminationNoteRequest, ExaminationNoteResponse
from .auth import get_audit_service

router = APIRouter(prefix="/api/cases", tags=["examination"])

#: The examination notes are findings about a case, so they follow exactly the
#: access rule the report itself does.
ACTION_EXAMINATION_NOTE = "examination_note"


def _load_case(session: Session, principal: Principal, case_id: str) -> Case:
    case = session.get(Case, case_id)
    if case is None or case.tenant_id != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found.")
    if not principal.has_role(Role.SUPERVISOR) and case.owner_id != principal.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found.")
    return case


def _to_response(session: Session, note: ExaminationNote) -> ExaminationNoteResponse:
    author = session.get(User, note.author_id)
    try:
        marks = [str(mark) for mark in json.loads(note.marks or "[]")]
    except (ValueError, TypeError):
        marks = []
    return ExaminationNoteResponse(
        id=note.id,
        section=note.section.value,
        ordinal=note.ordinal,
        body=note.body,
        marks=marks,
        author=author.email if author else note.author_id,
        created_at=note.created_at,
        updated_at=note.updated_at,
    )


@router.get("/{case_id}/examination-notes", response_model=list[ExaminationNoteResponse])
def list_notes(
    case_id: str,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> list[ExaminationNoteResponse]:
    _load_case(session, principal, case_id)
    notes = session.exec(
        select(ExaminationNote)
        .where(
            ExaminationNote.tenant_id == principal.tenant_id,
            ExaminationNote.case_id == case_id,
        )
        .order_by(ExaminationNote.section, ExaminationNote.ordinal)
    ).all()
    return [_to_response(session, note) for note in notes]


@router.put("/{case_id}/examination-notes", response_model=list[ExaminationNoteResponse])
def replace_notes(
    case_id: str,
    payload: list[ExaminationNoteRequest],
    request: Request,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
    audit: AuditService = Depends(get_audit_service),
) -> list[ExaminationNoteResponse]:
    """Replace this case's examination notes with the supplied set.

    A whole-set replace rather than per-note edits: the observations are
    numbered in the report and their order is meaningful, so the examiner edits
    the list as a list. The previous set is discarded, but the audit record
    keeps a count of what was there, and nothing is silently merged.
    """
    _load_case(session, principal, case_id)

    existing = session.exec(
        select(ExaminationNote).where(
            ExaminationNote.tenant_id == principal.tenant_id,
            ExaminationNote.case_id == case_id,
        )
    ).all()
    for note in existing:
        session.delete(note)

    written: list[ExaminationNote] = []
    for entry in payload:
        if not entry.body.strip():
            continue
        try:
            section = ExaminationSection(entry.section)
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Unknown section {entry.section!r}. Expected one of: "
                + ", ".join(item.value for item in ExaminationSection),
            ) from exc

        note = ExaminationNote(
            tenant_id=principal.tenant_id,
            case_id=case_id,
            section=section,
            ordinal=entry.ordinal,
            body=entry.body.strip(),
            marks=json.dumps([mark.strip() for mark in entry.marks if mark.strip()]),
            author_id=principal.id,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(note)
        written.append(note)

    ip_address, user_agent = client_context(request)
    audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_EXAMINATION_NOTE,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="case",
        resource_id=case_id,
        detail={
            "replaced": len(existing),
            "written": len(written),
            "sections": sorted({note.section.value for note in written}),
        },
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()
    for note in written:
        session.refresh(note)
    return [_to_response(session, note) for note in written]


__all__ = ["ACTION_EXAMINATION_NOTE", "router"]
