from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session, select

from nexgen_engine.inference.pipeline import InvalidImageError, NoFaceDetectedError

from ...core.config import Settings, get_settings
from ...core.dependencies import (
    Principal,
    client_context,
    enforce_rate_limit,
    get_current_principal,
    require_supervisor,
)
from ...db.models import Case, Subject, Template
from ...db.session import get_session
from ...services.audit_service import (
    ACTION_ENROL,
    ACTION_SUBJECT_DELETE,
    ACTION_TEMPLATE_DELETE,
    AuditService,
)
from ...services.engine_service import EngineService, get_engine_service
from ...services.storage_service import (
    ImageTooLargeError,
    StorageService,
    UnsupportedImageError,
    decode_base64_image,
)
from ..schemas import (
    EnrolRequest,
    EnrolResponse,
    SubjectResponse,
    TemplateResponse,
)
from .auth import get_audit_service
from .search import get_storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/subjects", tags=["subjects"])


@router.post("", response_model=EnrolResponse, status_code=status.HTTP_201_CREATED)
def enrol(
    payload: EnrolRequest,
    request: Request,
    principal: Principal = Depends(require_supervisor),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    engine: EngineService = Depends(get_engine_service),
    storage: StorageService = Depends(get_storage),
    audit: AuditService = Depends(get_audit_service),
) -> EnrolResponse:
    """Add a person, or another image of an existing person, to the gallery.

    Supervisor-only: enrolment determines who the system is *able* to find, so it
    is a deliberately higher-privilege action than running a search.

    Poor enrolment images are the most common cause of downstream false matches,
    so a low-quality image is rejected outright rather than quietly enrolled.
    """
    ip_address, user_agent = client_context(request)

    try:
        raw = decode_base64_image(payload.image_base64, settings.max_upload_bytes)
    except (UnsupportedImageError, ImageTooLargeError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    try:
        result = engine.encode(raw)
    except NoFaceDetectedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    if not result.quality.accepted:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            {
                "message": "Image quality is too low to enrol. A weak enrolment degrades every "
                "future search against this subject.",
                "quality": result.quality.as_dict(),
            },
        )
    if result.faces_detected > 1:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Multiple faces detected. Enrolment requires an image containing exactly one subject.",
        )

    case = _resolve_case(session, principal.tenant_id, payload.case_id)
    subject = _resolve_or_create_subject(session, principal, payload, case)
    stored = storage.store(principal.tenant_id, raw, category="enrolments")

    duplicate = session.exec(
        select(Template).where(
            Template.tenant_id == principal.tenant_id,
            Template.subject_id == subject.id,
            Template.image_sha256 == stored.sha256,
        )
    ).first()
    if duplicate is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "This exact image is already enrolled for this subject.",
        )

    sealed = engine.seal(result.embedding, principal.tenant_id)
    template = Template(
        tenant_id=principal.tenant_id,
        subject_id=subject.id,
        nonce=sealed.nonce,
        ciphertext=sealed.ciphertext,
        dimensions=sealed.dimensions,
        recognizer_backend=engine.runtime.recognizer.info.backend,
        recognizer_pack=engine.runtime.recognizer.info.model_pack,
        quality_score=result.quality.score,
        detector=result.detector_name,
        image_sha256=stored.sha256,
        image_path=stored.path,
        created_by=principal.id,
    )
    session.add(template)
    session.flush()

    # Make sure the tenant's gallery is materialised before adding to it, or the
    # later lazy load would re-add this template from the database and duplicate
    # it in the index.
    engine.ensure_loaded(session, principal.tenant_id)
    engine.register(
        principal.tenant_id,
        template.id,
        subject.id,
        result.embedding,
        {"quality": result.quality.score, "recognizer_pack": template.recognizer_pack},
    )

    record = audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_ENROL,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="template",
        resource_id=template.id,
        lawful_basis=payload.lawful_basis.strip() or (case.lawful_basis if case else ""),
        detail={
            "subject_id": subject.id,
            "image_sha256": stored.sha256,
            "quality": result.quality.score,
            "recognizer_pack": template.recognizer_pack,
            "recognition_capable": result.recognition_capable,
        },
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()
    session.refresh(subject)
    session.refresh(template)

    warnings = list(result.reasons)
    if not result.recognition_capable:
        warnings.append(
            "Stored with the deterministic stub: this template cannot match anyone. "
            "Re-enrol after installing the recognition model."
        )

    return EnrolResponse(
        subject=SubjectResponse.model_validate(subject),
        template=TemplateResponse.model_validate(template),
        quality=result.quality.as_dict(),
        liveness=result.liveness.as_dict(),
        warnings=warnings,
        audit_hash=record.entry_hash,
    )


@router.get("", response_model=list[SubjectResponse])
def list_subjects(
    case_id: str | None = None,
    limit: int = 100,
    principal: Principal = Depends(enforce_rate_limit),
    session: Session = Depends(get_session),
) -> list[SubjectResponse]:
    statement = select(Subject).where(
        Subject.tenant_id == principal.tenant_id,
        Subject.active == True,  # noqa: E712
    )
    if case_id:
        statement = statement.where(Subject.case_id == case_id)
    statement = statement.order_by(Subject.created_at.desc()).limit(max(1, min(limit, 500)))
    return [SubjectResponse.model_validate(subject) for subject in session.exec(statement).all()]


@router.get("/{subject_id}", response_model=SubjectResponse)
def get_subject(
    subject_id: str,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> SubjectResponse:
    return SubjectResponse.model_validate(_load_subject(session, principal.tenant_id, subject_id))


@router.get("/{subject_id}/templates", response_model=list[TemplateResponse])
def list_templates(
    subject_id: str,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> list[TemplateResponse]:
    """Template metadata only.

    The template vector itself is never returned. An ArcFace embedding is enough
    to reconstruct a recognizable approximation of the face, so it is treated as
    equivalent to the biometric it was derived from.
    """
    _load_subject(session, principal.tenant_id, subject_id)
    rows = session.exec(
        select(Template).where(
            Template.tenant_id == principal.tenant_id,
            Template.subject_id == subject_id,
        )
    ).all()
    return [TemplateResponse.model_validate(row) for row in rows]


@router.delete("/{subject_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_subject(
    subject_id: str,
    request: Request,
    principal: Principal = Depends(require_supervisor),
    session: Session = Depends(get_session),
    engine: EngineService = Depends(get_engine_service),
    storage: StorageService = Depends(get_storage),
    audit: AuditService = Depends(get_audit_service),
) -> None:
    """Erase a subject: templates, enrolment images, and gallery entries.

    A real deletion, not a flag. Retaining biometric data after an erasure
    request is the failure mode that gets these systems shut down. Search
    history is preserved, because deleting the record of past searches would
    destroy the audit trail; the candidate rows keep the subject id only.
    """
    subject = _load_subject(session, principal.tenant_id, subject_id)

    templates = session.exec(
        select(Template).where(
            Template.tenant_id == principal.tenant_id,
            Template.subject_id == subject_id,
        )
    ).all()

    for template in templates:
        if template.image_path:
            storage.delete(template.image_path)
        session.delete(template)

    removed = engine.unregister_subject(principal.tenant_id, subject_id)
    session.delete(subject)

    ip_address, user_agent = client_context(request)
    audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_SUBJECT_DELETE,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="subject",
        resource_id=subject_id,
        detail={"templates_deleted": len(templates), "index_entries_removed": removed},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()


@router.delete("/{subject_id}/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_template(
    subject_id: str,
    template_id: str,
    request: Request,
    principal: Principal = Depends(require_supervisor),
    session: Session = Depends(get_session),
    engine: EngineService = Depends(get_engine_service),
    storage: StorageService = Depends(get_storage),
    audit: AuditService = Depends(get_audit_service),
) -> None:
    _load_subject(session, principal.tenant_id, subject_id)
    template = session.get(Template, template_id)
    if template is None or template.tenant_id != principal.tenant_id or template.subject_id != subject_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Template not found.")

    if template.image_path:
        storage.delete(template.image_path)
    engine.unregister(principal.tenant_id, template_id)
    session.delete(template)

    ip_address, user_agent = client_context(request)
    audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_TEMPLATE_DELETE,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="template",
        resource_id=template_id,
        detail={"subject_id": subject_id},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()


# ------------------------------------------------------------------ helpers --


def _resolve_case(session: Session, tenant_id: str, case_id: str | None) -> Case | None:
    if not case_id:
        return None
    case = session.get(Case, case_id)
    if case is None or case.tenant_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found.")
    return case


def _load_subject(session: Session, tenant_id: str, subject_id: str) -> Subject:
    subject = session.get(Subject, subject_id)
    if subject is None or subject.tenant_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Subject not found.")
    return subject


def _resolve_or_create_subject(
    session: Session,
    principal: Principal,
    payload: EnrolRequest,
    case: Case | None,
) -> Subject:
    if payload.subject_id:
        return _load_subject(session, principal.tenant_id, payload.subject_id)

    if payload.external_ref.strip():
        existing = session.exec(
            select(Subject).where(
                Subject.tenant_id == principal.tenant_id,
                Subject.external_ref == payload.external_ref.strip(),
            )
        ).first()
        if existing is not None:
            return existing

    subject = Subject(
        tenant_id=principal.tenant_id,
        external_ref=payload.external_ref.strip(),
        display_name=payload.display_name.strip(),
        notes=payload.notes.strip(),
        case_id=case.id if case else None,
        enrolled_by=principal.id,
    )
    session.add(subject)
    session.flush()
    return subject


__all__ = ["router"]
