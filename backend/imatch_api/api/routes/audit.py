from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlmodel import Session, select

from ...core.config import Settings, get_settings
from ...core.dependencies import Principal, get_current_principal, require_admin
from ...db.models import AuditRecord, EnhancementRun, SearchRun, Template
from ...db.session import get_session
from ...services.audit_service import (
    ACTION_ENHANCE,
    ACTION_ENROL,
    ACTION_SEARCH,
    ACTION_VERIFY,
    AuditService,
)
from ...services.storage_service import StorageService
from ..schemas import AuditImageRef, AuditRecordResponse, ChainVerificationResponse
from .search import get_storage

router = APIRouter(prefix="/api/audit", tags=["audit"])

_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}


def _detail_dict(record: AuditRecord) -> dict:
    try:
        parsed = json.loads(record.detail or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _image_sources(record: AuditRecord, session: Session) -> list[tuple[str, str, str, str]]:
    """Every stored image an audit entry references, as (key, label, sha256, path).

    One resolver shared by the list response and the image endpoint, so the
    thumbnails a client is offered and the bytes it can actually fetch never
    disagree. Verify entries carry their paths in the chained detail; search,
    enhancement and enrolment entries point at rows that hold theirs.
    """
    detail = _detail_dict(record)
    sources: list[tuple[str, str, str, str]] = []

    def add(key: str, label: str, sha256: str | None, path: str | None) -> None:
        if path:
            sources.append((key, label, sha256 or "", path))

    if record.action == ACTION_VERIFY:
        add("reference", "Reference", detail.get("reference_sha256"), detail.get("reference_path"))
        add("probe", "Probe", detail.get("probe_sha256"), detail.get("probe_path"))
    elif record.action == ACTION_SEARCH:
        if detail.get("probe_path"):
            add("probe", "Probe", detail.get("probe_sha256"), detail.get("probe_path"))
        elif record.resource_type == "search_run" and record.resource_id:
            run = session.get(SearchRun, record.resource_id)
            if run is not None and run.tenant_id == record.tenant_id:
                add("probe", "Probe", run.probe_sha256, run.probe_path)
    elif record.action == ACTION_ENHANCE and record.resource_type == "enhancement_run" and record.resource_id:
        run = session.get(EnhancementRun, record.resource_id)
        if run is not None and run.tenant_id == record.tenant_id:
            add("original", "Original", run.original_sha256, run.original_path)
            add("enhanced", "Enhanced", run.enhanced_sha256, run.enhanced_path)
    elif record.action == ACTION_ENROL and record.resource_type == "template" and record.resource_id:
        template = session.get(Template, record.resource_id)
        if template is not None and template.tenant_id == record.tenant_id:
            add("image", "Enrolment image", template.image_sha256, template.image_path)

    return sources


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
    responses = []
    for record in records:
        response = AuditRecordResponse.model_validate(record)
        response.images = [
            AuditImageRef(key=key, label=label, sha256=sha256)
            for key, label, sha256, _path in _image_sources(record, session)
        ]
        responses.append(response)
    return responses


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


@router.get("/{record_id}/image/{key}")
def get_audit_image(
    record_id: str,
    key: str,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
    storage: StorageService = Depends(get_storage),
) -> Response:
    """Serve one of the images an audit entry references.

    The bytes are subject to the probe retention window; the sha256 in the
    entry's chained detail outlives them, so an expired image is a 404 here
    while the record of *which* image it was remains verifiable forever.
    """
    record = session.get(AuditRecord, record_id)
    if record is None or record.tenant_id != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Audit record not found.")

    match = next(
        (source for source in _image_sources(record, session) if source[0] == key),
        None,
    )
    if match is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"This audit record does not reference an image named '{key}'.",
        )

    _key, _label, sha256, path = match
    payload = storage.read(path)
    if payload is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "The image is no longer in storage; its retention window has likely expired.",
        )

    extension = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return Response(
        content=payload,
        media_type=_MEDIA_TYPES.get(extension, "application/octet-stream"),
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Image-Key": key,
            "X-Image-SHA256": sha256,
        },
    )


__all__ = ["router"]
