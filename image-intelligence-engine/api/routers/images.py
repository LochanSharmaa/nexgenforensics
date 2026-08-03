"""Image upload and retrieval.

Stage 1 of the pipeline, reachable directly so an investigator can add a probe
before starting a run. Everything computed is a property of the file — hashes,
dimensions, EXIF. No facial analysis happens or could: the dependency graph
contains no model with facial semantics (ARCHITECTURE §14).

The upload also opens the image's chain of custody. That first record is the
anchor everything later derives from — a screenshot, a report inclusion, an
export — so it is written in the same transaction as the row itself.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Request, UploadFile, status
from fastapi.responses import Response

from image_discovery.ingest import ingest
from shared.enums import ActorKind, ArtifactType, CustodyAction, ImageRole
from shared.errors import NotFoundError, ValidationError
from shared.logging import get_logger

from ..dependencies import (
    AuditRepoDep,
    CurrentUser,
    CustodyRepoDep,
    ImageRepoDep,
    InvestigationRepoDep,
    ObjectStoreDep,
    SettingsDep,
    client_label,
)
from ..schemas import ImageResponse, ImageUploadResponse

logger = get_logger(__name__)
router = APIRouter(tags=["images"])


async def _owned(investigations, investigation_id: uuid.UUID, user):  # noqa: ANN001
    investigation = await investigations.get(investigation_id)
    if investigation.owner_id != user.id:
        raise NotFoundError(f"Investigation {investigation_id} not found.")
    return investigation


@router.post(
    "/investigations/{investigation_id}/images",
    response_model=ImageUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_image(
    investigation_id: uuid.UUID,
    request: Request,
    user: CurrentUser,
    settings: SettingsDep,
    investigations: InvestigationRepoDep,
    images: ImageRepoDep,
    custody: CustodyRepoDep,
    audit: AuditRepoDep,
    store: ObjectStoreDep,
    file: UploadFile = File(...),
) -> ImageUploadResponse:
    """Upload a probe image.

    The content type is sniffed from the bytes, never trusted from the request
    header — a `.jpg` filename with a PDF inside is refused.
    """
    investigation = await _owned(investigations, investigation_id, user)
    raw = await file.read()

    try:
        ingested = ingest(raw)
    except ValidationError as exc:
        await audit.record(
            action="image.upload", outcome="refused",
            investigation_id=investigation_id, actor_id=user.id,
            actor_label=client_label(request, user),
            detail={"reason": str(exc), "filename": file.filename or ""},
        )
        await audit.session.commit()
        raise

    # Identical bytes are a re-upload, not a second probe. Creating another row
    # would double-count the image in every later stage.
    existing = await images.find_by_sha256(investigation_id, ingested.sha256)
    if existing is not None:
        return ImageUploadResponse(
            image=ImageResponse.model_validate(existing), deduplicated=True
        )

    storage_key = store.put(
        raw, prefix=f"images/{investigation_id}", extension=ingested.image_format.lower()
    )
    image = await images.add(
        investigation_id=investigation_id,
        role=ImageRole.PROBE,
        sha256=ingested.sha256,
        phash=ingested.phash,
        dhash=ingested.dhash,
        whash=ingested.whash,
        width=ingested.width,
        height=ingested.height,
        file_size=ingested.file_size,
        mime_type=ingested.mime_type,
        exif=ingested.exif,
        storage_key=storage_key,
    )

    await custody.record(
        investigation_id=investigation_id,
        artifact_type=ArtifactType.IMAGE,
        artifact_id=image.id,
        action=CustodyAction.COLLECTED,
        content_hash=ingested.sha256,
        actor_id=user.id,
        actor_kind=ActorKind.HUMAN,
        source_uri=f"upload:{file.filename or 'unnamed'}",
        storage_location=storage_key,
        transformation={
            "tool": "image_discovery.ingest",
            "format": ingested.image_format,
            "dimensions": f"{ingested.width}x{ingested.height}",
        },
    )

    await audit.record(
        action="image.upload", outcome="stored",
        investigation_id=investigation_id, actor_id=user.id,
        actor_label=client_label(request, user),
        lawful_basis=investigation.lawful_basis,
        resource_type="image", resource_id=str(image.id),
        detail={
            "sha256": ingested.sha256,
            "phash": ingested.phash,
            "format": ingested.image_format,
            "bytes": ingested.file_size,
            # Surfaced because location metadata is the most privacy-sensitive
            # thing an upload can carry, and an operator should know it is there.
            "has_gps_exif": ingested.has_gps,
        },
    )
    logger.info(
        "image.uploaded", investigation_id=str(investigation_id),
        image_id=str(image.id), has_gps=ingested.has_gps,
    )
    return ImageUploadResponse(
        image=ImageResponse.model_validate(image), deduplicated=False
    )


@router.get(
    "/investigations/{investigation_id}/images", response_model=list[ImageResponse]
)
async def list_images(
    investigation_id: uuid.UUID,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    images: ImageRepoDep,
    role: str | None = None,
) -> list[ImageResponse]:
    await _owned(investigations, investigation_id, user)
    rows = await images.list_for_investigation(investigation_id, role=role)
    return [ImageResponse.model_validate(row) for row in rows]


@router.get("/images/{image_id}", response_model=ImageResponse)
async def get_image(
    image_id: uuid.UUID,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    images: ImageRepoDep,
) -> ImageResponse:
    image = await images.get(image_id)
    await _owned(investigations, image.investigation_id, user)
    return ImageResponse.model_validate(image)


@router.get("/images/{image_id}/content")
async def image_content(
    image_id: uuid.UUID,
    user: CurrentUser,
    investigations: InvestigationRepoDep,
    images: ImageRepoDep,
    store: ObjectStoreDep,
) -> Response:
    """The image bytes.

    Served through the API rather than a public URL so ownership is checked on
    every read — evidence must not become reachable by anyone who guesses a key.
    """
    image = await images.get(image_id)
    await _owned(investigations, image.investigation_id, user)
    return Response(
        content=store.get(image.storage_key),
        media_type=image.mime_type or "application/octet-stream",
        headers={"Cache-Control": "private, max-age=3600"},
    )


__all__ = ["router"]
