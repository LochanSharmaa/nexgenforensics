"""Image analysis — the observation stage.

`POST /analyze` asks a vision model what is visible in a probe image and stores
every reading as an observation sourced to that image. `GET /analysis` returns
what was stored without spending another model call.

The separation this endpoint exists to hold: it produces **observations**, not
facts. "The sign reads MERIDIAN LOGISTICS" is checkable by looking at the
picture. "This is Meridian's head office" is a claim about the world and needs
pages to support it — which is what `/discover` and `/findings` are for.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request

from shared.enums import ObservationMethod
from shared.errors import NotFoundError, ValidationError
from shared.logging import get_logger
from vision.registry import build_provider
from vision.service import VisionService

from ..dependencies import (
    AuditRepoDep,
    ClockDep,
    CurrentUser,
    ImageRepoDep,
    InvestigationRepoDep,
    ObjectStoreDep,
    ObservationRepoDep,
    SessionDep,
    SettingsDep,
    client_label,
)
from ..schemas import VisionAnalysisResponse

logger = get_logger(__name__)
router = APIRouter(tags=["vision"])


async def _owned(investigations, investigation_id: uuid.UUID, user):  # noqa: ANN001
    investigation = await investigations.get(investigation_id)
    if investigation.owner_id != user.id:
        raise NotFoundError(f"Investigation {investigation_id} not found.")
    return investigation


@router.post(
    "/investigations/{investigation_id}/analyze", response_model=VisionAnalysisResponse
)
async def analyze_image(
    investigation_id: uuid.UUID,
    request: Request,
    user: CurrentUser,
    settings: SettingsDep,
    session: SessionDep,
    clock: ClockDep,
    investigations: InvestigationRepoDep,
    images: ImageRepoDep,
    store: ObjectStoreDep,
    audit: AuditRepoDep,
    image_id: uuid.UUID | None = None,
) -> VisionAnalysisResponse:
    """Read the image and record what is visible."""
    investigation = await _owned(investigations, investigation_id, user)

    if not settings.vision_enabled:
        raise ValidationError("Vision analysis is disabled (IIE_VISION_ENABLED=false).")

    probes = await images.list_for_investigation(investigation_id, role="PROBE")
    if not probes:
        raise ValidationError("Upload a probe image before running analysis.")
    image = await images.get(image_id) if image_id else probes[0]

    provider = build_provider(settings)
    service = VisionService(session, settings, provider, clock)
    result = await service.analyse(
        investigation_id=investigation_id,
        image=image,
        image_bytes=store.get(image.storage_key),
    )

    await audit.record(
        action="vision.analyze",
        outcome=(
            "unavailable"
            if not result.analysis.available
            else (
                "failed"
                if result.analysis.error
                else f"{len(result.observation_ids)} observations"
            )
        ),
        investigation_id=investigation_id,
        actor_id=user.id,
        actor_label=client_label(request, user),
        lawful_basis=investigation.lawful_basis,
        resource_type="image",
        resource_id=str(image.id),
        detail={
            "provider": result.analysis.provider,
            "model": result.analysis.model,
            "observations": len(result.observation_ids),
            "clues": len(result.clues),
            "people_present": result.analysis.people_present,
            # Refusals are audited. A model that keeps trying to identify people
            # should be visible in the trail, not only in a log line.
            "rejected": [r["rule"] for r in result.analysis.rejected],
            "error": result.analysis.error,
        },
    )
    return VisionAnalysisResponse(**result.as_dict())


@router.get(
    "/investigations/{investigation_id}/analysis", response_model=VisionAnalysisResponse
)
async def get_analysis(
    investigation_id: uuid.UUID,
    user: CurrentUser,
    settings: SettingsDep,
    clock: ClockDep,
    investigations: InvestigationRepoDep,
    observations: ObservationRepoDep,
) -> VisionAnalysisResponse:
    """Stored vision observations, without spending another model call.

    Rebuilt from `observations` rather than from a cached response object: the
    evidence table is the source of truth, so what this returns is exactly what
    a reviewer would find in the record.
    """
    await _owned(investigations, investigation_id, user)
    rows = await observations.by_method(investigation_id, ObservationMethod.VISION)

    by_category: dict[str, list[dict]] = {}
    for row in rows:
        # The category was written into the context snippet as "[CATEGORY] …";
        # parsing it back keeps the observations table free of a vision-only
        # column that nothing else would use.
        snippet = row.context_snippet or ""
        category = "VISUAL_CLUE"
        detail = snippet
        if snippet.startswith("["):
            closing = snippet.find("]")
            if closing > 0:
                category = snippet[1:closing]
                detail = snippet[closing + 1 :].strip()
        by_category.setdefault(category, []).append(
            {
                "category": category,
                "value": row.raw_value,
                "detail": detail,
                "confidence": row.method_confidence or 0.0,
                "verbatim": "not verbatim" not in snippet,
            }
        )

    model = ""
    if rows:
        # `vision@0.1.0:gemini-2.0-flash` — the model that produced these.
        version = rows[0].extractor_version
        model = version.split(":", 1)[1] if ":" in version else version

    return VisionAnalysisResponse(
        provider="gemini",
        model=model,
        available=True,
        observation_count=len(rows),
        by_category=by_category,
    )


__all__ = ["router"]
