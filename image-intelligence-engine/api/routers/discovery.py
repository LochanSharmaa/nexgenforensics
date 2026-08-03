"""Discovery and findings.

`POST /discover` runs the configured providers for a probe image and persists
what they returned. `GET /findings` renders that as the investigator-facing
view: where the image was found, what the sources are, and — just as
importantly — which providers were never asked because they have no key.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request

from image_discovery.service import DiscoveryService
from providers.registry import manifests
from shared.errors import NotFoundError, ValidationError
from shared.logging import get_logger

from ..dependencies import (
    AuditRepoDep,
    ClockDep,
    CurrentUser,
    ImageRepoDep,
    InvestigationRepoDep,
    ObjectStoreDep,
    SessionDep,
    SettingsDep,
    client_label,
)
from ..schemas import DiscoverRequest, FindingsResponse, ProviderInfo

logger = get_logger(__name__)
router = APIRouter(tags=["discovery"])


async def _owned(investigations, investigation_id: uuid.UUID, user):  # noqa: ANN001
    investigation = await investigations.get(investigation_id)
    if investigation.owner_id != user.id:
        raise NotFoundError(f"Investigation {investigation_id} not found.")
    return investigation


@router.get("/providers", response_model=list[ProviderInfo])
async def list_providers(user: CurrentUser, settings: SettingsDep) -> list[ProviderInfo]:
    """Every discovery plugin and whether it can actually run.

    Reported before a search rather than after, so an investigator knows what
    coverage to expect instead of inferring it from an empty result.
    """
    return [ProviderInfo(**entry) for entry in manifests(settings)]


@router.post(
    "/investigations/{investigation_id}/discover", response_model=FindingsResponse
)
async def discover(
    investigation_id: uuid.UUID,
    payload: DiscoverRequest,
    request: Request,
    user: CurrentUser,
    settings: SettingsDep,
    session: SessionDep,
    clock: ClockDep,
    investigations: InvestigationRepoDep,
    images: ImageRepoDep,
    store: ObjectStoreDep,
    audit: AuditRepoDep,
) -> FindingsResponse:
    """Ask the configured providers where this image appears."""
    investigation = await _owned(investigations, investigation_id, user)

    probes = await images.list_for_investigation(investigation_id, role="PROBE")
    if not probes:
        raise ValidationError(
            "Upload a probe image before running discovery — there is nothing to search for."
        )
    image = (
        await images.get(payload.image_id)
        if payload.image_id
        else probes[0]
    )

    service = DiscoveryService(session, settings, clock)
    view = await service.discover(
        investigation_id=investigation_id,
        image=image,
        image_bytes=store.get(image.storage_key),
        manual_urls=payload.urls,
    )

    await audit.record(
        action="discovery.run",
        outcome=f"{len(view.findings)} appearance(s)",
        investigation_id=investigation_id,
        actor_id=user.id,
        actor_label=client_label(request, user),
        lawful_basis=investigation.lawful_basis,
        resource_type="image",
        resource_id=str(image.id),
        detail={
            "providers_run": view.providers_run,
            "providers_unconfigured": [p["name"] for p in view.providers_unconfigured],
            "manual_urls": len(payload.urls or []),
            "found": len(view.findings),
        },
    )
    return FindingsResponse(**view.as_dict())


@router.get(
    "/investigations/{investigation_id}/findings", response_model=FindingsResponse
)
async def findings(
    investigation_id: uuid.UUID,
    user: CurrentUser,
    settings: SettingsDep,
    session: SessionDep,
    clock: ClockDep,
    investigations: InvestigationRepoDep,
) -> FindingsResponse:
    """What has been found so far, without running anything new."""
    await _owned(investigations, investigation_id, user)
    view = await DiscoveryService(session, settings, clock).findings_for(investigation_id)
    return FindingsResponse(**view.as_dict())


__all__ = ["router"]
