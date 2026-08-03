"""Liveness, readiness and metrics.

`/health` is liveness — the process is up. `/health/ready` actually touches
Postgres and Redis, because a container reporting healthy while its database is
unreachable will happily receive traffic it cannot serve. Docker Compose depends
on the readiness probe, not the liveness one.

Infrastructure access lives in `api.services`; this module holds HTTP shape only.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from shared import metrics

from ..dependencies import SessionDep, SettingsDep
from ..schemas import HealthResponse
from ..services import collect_health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep) -> HealthResponse:
    """Liveness. Deliberately touches no dependency."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        components=[],
    )


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(
    settings: SettingsDep, session: SessionDep, response: Response
) -> HealthResponse:
    """Readiness. Verifies the dependencies a real request would need."""
    components = await collect_health(session, settings)
    healthy = all(component.healthy for component in components)

    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if healthy else "degraded",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        components=components,
    )


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(settings: SettingsDep) -> Response:
    if not settings.metrics_enabled:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)


__all__ = ["router"]
