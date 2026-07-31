from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlmodel import Session

from ...core.config import Settings, get_settings
from ...core.dependencies import Principal, get_current_principal
from ...db.session import get_session
from ...services.engine_service import EngineService, get_engine_service
from ..schemas import EngineStatusResponse, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/api/health", response_model=HealthResponse)
def health(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    engine: EngineService = Depends(get_engine_service),
) -> HealthResponse:
    """Liveness and readiness in one call.

    Unauthenticated on purpose so load balancers can reach it. It exposes no
    tenant data and no configuration values beyond the environment name.
    """
    try:
        # A raw probe, so use execute() rather than SQLModel's exec(), which
        # expects a typed select statement.
        session.execute(text("SELECT 1"))
        database = "ok"
    except Exception as exc:  # pragma: no cover - depends on the database host
        logger.error("Database health check failed: %s", exc)
        database = "unavailable"

    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        version="1.0.0",
        environment=settings.env,
        database=database,
        recognition_capable=engine.recognition_capable,
    )


@router.get("/api/imatch/engine/status", response_model=EngineStatusResponse)
def engine_status(
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
    engine: EngineService = Depends(get_engine_service),
) -> EngineStatusResponse:
    """What the engine actually is right now.

    ``recognizer.recognition_capable`` is the field that matters: when it is
    false the service is running the deterministic stub and no score it returns
    means anything. The console surfaces this as a banner.
    """
    stats = engine.stats(session, principal.tenant_id)
    payload = engine.status()
    payload["gallery"] = {"templates": stats.templates, "subjects": stats.subjects}
    return EngineStatusResponse(**payload)


__all__ = ["router"]


@router.get("/api/imatch/engine/metrics")
def engine_metrics(
    principal: Principal = Depends(get_current_principal),
    engine: EngineService = Depends(get_engine_service),
) -> dict:
    """Per-stage latency percentiles for the running process.

    Authenticated deliberately. Timing data is a side channel: response times
    leak gallery size and whether a probe matched, so this sits behind the same
    auth as every other biometric endpoint rather than on an open /metrics path.

    Percentiles come from a bounded in-process window (see
    nexgen_engine/observability.py), so they describe THIS process since it
    started -- not a historical series and not a cluster. Restarting the
    service resets them.
    """
    from nexgen_engine.observability import LATENCY

    snapshot = LATENCY.snapshot()
    snapshot["model"] = engine.runtime.recognizer.info.as_dict()
    snapshot["device"] = engine.runtime.device
    return snapshot
