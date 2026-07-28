from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import admin, audit, auth, cases, health, reports, search, subjects
from .core.config import get_settings
from .db.session import init_database
from .services.engine_service import get_engine_service

logger = logging.getLogger(__name__)

DESCRIPTION = """
NexGen iMATCH -- facial recognition for forensic investigation.

**What this system does:** given a probe image, it ranks visually similar faces
from a gallery your organisation enrolled, and records who searched for what, when,
and on what stated authority.

**What it does not do:** it does not identify people. A similarity score is not a
probability that two images show the same person. Every result is an investigative
lead requiring examiner verification before it is relied upon.

The service will not start without real recognition weights. There is no fallback
mode: a substitute embedding would produce numbers that look like similarity
scores and mean nothing, which is worse in an investigation than an outage.
Inspect `GET /api/imatch/engine/status` for the loaded model, device, and
thresholds actually in effect.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    init_database()
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    settings.audit_path.parent.mkdir(parents=True, exist_ok=True)

    engine = get_engine_service()
    # Load models during startup rather than inside the first user request, which
    # would otherwise hang for the seconds an ONNX pack takes to load. A failure
    # here aborts startup by design: the service must not accept biometric
    # searches it cannot actually perform.
    engine.warm_up()

    info = engine.runtime.recognizer.info
    logger.info(
        "iMATCH ready: %s (%s), %s-d templates on %s via %s.",
        info.model_pack,
        info.recognition_network,
        info.embedding_dim,
        engine.runtime.device,
        ", ".join(info.providers),
    )

    yield

    logger.info("iMATCH shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="NexGen iMATCH",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )

    app.add_middleware(
        CORSMiddleware,
        # Never "*": these endpoints carry biometric data behind credentialed
        # requests, and a wildcard origin with credentials is both invalid and
        # dangerous.
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        max_age=600,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # noqa: ANN001, ANN202
        """Attach a correlation id and baseline security headers."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        # API responses containing biometric findings must not sit in a shared
        # or browser cache.
        response.headers["Cache-Control"] = "no-store"
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        """Log the detail, return none of it.

        Stack traces and driver errors routinely leak schema, file paths, and
        occasionally credentials. The request id is the bridge between what the
        caller sees and what the operator can look up.
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("Unhandled error on %s %s [%s]", request.method, request.url.path, request_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error.", "request_id": request_id},
        )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(cases.router)
    app.include_router(reports.router)
    app.include_router(subjects.router)
    app.include_router(search.router)
    app.include_router(audit.router)
    app.include_router(admin.router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": "NexGen iMATCH",
            "version": "1.0.0",
            "documentation": "/docs" if not settings.is_production else "disabled in production",
            "health": "/api/health",
        }

    return app


app = create_app()


__all__ = ["app", "create_app"]
