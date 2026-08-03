"""FastAPI application factory."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from structlog.contextvars import bind_contextvars

from database.session import dispose_engine
from shared.config import Settings, get_settings
from shared.errors import IIEError
from shared.logging import clear_context, configure_logging, get_logger

from .routers import (
    audit,
    auth,
    discovery,
    health,
    images,
    investigations,
    pipeline,
    retention,
    review,
    vision,
)

logger = get_logger(__name__)

PROBLEM_JSON = "application/problem+json"

DESCRIPTION = """
Image Intelligence & OSINT Investigation Platform.

An uploaded image is the entry point to a structured public-information
investigation. Every statement links back to the page it came from.

**This platform performs no facial recognition.** It does not identify people
from facial features, compare faces, or build biometric databases. Identity
information is reported only where a discovered public page explicitly publishes
it, with attribution.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    logger.info(
        "api.startup",
        version=settings.app_version,
        environment=settings.environment,
        database=settings.database_url.split("@")[-1],  # never log credentials
    )
    yield
    await dispose_engine()
    logger.info("api.shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=DESCRIPTION,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # noqa: ANN001, ANN202
        """Bind a request id so every log line from this request is correlatable."""
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        bind_contextvars(request_id=request_id, path=request.url.path, method=request.method)
        try:
            response = await call_next(request)
        finally:
            clear_context()
        response.headers["x-request-id"] = request_id
        return response

    # -- error handling ----------------------------------------------------

    @app.exception_handler(IIEError)
    async def handle_iie_error(request: Request, exc: IIEError) -> JSONResponse:
        """Typed errors become RFC 9457 problem documents.

        Clients match on the stable `type` slug, never on message text, so
        wording can improve without breaking consumers.
        """
        if exc.http_status >= 500:
            logger.error("request.failed", error=type(exc).__name__, detail=exc.message)
        else:
            logger.info("request.rejected", error=type(exc).__name__, detail=exc.message)
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.as_problem(instance=str(request.url.path)),
            media_type=PROBLEM_JSON,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "type": "https://iie.invalid/problems/request-validation",
                "title": "RequestValidationError",
                "status": 422,
                "detail": "The request body failed validation.",
                "errors": [
                    {"field": ".".join(str(p) for p in err["loc"][1:]), "message": err["msg"]}
                    for err in exc.errors()
                ],
                "instance": str(request.url.path),
            },
            media_type=PROBLEM_JSON,
        )

    # -- routes ------------------------------------------------------------

    app.include_router(health.router)
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(investigations.router, prefix="/api/v1")
    app.include_router(audit.router, prefix="/api/v1")
    app.include_router(discovery.router, prefix="/api/v1")
    app.include_router(images.router, prefix="/api/v1")
    app.include_router(pipeline.router, prefix="/api/v1")
    app.include_router(retention.router, prefix="/api/v1")
    app.include_router(review.router, prefix="/api/v1")
    app.include_router(vision.router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "image-intelligence-engine",
            "version": settings.app_version,
            "docs": "/api/docs",
            "health": "/health",
        }

    return app


app = create_app()

__all__ = ["app", "create_app"]
