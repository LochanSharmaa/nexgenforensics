from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from .core.csrf import CSRF_COOKIE, CSRF_HEADER, request_is_exempt, tokens_match, validate_csrf_token
from fastapi.responses import JSONResponse

from .api.routes import (
    account,
    admin,
    audit,
    auth,
    cases,
    enhancement,
    examination,
    health,
    reports,
    search,
    subjects,
)
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

    if settings.single_user:
        # Deliberate product rule, not an oversight: iMATCH is an individual
        # tool. Stated at startup so a future multi-user operator reading the
        # log knows to set NEXGEN_SINGLE_USER=false.
        logger.warning(
            "Single-user mode: requests without credentials act as the local "
            "owner account. Set NEXGEN_SINGLE_USER=false for multi-user deployments."
        )

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
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=(), "
            "interest-cohort=()"
        )
        # Isolate the browsing context so a cross-origin opener cannot reach
        # this window, and so no other site can embed a response as a resource.
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["X-DNS-Prefetch-Control"] = "off"

        # Content-Security-Policy.
        #
        # This service returns JSON, not markup, so the strictest possible
        # policy is also the correct one: nothing should ever be loaded or
        # executed from an API response. `frame-ancestors 'none'` is the
        # modern form of X-Frame-Options and is kept alongside it because
        # older browsers honour only the header.
        #
        # /docs is the exception and gets a relaxed policy rather than none:
        # Swagger UI is served from a CDN and uses inline styles, so the strict
        # policy would leave an interactive page that silently renders blank.
        # It is unavailable in production anyway (docs_url is None there).
        if request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self' data: https:; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
            )
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; "
                "form-action 'none'; sandbox"
            )

        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response

    @app.middleware("http")
    async def csrf_guard(request: Request, call_next):  # noqa: ANN001, ANN202
        """Reject state-changing requests that rely on cookies without a token.

        Requests presenting Authorization or X-API-Key are exempt: a
        cross-origin page cannot set those headers, so they are not forgeable
        and demanding a token would break every API client to no benefit. What
        remains is the unauthenticated auth endpoints, where this prevents
        login CSRF -- an attacker forcing a victim into an account they control
        so the victim's later searches are audited against the wrong person.
        """
        if settings.csrf_enabled:
            has_auth_header = bool(
                request.headers.get("authorization") or request.headers.get("x-api-key")
            )
            if not request_is_exempt(request.method, has_auth_header):
                cookie = request.cookies.get(CSRF_COOKIE)
                header = request.headers.get(CSRF_HEADER)
                secret = settings.resolved_jwt_secret()
                if not tokens_match(cookie, header) or not validate_csrf_token(header, secret):
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={
                            "detail": "Missing or invalid CSRF token.",
                            "hint": f"GET /api/auth/csrf, then send {CSRF_HEADER}.",
                        },
                    )
        return await call_next(request)

    # Registered LAST so it is the OUTERMOST middleware.
    #
    # Starlette runs the most recently added middleware first, and that
    # ordering is load-bearing here: with CORS on the inside, a 403 from the
    # CSRF guard short-circuits before CORS can attach its headers, and the
    # browser reports a legitimate rejection as an opaque "Failed to fetch"
    # with no way for the client to see why. Outermost, every response —
    # including refusals raised by middleware — carries the CORS headers.
    app.add_middleware(
        CORSMiddleware,
        # Never "*": these endpoints carry biometric data behind credentialed
        # requests, and a wildcard origin with credentials is both invalid and
        # dangerous.
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-CSRF-Token"],
        max_age=600,
    )

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
    app.include_router(account.router)
    app.include_router(cases.router)
    app.include_router(reports.router)
    app.include_router(examination.router)
    app.include_router(subjects.router)
    app.include_router(search.router)
    app.include_router(enhancement.router)
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
