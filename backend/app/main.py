<<<<<<< HEAD
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.db.database import create_db_and_tables

# Import routers
from app.api import (
    routes_health,
    routes_projects,
    routes_brief,
    routes_concepts,
    routes_images,
    routes_feedback,
    routes_settings,
    routes_export,
    routes_billing,
    routes_auth
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Sentry if DSN is configured
if settings.SENTRY_DSN:
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
        )
        logger.info("Sentry initialized successfully.")
    except Exception as e:
        logger.warning(f"Failed to initialize Sentry: {e}")

# Ensure static dir exists before StaticFiles mount
os.makedirs("static/generated_images", exist_ok=True)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered creative direction assistant for designers. One imagination. Ten creative directions.",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# Set CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
=======
from __future__ import annotations

import base64

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .ai_engine import analyze_imatch_request, parse_checks
from .schemas import ImatchResponse
from nexgen_engine.api.service import EngineService
from nexgen_engine.api.routes import build_engine_router

app = FastAPI(
    title="NexGen Identity AI Service",
    version="0.1.0",
    description=(
        "Commercial biometric AI service for consent-aware demo face analysis, "
        "quality scoring, liveness scoring, and tenant-scoped demo matching."
    ),
)

engine_service = EngineService(audit_path="runtime/nexgen_audit.jsonl")
app.include_router(build_engine_router(engine_service))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
>>>>>>> 4be368fcc5342fef2e8a88b1f5104b94ffb43690
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD
# Database initialization on startup
@app.on_event("startup")
def on_startup():
    logger.info("Initializing database and tables...")
    create_db_and_tables()
    logger.info("Database initialized successfully.")

# Include routers
app.include_router(routes_health.router, prefix=settings.API_V1_STR)
app.include_router(routes_projects.router, prefix=settings.API_V1_STR)
app.include_router(routes_brief.router, prefix=settings.API_V1_STR)
app.include_router(routes_concepts.router, prefix=settings.API_V1_STR)
app.include_router(routes_images.router, prefix=settings.API_V1_STR)
app.include_router(routes_feedback.router, prefix=settings.API_V1_STR)
app.include_router(routes_settings.router, prefix=settings.API_V1_STR)
app.include_router(routes_export.router, prefix=settings.API_V1_STR)
app.include_router(routes_billing.router, prefix=settings.API_V1_STR)
app.include_router(routes_auth.router, prefix=settings.API_V1_STR)

# Serve static generated images
app.mount("/static", StaticFiles(directory="static", html=False), name="static")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to SIFS Imagination Expander AI API Backend.",
        "documentation": "/docs"
=======

@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "nexgen-identity-ai"}


@app.post("/api/imatch/search", response_model=ImatchResponse)
async def imatch_search(request: Request) -> ImatchResponse:
    parsed = await _read_imatch_request(request)
    image_bytes = parsed["image_bytes"]

    response = analyze_imatch_request(
        image_bytes=image_bytes,
        source_url=parsed["source_url"],
        mode=parsed["mode"],
        checks=parse_checks(parsed["checks"]),
        purpose=parsed["purpose"],
        lawful_use_reason=parsed["lawful_use_reason"],
    )

    if "image_or_source_url_required" in response.reasons:
        raise HTTPException(status_code=400, detail="Upload an image or provide a secure source URL.")

    return response


async def _read_imatch_request(request: Request) -> dict[str, object]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        data = await request.json()
        image_bytes = None
        if data.get("image_base64"):
            try:
                image_bytes = base64.b64decode(str(data["image_base64"]), validate=True)
            except Exception as exc:
                raise HTTPException(status_code=400, detail="image_base64 is invalid.") from exc
        return {
            "mode": str(data.get("mode", "single")),
            "purpose": data.get("purpose"),
            "lawful_use_reason": data.get("lawful_use_reason"),
            "checks": data.get("checks"),
            "source_url": data.get("source_url"),
            "image_bytes": image_bytes,
        }

    try:
        form = await request.form()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=415,
            detail="Multipart upload support requires python-multipart; send JSON with image_base64 or install backend requirements.",
        ) from exc
    image = form.get("image")
    image_bytes = None
    if image is not None and hasattr(image, "read"):
        image_bytes = await image.read()
        if len(image_bytes) > 12 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image must be 12MB or smaller.")
    return {
        "mode": str(form.get("mode") or "single"),
        "purpose": form.get("purpose"),
        "lawful_use_reason": form.get("lawful_use_reason"),
        "checks": form.get("checks"),
        "source_url": form.get("source_url"),
        "image_bytes": image_bytes,
>>>>>>> 4be368fcc5342fef2e8a88b1f5104b94ffb43690
    }
