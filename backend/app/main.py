from __future__ import annotations

import base64

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .ai_engine import analyze_imatch_request, parse_checks
from .schemas import ImatchResponse
from .forensic_report_routes import build_forensic_report_router
from nexgen_engine.api.service import EngineService
from nexgen_engine.api.routes import build_engine_router
from nexgen_engine.api.auth_routes import build_auth_router
from nexgen_engine.api.job_routes import build_job_router
from nexgen_engine.auth import AuthService
from nexgen_engine.jobs import JobQueue
from nexgen_engine.monitoring import MetricsRegistry
from nexgen_engine.settings import Settings
from nexgen_engine.storage import Database
from nexgen_engine.security.audit_logger import AuditLogger

app = FastAPI(
    title="NexGen Identity AI Service",
    version="0.1.0",
    description=(
        "Commercial biometric AI service for consent-aware demo face analysis, "
        "quality scoring, liveness scoring, and tenant-scoped demo matching."
    ),
)

settings = Settings.from_env()
metrics = MetricsRegistry()
database = Database(settings.runtime_dir / "nexgen.db")
database.migrate()
auth_service = AuthService(settings.secret_key)
job_queue = JobQueue(database)
engine_service = EngineService(
    audit_path=settings.runtime_dir / "nexgen_audit.jsonl",
    index_path=settings.faiss_index_path,
    database=database,
    template_secret=settings.encryption_key or settings.secret_key,
)
app.include_router(build_engine_router(engine_service))
app.include_router(build_auth_router(database, auth_service))
app.include_router(build_job_router(job_queue, auth_service))
app.include_router(build_forensic_report_router(auth_service, AuditLogger(settings.runtime_dir / 'nexgen_audit.jsonl'), settings.runtime_dir / 'reports'))

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health() -> dict[str, str | int | bool]:
    metrics.increment("health_requests")
    return {
        "status": "ok",
        "service": "nexgen-identity-ai",
        "database": settings.database_backend,
        "redis_configured": settings.redis_enabled,
        "index_count": int(engine_service.index.snapshot()["count"]),
    }


@app.get("/metrics")
def prometheus_metrics() -> str:
    return metrics.prometheus_text()


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
    }

