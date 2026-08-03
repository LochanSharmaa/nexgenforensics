"""arq worker.

Owns the heavy dependencies — Playwright, Tesseract, spaCy — which is why it is
a separate image from the api. Phase 2 ships the skeleton and a health task; the
stage runners arrive with the pipeline in Phases 6 onward.

The worker also hosts the scheduler for image-monitoring sweeps (REVISION_3 §13),
registered here as a cron job once monitoring lands in Phase 13.
"""

from __future__ import annotations

from typing import Any

from arq import cron
from arq.connections import RedisSettings

from database.session import dispose_engine, session_scope
from shared.config import get_settings
from shared.logging import configure_logging, get_logger

logger = get_logger(__name__)


async def ping(ctx: dict[str, Any]) -> dict[str, Any]:
    """Round-trip task proving the worker can reach the database.

    Deployment smoke test: if this fails, the worker is running but useless, and
    that is worth knowing before a pipeline run discovers it.
    """
    from sqlalchemy import text

    async with session_scope() as session:
        result = await session.execute(text("SELECT 1"))
        value = result.scalar_one()

    logger.info("worker.ping", job_id=ctx.get("job_id"), database=value == 1)
    return {"ok": value == 1, "job_id": ctx.get("job_id")}


async def heartbeat(ctx: dict[str, Any]) -> None:
    """Periodic liveness marker, so an idle worker is distinguishable from a
    wedged one in the logs."""
    logger.debug("worker.heartbeat")


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    logger.info(
        "worker.startup", version=settings.app_version, environment=settings.environment
    )


async def shutdown(ctx: dict[str, Any]) -> None:
    await dispose_engine()
    logger.info("worker.shutdown")


class WorkerSettings:
    """arq entry point: `arq worker.main.WorkerSettings`."""

    functions = [ping]
    cron_jobs = [cron(heartbeat, minute=None, second={0, 30}, run_at_startup=False)]
    on_startup = startup
    on_shutdown = shutdown

    # Bounded so a large investigation cannot saturate the host. Per-domain
    # politeness is enforced separately in the crawler; this is the global cap.
    max_jobs = 8
    job_timeout = 900          # 15 min: a slow page render must not wedge a slot
    keep_result = 3600
    max_tries = 3
    health_check_interval = 30

    @staticmethod
    def redis_settings() -> RedisSettings:
        return RedisSettings.from_dsn(get_settings().redis_url)


__all__ = ["WorkerSettings", "ping"]
