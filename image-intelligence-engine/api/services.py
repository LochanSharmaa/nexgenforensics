"""Application services.

Routers hold HTTP shape only; anything that touches infrastructure lives here.
This split exists because the architecture test forbids routers from importing
SQLAlchemy or a Redis client — a router that builds a query is the first step
toward logic only the HTTP layer can execute, which the worker then cannot reuse.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import Settings
from shared.logging import get_logger

from .schemas import ComponentHealth

logger = get_logger(__name__)


async def check_database(session: AsyncSession) -> ComponentHealth:
    """Round-trip a trivial query.

    A probe must report, never raise: a readiness endpoint that 500s tells an
    orchestrator nothing about *which* dependency is down.
    """
    try:
        await session.execute(text("SELECT 1"))
        return ComponentHealth(name="database", healthy=True)
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth(
            name="database", healthy=False, detail=f"{type(exc).__name__}: {exc}"
        )


async def check_redis(settings: Settings) -> ComponentHealth:
    try:
        from redis.asyncio import from_url

        client = from_url(settings.redis_url)
        try:
            await client.ping()
            return ComponentHealth(name="redis", healthy=True)
        finally:
            await client.aclose()
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth(
            name="redis", healthy=False, detail=f"{type(exc).__name__}: {exc}"
        )


async def collect_health(session: AsyncSession, settings: Settings) -> list[ComponentHealth]:
    components = [await check_database(session), await check_redis(settings)]
    unhealthy = [c.name for c in components if not c.healthy]
    if unhealthy:
        logger.warning("health.not_ready", failed=unhealthy)
    return components


__all__ = ["check_database", "check_redis", "collect_health"]
