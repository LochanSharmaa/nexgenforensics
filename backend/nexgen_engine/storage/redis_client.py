from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    count: int
    limit: int
    retry_after_seconds: int


class RedisClient:
    def __init__(self, url: str = "redis://localhost:6379/0", pool_size: int = 10) -> None:
        self.url = url
        self.pool_size = pool_size
        self._redis: Any = None
        self._memory_cache: dict[str, tuple[str, float | None]] = {}
        self._pubsub_messages: dict[str, list[str]] = {}

    @property
    def connected(self) -> bool:
        return self._redis is not None

    async def connect(self) -> None:
        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(
                self.url,
                max_connections=self.pool_size,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            await self._redis.ping()
        except Exception:
            self._redis = None

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
        self._redis = None

    async def get_json(self, key: str) -> dict[str, Any] | None:
        raw = await self.get(key)
        return json.loads(raw) if raw else None

    async def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        await self.set(key, json.dumps(value, sort_keys=True), ttl_seconds)

    async def get(self, key: str) -> str | None:
        if self._redis is not None:
            value = await self._redis.get(key)
            return str(value) if value is not None else None
        value = self._memory_cache.get(key)
        if value is None:
            return None
        payload, expires_at = value
        if expires_at is not None and expires_at <= time.time():
            self._memory_cache.pop(key, None)
            return None
        return payload

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        if self._redis is not None:
            await self._redis.set(key, value, ex=ttl_seconds)
            return
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        self._memory_cache[key] = (value, expires_at)

    async def hset_json(self, name: str, key: str, value: dict[str, Any]) -> None:
        payload = json.dumps(value, sort_keys=True)
        if self._redis is not None:
            await self._redis.hset(name, key, payload)
            return
        await self.set(f"{name}:{key}", payload)

    async def hget_json(self, name: str, key: str) -> dict[str, Any] | None:
        if self._redis is not None:
            payload = await self._redis.hget(name, key)
            return json.loads(payload) if payload else None
        return await self.get_json(f"{name}:{key}")

    async def publish(self, channel: str, payload: dict[str, Any]) -> None:
        message = json.dumps(payload, sort_keys=True)
        if self._redis is not None:
            await self._redis.publish(channel, message)
            return
        self._pubsub_messages.setdefault(channel, []).append(message)

    async def rate_limit(self, key: str, limit: int, window_seconds: int) -> RateLimitResult:
        now = time.time()
        bucket_key = f"rate:{key}"
        if self._redis is not None:
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(bucket_key, 0, now - window_seconds)
            pipe.zadd(bucket_key, {str(now): now})
            pipe.zcard(bucket_key)
            pipe.expire(bucket_key, window_seconds)
            _, _, count, _ = await pipe.execute()
            allowed = int(count) <= limit
            return RateLimitResult(allowed, int(count), limit, 0 if allowed else window_seconds)
        current = await self.get_json(bucket_key) or {"events": []}
        events = [float(item) for item in current.get("events", []) if float(item) > now - window_seconds]
        events.append(now)
        await self.set_json(bucket_key, {"events": events}, ttl_seconds=window_seconds)
        allowed = len(events) <= limit
        return RateLimitResult(allowed, len(events), limit, 0 if allowed else window_seconds)

