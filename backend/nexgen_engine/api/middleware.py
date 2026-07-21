from __future__ import annotations

import time
from collections import defaultdict, deque


class InMemoryRateLimiter:
    def __init__(self, limit: int = 1000, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_id: str) -> bool:
        now = time.time()
        bucket = self.events[client_id]
        while bucket and now - bucket[0] > self.window_seconds:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True


class ApiKeyAuth:
    def __init__(self, allowed_keys: set[str] | None = None) -> None:
        self.allowed_keys = allowed_keys or set()

    def allow(self, api_key: str | None) -> bool:
        return not self.allowed_keys or bool(api_key and api_key in self.allowed_keys)
