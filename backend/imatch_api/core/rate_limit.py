from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after: float


class SlidingWindowRateLimiter:
    """Per-identity request limiter using a sliding window.

    Scope and limits: state lives in this process only. Behind multiple workers
    the effective limit multiplies by the worker count, and it resets on
    restart. That is acceptable as a guard against runaway clients and casual
    abuse; it is NOT a defence against a distributed attacker. For that, put a
    shared store (Redis) or an edge rate limiter in front, and treat this as
    defence in depth.

    A sliding window is used rather than fixed buckets because fixed windows let
    a client send 2x the limit across a boundary instant.
    """

    def __init__(self, max_events: int, window_seconds: float = 60.0) -> None:
        self.max_events = max(1, max_events)
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, identity: str) -> RateLimitResult:
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            timestamps = self._events.setdefault(identity, deque())
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.max_events:
                retry_after = max(0.0, timestamps[0] + self.window_seconds - now)
                return RateLimitResult(allowed=False, remaining=0, retry_after=round(retry_after, 2))

            timestamps.append(now)
            return RateLimitResult(
                allowed=True,
                remaining=self.max_events - len(timestamps),
                retry_after=0.0,
            )

    def reset(self, identity: str | None = None) -> None:
        with self._lock:
            if identity is None:
                self._events.clear()
            else:
                self._events.pop(identity, None)

    def prune(self) -> int:
        """Drop identities with no recent activity. Returns how many were freed.

        Without this the dictionary grows once per distinct caller and never
        shrinks, which is a slow memory leak on a long-running server.
        """
        cutoff = time.monotonic() - self.window_seconds
        with self._lock:
            stale = [key for key, events in self._events.items() if not events or events[-1] < cutoff]
            for key in stale:
                del self._events[key]
            return len(stale)


__all__ = ["RateLimitResult", "SlidingWindowRateLimiter"]


# Registry of the limiters guarding unauthenticated auth endpoints.
#
# They are process-global by necessity: there is no principal to key on before
# sign-in, so state has to live somewhere shared. That makes them stateful
# across a test session, where dozens of logins arrive from one client and
# would otherwise trip the limit and fail unrelated tests. `reset_auth_limiters`
# exists so a test fixture can clear that state without weakening the limits
# themselves -- the production numbers stay exactly as configured.
_AUTH_LIMITERS: list["SlidingWindowRateLimiter"] = []


def register_auth_limiter(limiter: "SlidingWindowRateLimiter") -> "SlidingWindowRateLimiter":
    _AUTH_LIMITERS.append(limiter)
    return limiter


def reset_auth_limiters() -> None:
    for limiter in _AUTH_LIMITERS:
        limiter.reset()
