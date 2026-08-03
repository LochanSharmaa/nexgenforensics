"""Time, injected rather than called.

Every timestamp in this platform is evidence — collection times, custody
sequences, retention expiry, chain ordering. Code that calls
``datetime.now()`` directly cannot be tested for time-dependent behaviour
(has retention expired? is this the earliest appearance?) without sleeping or
monkeypatching, so time arrives through a port.

All timestamps are timezone-aware UTC. A naive datetime is rejected at the
boundary rather than being silently assumed local — for an artifact collected in
one timezone and reviewed in another, that assumption is a defect.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    """Real time. The production implementation."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    """Fixed time for tests, advanceable on demand."""

    def __init__(self, at: datetime | None = None) -> None:
        self._now = at or datetime(2026, 1, 1, tzinfo=UTC)
        require_utc(self._now)

    def now(self) -> datetime:
        return self._now

    def advance(self, **delta: float) -> datetime:
        self._now += timedelta(**delta)
        return self._now

    def set(self, at: datetime) -> None:
        require_utc(at)
        self._now = at


def utcnow() -> datetime:
    """Convenience for code with no injected clock. Prefer the port."""
    return datetime.now(UTC)


def require_utc(value: datetime) -> datetime:
    """Reject naive datetimes at the boundary."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            "Naive datetime rejected. Every timestamp in this platform is evidence "
            "and must carry an explicit timezone."
        )
    return value.astimezone(UTC)


def isoformat(value: datetime) -> str:
    return require_utc(value).isoformat(timespec="milliseconds")


def coerce_utc(value: datetime) -> datetime:
    """Interpret a naive datetime as UTC instead of rejecting it.

    For the persistence boundary only. Some backends — SQLite notably, but also
    any driver configured without timezone support — drop `tzinfo` on the way
    back out even when a timezone-aware value was written. Since this platform
    only ever *writes* UTC, reading a naive value back as UTC is a faithful
    round-trip rather than an assumption.

    This distinction matters for the hash chains: if a timestamp serialised
    differently after a round-trip, every chain verification would fail and the
    tamper-evidence guarantee would be worthless.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def chain_timestamp(value: datetime) -> str:
    """Canonical timestamp form used inside hash-chain bodies.

    Must produce identical output for a value just constructed and for the same
    value read back from storage.
    """
    return coerce_utc(value).isoformat(timespec="microseconds")


__all__ = [
    "Clock",
    "FrozenClock",
    "SystemClock",
    "chain_timestamp",
    "coerce_utc",
    "isoformat",
    "require_utc",
    "utcnow",
]
