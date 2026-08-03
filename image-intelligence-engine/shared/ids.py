"""UUIDv7 identifiers.

Time-ordered by construction (RFC 9562): the first 48 bits are a Unix
millisecond timestamp, so ids sort chronologically and index without the random
page splits that UUIDv4 causes on a B-tree. On evidence tables that only ever
append, that difference compounds.

A secondary benefit for this platform specifically: `ORDER BY id` on
observations is chronological, so an evidence chain renders in collection order
without a join or an extra index.

Python 3.11 has no `uuid.uuid7`, so it is implemented here rather than adding a
dependency for eighteen lines.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Protocol


def uuid7(when_ms: int | None = None) -> uuid.UUID:
    """Generate a UUIDv7.

    Layout: 48-bit timestamp | version(4) | 12-bit rand_a |
    variant(2) | 62-bit rand_b.
    """
    timestamp = int(time.time() * 1000) if when_ms is None else when_ms
    if timestamp < 0:
        raise ValueError("timestamp must not be negative")

    payload = bytearray(os.urandom(16))
    payload[0:6] = timestamp.to_bytes(6, "big")

    payload[6] = (payload[6] & 0x0F) | 0x70          # version 7
    payload[8] = (payload[8] & 0x3F) | 0x80          # RFC 4122 variant

    return uuid.UUID(bytes=bytes(payload))


def timestamp_ms_of(value: uuid.UUID) -> int:
    """Recover the embedded millisecond timestamp from a UUIDv7."""
    if value.version != 7:
        raise ValueError(f"expected a UUIDv7, got version {value.version}")
    return int.from_bytes(value.bytes[0:6], "big")


class IdGenerator(Protocol):
    """Injected so tests can be deterministic (ARCHITECTURE §4.2)."""

    def __call__(self) -> uuid.UUID: ...


def new_id() -> uuid.UUID:
    return uuid7()


class SequentialIdGenerator:
    """Deterministic generator for tests. Ids remain valid, sortable UUIDv7s."""

    def __init__(self, start_ms: int = 1_700_000_000_000) -> None:
        self._next_ms = start_ms

    def __call__(self) -> uuid.UUID:
        value = uuid7(self._next_ms)
        self._next_ms += 1
        return value


__all__ = ["IdGenerator", "SequentialIdGenerator", "new_id", "timestamp_ms_of", "uuid7"]
