"""Hash-chain primitives, shared by the audit log and the custody chain.

Both are append-only tamper-evident logs with the same construction: each
record's hash covers its own content plus the previous record's hash, so editing
history breaks every hash after the edit and verification reports the first
divergence.

They are separate *logs* (the audit log answers "what did the system do?", the
custody chain answers "what happened to this artifact?") but identical
*machinery*, so the machinery lives here and neither imports the other.

Canonical serialisation matters more than it looks: if the same record hashed
differently between processes, every chain check would fail and the guarantee
would be worthless. Hence sorted keys, no whitespace, and explicit string
coercion for anything JSON cannot represent natively.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

GENESIS: str = "0" * 64
"""Predecessor hash of the first record in any chain."""


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Deterministic serialisation. Same content always yields the same bytes."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(payload: Mapping[str, Any]) -> str:
    """SHA256 over a record's canonical form, independent of any chain."""
    return sha256_hex(canonical_json(payload).encode("utf-8"))


def chain_hash(payload: Mapping[str, Any], previous_hash: str) -> str:
    """Link a record to its predecessor.

    The previous hash is appended to the canonical body rather than folded into
    it, so a record's own content hash stays computable without knowing its
    position in the chain.
    """
    material = canonical_json(payload) + previous_hash
    return sha256_hex(material.encode("utf-8"))


class ChainVerificationResult:
    """Outcome of re-walking a chain."""

    __slots__ = ("valid", "records", "broken_at", "reason")

    def __init__(self, valid: bool, records: int, broken_at: int | None, reason: str) -> None:
        self.valid = valid
        self.records = records
        self.broken_at = broken_at
        self.reason = reason

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "records": self.records,
            "broken_at": self.broken_at,
            "reason": self.reason,
        }

    def __repr__(self) -> str:
        return f"<ChainVerificationResult valid={self.valid} records={self.records}>"


def verify_chain(
    records: Iterable[Mapping[str, Any]],
    *,
    body_fields: tuple[str, ...],
    previous_field: str = "previous_hash",
    hash_field: str = "entry_hash",
) -> ChainVerificationResult:
    """Re-walk a chain and report the first record that fails.

    ``body_fields`` names the columns covered by the hash. Anything outside it —
    surrogate keys, denormalised joins added for display — is excluded, so a
    read-model change cannot invalidate a historical chain.
    """
    previous = GENESIS
    count = 0

    for index, record in enumerate(records):
        count = index + 1

        if record.get(previous_field) != previous:
            return ChainVerificationResult(
                False, index, index,
                "previous_hash does not match the preceding record — a record was "
                "inserted, removed, or reordered",
            )

        try:
            body = {field: record[field] for field in body_fields}
        except KeyError as exc:
            return ChainVerificationResult(
                False, index, index, f"record is missing hashed field {exc.args[0]!r}"
            )

        expected = chain_hash(body, previous)
        if expected != record.get(hash_field):
            return ChainVerificationResult(
                False, index, index,
                "entry_hash does not match the record contents — this record was edited",
            )

        previous = record[hash_field]

    return ChainVerificationResult(True, count, None, "")


__all__ = [
    "GENESIS",
    "ChainVerificationResult",
    "canonical_json",
    "chain_hash",
    "content_hash",
    "sha256_hex",
    "verify_chain",
]
