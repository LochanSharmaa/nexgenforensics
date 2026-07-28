from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from sqlmodel import Session, select

from ..db.models import AuditRecord, utcnow

logger = logging.getLogger(__name__)

# Actions worth reconstructing months later, when someone asks who searched for
# whom and on what authority.
ACTION_LOGIN = "auth.login"
ACTION_LOGIN_FAILED = "auth.login_failed"
ACTION_LOGOUT = "auth.logout"
ACTION_SEARCH = "biometric.search"
ACTION_VERIFY = "biometric.verify"
ACTION_ENROL = "biometric.enrol"
ACTION_TEMPLATE_DELETE = "biometric.template_delete"
ACTION_SUBJECT_DELETE = "biometric.subject_delete"
ACTION_ADJUDICATE = "case.adjudicate"
ACTION_CASE_CREATE = "case.create"
ACTION_CASE_UPDATE = "case.update"
ACTION_EXPORT = "case.export"
ACTION_USER_CREATE = "admin.user_create"
ACTION_KEY_CREATE = "admin.api_key_create"
ACTION_KEY_REVOKE = "admin.api_key_revoke"


@dataclass(frozen=True)
class ChainVerification:
    valid: bool
    checked: int
    broken_at: str | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "records_checked": self.checked,
            "broken_at": self.broken_at,
            "reason": self.reason,
        }


class AuditService:
    """Append-only, hash-chained audit trail.

    Each record's hash covers its own content plus the previous record's hash,
    per tenant. Altering or deleting any record invalidates every hash after it,
    so tampering is detectable even by someone with write access to the database.

    Records are also mirrored to a JSONL file. If the database is rewritten
    wholesale the file still holds the original chain, and vice versa; an
    attacker has to compromise both consistently.

    What this does and does not give you: it proves the log has not been edited
    since it was written. It does not prove the log is complete, since an
    attacker with database access could truncate the tail. Ship the JSONL to
    write-once storage if you need that too.
    """

    def __init__(self, log_path: Path | str | None = None) -> None:
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        session: Session,
        *,
        tenant_id: str,
        action: str,
        actor_id: str = "",
        actor_label: str = "",
        resource_type: str = "",
        resource_id: str = "",
        outcome: str = "success",
        lawful_basis: str = "",
        detail: dict[str, Any] | None = None,
        ip_address: str = "",
        user_agent: str = "",
    ) -> AuditRecord:
        detail_json = json.dumps(detail or {}, sort_keys=True, default=str)

        # Serialize chain extension: two concurrent searches reading the same
        # previous hash would otherwise fork the chain and both appear valid
        # individually while the sequence is wrong.
        with self._lock:
            latest = self._latest(session, tenant_id)
            previous_hash = latest.entry_hash if latest else ""
            sequence = (latest.sequence + 1) if latest else 1
            timestamp = utcnow()
            body = {
                "sequence": sequence,
                "tenant_id": tenant_id,
                "actor_id": actor_id,
                "actor_label": actor_label,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "outcome": outcome,
                "lawful_basis": lawful_basis,
                "detail": detail_json,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "previous_hash": previous_hash,
                "created_at": _canonical_timestamp(timestamp),
            }
            entry_hash = _digest(body)

            record = AuditRecord(
                tenant_id=tenant_id,
                sequence=sequence,
                actor_id=actor_id,
                actor_label=actor_label,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                outcome=outcome,
                lawful_basis=lawful_basis,
                detail=detail_json,
                ip_address=ip_address,
                user_agent=user_agent,
                previous_hash=previous_hash,
                entry_hash=entry_hash,
                created_at=timestamp,
            )
            session.add(record)
            session.flush()
            self._mirror({**body, "entry_hash": entry_hash, "id": record.id})

        return record

    def verify_chain(self, session: Session, tenant_id: str) -> ChainVerification:
        # Ordered by chain position, never by timestamp. See AuditRecord.sequence.
        records = session.exec(
            select(AuditRecord)
            .where(AuditRecord.tenant_id == tenant_id)
            .order_by(AuditRecord.sequence)
        ).all()

        previous_hash = ""
        for index, record in enumerate(records):
            if record.sequence != index + 1:
                return ChainVerification(
                    valid=False,
                    checked=index,
                    broken_at=record.id,
                    reason=(
                        f"Chain position {record.sequence} breaks the sequence; a record "
                        "before it was removed."
                    ),
                )
            if record.previous_hash != previous_hash:
                return ChainVerification(
                    valid=False,
                    checked=index,
                    broken_at=record.id,
                    reason="Previous-hash link does not match the preceding record.",
                )
            expected = _digest(
                {
                    "sequence": record.sequence,
                    "tenant_id": record.tenant_id,
                    "actor_id": record.actor_id,
                    "actor_label": record.actor_label,
                    "action": record.action,
                    "resource_type": record.resource_type,
                    "resource_id": record.resource_id,
                    "outcome": record.outcome,
                    "lawful_basis": record.lawful_basis,
                    "detail": record.detail,
                    "ip_address": record.ip_address,
                    "user_agent": record.user_agent,
                    "previous_hash": record.previous_hash,
                    "created_at": _canonical_timestamp(record.created_at),
                }
            )
            if expected != record.entry_hash:
                return ChainVerification(
                    valid=False,
                    checked=index,
                    broken_at=record.id,
                    reason="Record content does not match its stored hash.",
                )
            previous_hash = record.entry_hash

        return ChainVerification(valid=True, checked=len(records))

    def _latest(self, session: Session, tenant_id: str) -> AuditRecord | None:
        """Tail of this tenant's chain, by explicit position rather than time."""
        return session.exec(
            select(AuditRecord)
            .where(AuditRecord.tenant_id == tenant_id)
            .order_by(AuditRecord.sequence.desc())
            .limit(1)
        ).first()

    def _mirror(self, payload: dict[str, Any]) -> None:
        if not self.log_path:
            return
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        except OSError as exc:
            # A failed mirror must not abort the operation being audited, but it
            # is itself a security-relevant event.
            logger.error("Could not mirror audit record to %s: %s", self.log_path, exc)


def _canonical_timestamp(value: datetime) -> str:
    """Timestamp string used for hashing, identical on write and on read-back.

    The hash must be reproducible from a stored row, and databases do not
    round-trip timezone information faithfully: a value written as
    timezone-aware UTC comes back from SQLite (and from a naive PostgreSQL
    column) with ``tzinfo`` stripped, so ``isoformat()`` loses the ``+00:00``
    suffix and every hash fails to verify.

    Normalising both sides through UTC makes the digest depend on the instant
    rather than on how the driver happened to represent it.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _digest(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()


__all__ = [
    "ACTION_ADJUDICATE",
    "ACTION_CASE_CREATE",
    "ACTION_CASE_UPDATE",
    "ACTION_ENROL",
    "ACTION_EXPORT",
    "ACTION_KEY_CREATE",
    "ACTION_KEY_REVOKE",
    "ACTION_LOGIN",
    "ACTION_LOGIN_FAILED",
    "ACTION_LOGOUT",
    "ACTION_SEARCH",
    "ACTION_SUBJECT_DELETE",
    "ACTION_TEMPLATE_DELETE",
    "ACTION_USER_CREATE",
    "ACTION_VERIFY",
    "AuditService",
    "ChainVerification",
]
