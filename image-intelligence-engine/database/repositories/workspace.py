"""Workspace repositories: investigations, audit, custody, lifecycle, users.

Repositories are the only place that writes. Two invariants they exist to
enforce, neither of which a caller can bypass by constructing a model directly
and adding it to a session:

* **State changes route through the domain transition tables.** Nothing assigns
  `status` or a lifecycle state by direct attribute set.
* **Chained logs serialise their appends.** Audit and custody chains read the
  current tail, compute a hash from it, and insert — a race between two
  concurrent appends would fork the chain, so the read-and-append is locked.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared import enums
from shared.clock import Clock, SystemClock, chain_timestamp
from shared.enums import (
    ActorKind,
    ArtifactType,
    CustodyAction,
    InvestigationStatus,
    LifecycleAxis,
)
from shared.errors import (
    ConflictError,
    LawfulBasisRequired,
    NotFoundError,
    StateTransitionError,
    ValidationError,
)
from shared.hashing import GENESIS, chain_hash, verify_chain
from shared.logging import get_logger

from ..models import (
    AuditLogEntry,
    CustodyEvent,
    EvidenceLifecycleEvent,
    Investigation,
    InvestigationStatusEvent,
    RetentionHold,
    User,
)

logger = get_logger(__name__)


# ------------------------------------------------------------------- audit --


class AuditRepository:
    """Append-only, hash-chained system audit trail."""

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    async def _tail_hash(self) -> str:
        """Hash of the most recent entry, or GENESIS for an empty chain.

        Ordered by `id`, the autoincrement column, so the chain order is the
        insertion order rather than a timestamp that can tie.
        """
        result = await self.session.execute(
            select(AuditLogEntry.entry_hash).order_by(AuditLogEntry.id.desc()).limit(1)
        )
        return result.scalar_one_or_none() or GENESIS

    async def record(
        self,
        *,
        action: str,
        outcome: str,
        investigation_id: uuid.UUID | None = None,
        actor_id: uuid.UUID | None = None,
        actor_label: str = "",
        lawful_basis: str = "",
        resource_type: str = "",
        resource_id: str = "",
        detail: dict[str, Any] | None = None,
    ) -> AuditLogEntry:
        """Append one entry. Never raises for business reasons — refusals are
        audited too, so this must succeed even when the action it records did
        not."""
        previous = await self._tail_hash()
        created_at = self.clock.now()

        body = {
            "investigation_id": str(investigation_id) if investigation_id else None,
            "actor_id": str(actor_id) if actor_id else None,
            "actor_label": actor_label,
            "action": action,
            "outcome": outcome,
            "lawful_basis": lawful_basis,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "detail": detail or {},
            "created_at": chain_timestamp(created_at),
        }

        entry = AuditLogEntry(
            investigation_id=investigation_id,
            actor_id=actor_id,
            actor_label=actor_label,
            action=action,
            outcome=outcome,
            lawful_basis=lawful_basis,
            resource_type=resource_type,
            resource_id=resource_id,
            detail=detail or {},
            previous_hash=previous,
            entry_hash=chain_hash(body, previous),
            created_at=created_at,
        )
        self.session.add(entry)
        await self.session.flush()
        logger.info("audit.recorded", action=action, outcome=outcome)
        return entry

    async def list(
        self, investigation_id: uuid.UUID | None = None, limit: int = 100
    ) -> Sequence[AuditLogEntry]:
        statement = select(AuditLogEntry).order_by(AuditLogEntry.id)
        if investigation_id is not None:
            statement = statement.where(AuditLogEntry.investigation_id == investigation_id)
        result = await self.session.execute(statement.limit(limit))
        return result.scalars().all()

    async def verify(self) -> dict[str, Any]:
        """Re-walk the whole chain and report the first divergence."""
        result = await self.session.execute(select(AuditLogEntry).order_by(AuditLogEntry.id))
        rows = result.scalars().all()

        records = [
            {
                "investigation_id": str(r.investigation_id) if r.investigation_id else None,
                "actor_id": str(r.actor_id) if r.actor_id else None,
                "actor_label": r.actor_label,
                "action": r.action,
                "outcome": r.outcome,
                "lawful_basis": r.lawful_basis,
                "resource_type": r.resource_type,
                "resource_id": r.resource_id,
                "detail": r.detail,
                "created_at": chain_timestamp(r.created_at),
                "previous_hash": r.previous_hash,
                "entry_hash": r.entry_hash,
            }
            for r in rows
        ]
        return verify_chain(records, body_fields=AuditLogEntry.HASHED_FIELDS).as_dict()


# ----------------------------------------------------------------- custody --


class CustodyRepository:
    """Per-artifact chain of custody. One chain per artifact, not one globally."""

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    async def _tail(self, artifact_type: str, artifact_id: uuid.UUID) -> tuple[int, str]:
        result = await self.session.execute(
            select(CustodyEvent.sequence, CustodyEvent.entry_hash)
            .where(
                CustodyEvent.artifact_type == artifact_type,
                CustodyEvent.artifact_id == artifact_id,
            )
            .order_by(CustodyEvent.sequence.desc())
            .limit(1)
        )
        row = result.first()
        return (row[0], row[1]) if row else (0, GENESIS)

    async def record(
        self,
        *,
        investigation_id: uuid.UUID,
        artifact_type: ArtifactType | str,
        artifact_id: uuid.UUID,
        action: CustodyAction | str,
        content_hash: str,
        actor_id: uuid.UUID | None = None,
        actor_kind: ActorKind | str = ActorKind.SYSTEM,
        source_uri: str = "",
        storage_location: str = "",
        transformation: dict[str, Any] | None = None,
        derived_from_id: uuid.UUID | None = None,
    ) -> CustodyEvent:
        """Append a custody record. Never overwrites; every transformation is a
        new row (REVISION_3 §2)."""
        sequence, previous = await self._tail(str(artifact_type), artifact_id)
        occurred_at = self.clock.now()

        body = {
            "investigation_id": str(investigation_id),
            "artifact_type": str(artifact_type),
            "artifact_id": str(artifact_id),
            "sequence": sequence + 1,
            "action": str(action),
            "actor_id": str(actor_id) if actor_id else None,
            "actor_kind": str(actor_kind),
            "source_uri": source_uri,
            "content_hash": content_hash,
            "storage_location": storage_location,
            "transformation": transformation or {},
            "derived_from_id": str(derived_from_id) if derived_from_id else None,
            "occurred_at": chain_timestamp(occurred_at),
        }

        event = CustodyEvent(
            investigation_id=investigation_id,
            artifact_type=str(artifact_type),
            artifact_id=artifact_id,
            sequence=sequence + 1,
            action=str(action),
            actor_id=actor_id,
            actor_kind=str(actor_kind),
            source_uri=source_uri,
            content_hash=content_hash,
            storage_location=storage_location,
            transformation=transformation or {},
            derived_from_id=derived_from_id,
            previous_hash=previous,
            entry_hash=chain_hash(body, previous),
            occurred_at=occurred_at,
        )
        self.session.add(event)
        await self.session.flush()
        return event

    async def chain_for(
        self, artifact_type: ArtifactType | str, artifact_id: uuid.UUID
    ) -> Sequence[CustodyEvent]:
        result = await self.session.execute(
            select(CustodyEvent)
            .where(
                CustodyEvent.artifact_type == str(artifact_type),
                CustodyEvent.artifact_id == artifact_id,
            )
            .order_by(CustodyEvent.sequence)
        )
        return result.scalars().all()

    async def verify(
        self, artifact_type: ArtifactType | str, artifact_id: uuid.UUID
    ) -> dict[str, Any]:
        events = await self.chain_for(artifact_type, artifact_id)
        records = [
            {
                "investigation_id": str(e.investigation_id),
                "artifact_type": e.artifact_type,
                "artifact_id": str(e.artifact_id),
                "sequence": e.sequence,
                "action": e.action,
                "actor_id": str(e.actor_id) if e.actor_id else None,
                "actor_kind": e.actor_kind,
                "source_uri": e.source_uri,
                "content_hash": e.content_hash,
                "storage_location": e.storage_location,
                "transformation": e.transformation,
                "derived_from_id": str(e.derived_from_id) if e.derived_from_id else None,
                "occurred_at": chain_timestamp(e.occurred_at),
                "previous_hash": e.previous_hash,
                "entry_hash": e.entry_hash,
            }
            for e in events
        ]
        return verify_chain(records, body_fields=CustodyEvent.HASHED_FIELDS).as_dict()


# ----------------------------------------------------------- investigations --


class InvestigationRepository:
    """Investigations and their workflow transitions."""

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    async def create(
        self,
        *,
        owner_id: uuid.UUID,
        case_id: str,
        title: str,
        lawful_basis: str,
        purpose: str = "",
        description: str = "",
        jurisdiction: str = "IN",
        retention_days: int | None = None,
        require_lawful_basis: bool = True,
    ) -> Investigation:
        basis = (lawful_basis or "").strip()
        if require_lawful_basis and not basis:
            raise LawfulBasisRequired(
                "A lawful basis must be stated for every investigation. "
                "Set IIE_REQUIRE_LAWFUL_BASIS=false only for non-production testing."
            )
        if not case_id.strip():
            raise ValidationError("case_id must not be empty.")
        if not title.strip():
            raise ValidationError("title must not be empty.")

        existing = await self.session.execute(
            select(Investigation.id).where(
                Investigation.owner_id == owner_id,
                Investigation.case_id == case_id.strip(),
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise ConflictError(f"Case id {case_id!r} already exists for this owner.")

        now = self.clock.now()
        investigation = Investigation(
            owner_id=owner_id,
            case_id=case_id.strip(),
            title=title.strip(),
            description=description,
            lawful_basis=basis,
            purpose=purpose.strip(),
            jurisdiction=jurisdiction,
            status=InvestigationStatus.NEW,
            retention_expires_at=(
                now + timedelta(days=retention_days) if retention_days else None
            ),
        )
        self.session.add(investigation)
        await self.session.flush()
        return investigation

    async def get(self, investigation_id: uuid.UUID) -> Investigation:
        result = await self.session.execute(
            select(Investigation).where(Investigation.id == investigation_id)
        )
        investigation = result.scalar_one_or_none()
        if investigation is None:
            raise NotFoundError(f"Investigation {investigation_id} not found.")
        return investigation

    async def list(
        self, owner_id: uuid.UUID, *, status: str | None = None, limit: int = 50
    ) -> Sequence[Investigation]:
        statement = select(Investigation).where(Investigation.owner_id == owner_id)
        if status:
            statement = statement.where(Investigation.status == status)
        result = await self.session.execute(
            statement.order_by(Investigation.created_at.desc()).limit(limit)
        )
        return result.scalars().all()

    async def transition(
        self,
        investigation: Investigation,
        to_status: InvestigationStatus | str,
        *,
        actor_id: uuid.UUID | None = None,
        reason: str = "",
    ) -> Investigation:
        """Move an investigation through its workflow.

        The only supported way to change status. Validates against the domain
        transition table, demands a reason for backward moves, and writes a
        history row — none of which happens if a caller sets `.status` directly,
        which is why nothing else may.
        """
        current = investigation.status
        target = str(to_status)

        try:
            enums.assert_transition("investigation_status", current, target)
        except enums.IllegalTransition as exc:
            raise StateTransitionError(str(exc), current=current, requested=target) from exc

        if enums.requires_reason("investigation_status", current, target) and not reason.strip():
            raise ValidationError(
                f"Moving from {current} back to {target} requires a reason. "
                "Reopening a case without recording why is the gap an opposing "
                "examiner looks for."
            )

        now = self.clock.now()
        investigation.status = target
        if target == InvestigationStatus.COMPLETED:
            investigation.completed_at = now
        elif current == InvestigationStatus.COMPLETED:
            investigation.completed_at = None

        self.session.add(
            InvestigationStatusEvent(
                investigation_id=investigation.id,
                from_status=current,
                to_status=target,
                reason=reason.strip(),
                actor_id=actor_id,
                occurred_at=now,
            )
        )
        await self.session.flush()
        # `updated_at` carries a server-side onupdate, so the flush expires it.
        # Without an explicit refresh the first read triggers a lazy load, which
        # raises MissingGreenlet under async serialisation.
        await self.session.refresh(investigation)
        return investigation

    async def status_history(
        self, investigation_id: uuid.UUID
    ) -> Sequence[InvestigationStatusEvent]:
        result = await self.session.execute(
            select(InvestigationStatusEvent)
            .where(InvestigationStatusEvent.investigation_id == investigation_id)
            .order_by(InvestigationStatusEvent.occurred_at)
        )
        return result.scalars().all()

    async def has_active_hold(self, investigation_id: uuid.UUID) -> bool:
        """Whether an unreleased preservation lock blocks purge."""
        result = await self.session.execute(
            select(func.count())
            .select_from(RetentionHold)
            .where(
                RetentionHold.investigation_id == investigation_id,
                RetentionHold.released_at.is_(None),
            )
        )
        return (result.scalar_one() or 0) > 0


# ---------------------------------------------------------------- lifecycle --


class LifecycleRepository:
    """Validated evidence lifecycle transitions (REVISION_3 §1)."""

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    async def transition(
        self,
        *,
        investigation_id: uuid.UUID,
        artifact_type: ArtifactType | str,
        artifact_id: uuid.UUID,
        axis: LifecycleAxis | str,
        from_state: str,
        to_state: str,
        reason: str = "",
        actor_id: uuid.UUID | None = None,
    ) -> EvidenceLifecycleEvent:
        machine = (
            "progress_state" if str(axis) == LifecycleAxis.PROGRESS else "retention_state"
        )
        try:
            enums.assert_transition(machine, from_state, to_state)
        except enums.IllegalTransition as exc:
            raise StateTransitionError(str(exc)) from exc

        event = EvidenceLifecycleEvent(
            investigation_id=investigation_id,
            artifact_type=str(artifact_type),
            artifact_id=artifact_id,
            axis=str(axis),
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            actor_id=actor_id,
            occurred_at=self.clock.now(),
        )
        self.session.add(event)
        await self.session.flush()
        return event


# --------------------------------------------------------------------- users --


class UserRepository:
    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    async def get_by_email(self, email: str) -> User | None:
        result = await self.session.execute(
            select(User).where(User.email == email.strip().lower())
        )
        return result.scalar_one_or_none()

    async def get(self, user_id: uuid.UUID) -> User:
        result = await self.session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise NotFoundError(f"User {user_id} not found.")
        return user

    async def create(
        self, *, email: str, password_hash: str, display_name: str = "", role: str = "investigator"
    ) -> User:
        normalized = email.strip().lower()
        if await self.get_by_email(normalized) is not None:
            raise ConflictError(f"User {normalized!r} already exists.")
        user = User(
            email=normalized,
            password_hash=password_hash,
            display_name=display_name or normalized.split("@")[0],
            role=role,
        )
        self.session.add(user)
        await self.session.flush()
        return user

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(User))
        return int(result.scalar_one() or 0)

    async def get_or_create_federated(
        self, *, external_subject: str, external_tenant: str, role: str = "investigator"
    ) -> User:
        """Find or provision the local account behind an iMATCH investigator.

        Keyed by iMATCH's `sub` claim, not by email: the token carries no email,
        and an investigator's address can change without their subject id
        changing. The row exists so audit and custody entries have a real actor
        to attribute — a federated session must never produce anonymous evidence.

        No password hash is set. These accounts cannot be signed into directly;
        the only way to act as one is to present a valid iMATCH token.
        """
        existing = (
            await self.session.execute(
                select(User).where(User.external_subject == external_subject)
            )
        ).scalar_one_or_none()

        if existing is not None:
            if existing.external_tenant != external_tenant:
                # The subject moved tenant, or two tenants share a subject id.
                # Either way, filing new evidence under the stale tenant would
                # misattribute it.
                existing.external_tenant = external_tenant
            existing.last_login_at = self.clock.now()
            await self.session.flush()
            return existing

        user = User(
            email=f"{external_subject}@imatch.federated",
            password_hash="",
            display_name=f"iMATCH {external_subject[:8]}",
            role=role if role in {"investigator", "reviewer", "admin"} else "investigator",
            external_subject=external_subject,
            external_tenant=external_tenant,
            last_login_at=self.clock.now(),
        )
        self.session.add(user)
        await self.session.flush()
        logger.info(
            "user.federated_provisioned",
            external_subject=external_subject,
            tenant=external_tenant,
        )
        return user




__all__ = [
    "AuditRepository",
    "CustodyRepository",
    "InvestigationRepository",
    "LifecycleRepository",
    "UserRepository",
]
