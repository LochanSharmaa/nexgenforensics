"""Workspace, integrity and retention tables.

Scope is deliberate. Phase 2's acceptance criteria require migrations to run, an
empty investigation to be creatable through the API, and audit logging to work
*before* feature development begins. So this module carries the workspace and
integrity tables only; the evidence tables (observations, entities, facts,
graph, timeline) land in Phase 3 against the full DATA_MODEL specification.

The integrity machinery comes first on purpose. An audit log added after the
features it is supposed to record is an audit log with a hole in it.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.enums import (
    ActorKind,
    ArtifactType,
    CustodyAction,
    InvestigationStatus,
    LifecycleAxis,
    PipelineStage,
    ProgressState,
    RetentionState,
    RunStatus,
    RunTrigger,
    StageStatus,
)
from shared.ids import new_id

from ..base import Base, TimestampMixin, enum_column

# ---------------------------------------------------------------- identity --


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    role: Mapped[str] = enum_column(
        ("investigator", "reviewer", "admin"),
        constraint_name="role_valid",
        default="investigator",
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Set when the account was provisioned from a NexGen iMATCH token. Keyed by
    # iMATCH's `sub` claim rather than by email, because the token carries no
    # email and an investigator's address can change without their id changing.
    external_subject: Mapped[str | None] = mapped_column(
        String(120), nullable=True, unique=True
    )
    external_tenant: Mapped[str] = mapped_column(String(120), nullable=False, default="")


# ------------------------------------------------------------- workspace ----


class Investigation(Base, TimestampMixin):
    """A persistent case. The unit of work (ARCHITECTURE §5)."""

    __tablename__ = "investigations"
    __table_args__ = (
        UniqueConstraint("owner_id", "case_id", name="uq_investigations_owner_case"),
        Index("ix_investigations_owner_status", "owner_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)

    case_id: Mapped[str] = mapped_column(String(120), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Required by policy. Recorded on the investigation and echoed into the
    # audit chain, so "why was this examined" survives independently of this row.
    lawful_basis: Mapped[str] = mapped_column(String(500), nullable=False)
    purpose: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    jurisdiction: Mapped[str] = mapped_column(String(8), nullable=False, default="IN")

    status: Mapped[str] = enum_column(
        tuple(InvestigationStatus), constraint_name="status_valid",
        default=InvestigationStatus.NEW, index=True,
    )

    retention_expires_at: Mapped[datetime | None] = mapped_column(nullable=True, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    owner: Mapped[User] = relationship(lazy="joined")


class InvestigationStatusEvent(Base):
    """Workflow transition history.

    Separate from the audit log because status history is queried as a sequence
    ("how did this case get to COMPLETED?"), and because backward transitions
    carry a mandatory reason that deserves a typed column rather than a JSON key.
    """

    __tablename__ = "investigation_status_events"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    from_status: Mapped[str] = enum_column(
        tuple(InvestigationStatus), constraint_name="from_status_valid"
    )
    to_status: Mapped[str] = enum_column(
        tuple(InvestigationStatus), constraint_name="to_status_valid"
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    actor_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(nullable=False)


# -------------------------------------------------------------- pipeline ----


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    trigger: Mapped[str] = enum_column(tuple(RunTrigger), constraint_name="trigger_valid")
    status: Mapped[str] = enum_column(
        tuple(RunStatus), constraint_name="status_valid", default=RunStatus.QUEUED
    )
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")


class PipelineStageRow(Base):
    """Per-stage state. This is what makes an investigation resumable.

    Resume finds the first non-OK stage in PIPELINE_ORDER, so a crash in a late
    stage never re-runs the paid discovery call in an early one.
    """

    __tablename__ = "pipeline_stages"
    __table_args__ = (UniqueConstraint("run_id", "stage", name="uq_pipeline_stages_run_stage"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage: Mapped[str] = enum_column(tuple(PipelineStage), constraint_name="stage_valid")
    status: Mapped[str] = enum_column(
        tuple(StageStatus), constraint_name="status_valid", default=StageStatus.PENDING
    )
    items_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    items_done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    items_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")


# ------------------------------------------------- configuration snapshots --


class ConfigSnapshot(Base):
    """The system configuration a run executed under (REVISION_3 §9).

    Captured per run, not per investigation: a case re-run after a model upgrade
    contains findings from two different extraction regimes, and the report must
    be able to say which came from which.
    """

    __tablename__ = "config_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=True
    )
    app_version: Mapped[str] = mapped_column(String(64), nullable=False)
    ruleset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    scorer_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    classifier_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    parser_versions: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    extractor_versions: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    provider_versions: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    prompt_versions: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    thresholds: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    captured_at: Mapped[datetime] = mapped_column(nullable=False)


# ------------------------------------------------------------- retention ----


class RetentionPolicy(Base, TimestampMixin):
    __tablename__ = "retention_policies"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    jurisdiction: Mapped[str] = mapped_column(String(8), nullable=False, default="IN")
    default_days: Mapped[int] = mapped_column(Integer, nullable=False)
    auto_purge: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    export_before_purge: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    __table_args__ = (
        CheckConstraint("default_days >= 1", name="ck_retention_policies_days_positive"),
    )


class RetentionHold(Base):
    """A preservation lock. Blocks purge absolutely while unreleased.

    Holds win over policy, always. A legal hold a scheduler could override is not
    a hold.
    """

    __tablename__ = "retention_holds"
    __table_args__ = (
        Index("ix_retention_holds_active", "investigation_id", "released_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    artifact_type: Mapped[str] = enum_column(
        tuple(ArtifactType), constraint_name="artifact_type_valid",
        default=ArtifactType.INVESTIGATION,
    )
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    placed_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    placed_at: Mapped[datetime] = mapped_column(nullable=False)
    released_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    released_at: Mapped[datetime | None] = mapped_column(nullable=True)

    @property
    def is_active(self) -> bool:
        return self.released_at is None


# ------------------------------------------------------------- lifecycle ----


class EvidenceLifecycleEvent(Base):
    """One validated state transition on one artifact (REVISION_3 §1)."""

    __tablename__ = "evidence_lifecycle_events"
    __table_args__ = (
        Index("ix_lifecycle_artifact", "artifact_type", "artifact_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    artifact_type: Mapped[str] = enum_column(
        tuple(ArtifactType), constraint_name="artifact_type_valid"
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    axis: Mapped[str] = enum_column(tuple(LifecycleAxis), constraint_name="axis_valid")
    from_state: Mapped[str] = enum_column(
        tuple(ProgressState) + tuple(RetentionState), constraint_name="from_state_valid"
    )
    to_state: Mapped[str] = enum_column(
        tuple(ProgressState) + tuple(RetentionState), constraint_name="to_state_valid"
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    actor_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(nullable=False)


# ---------------------------------------------------------------- custody ----


class CustodyEvent(Base):
    """Per-artifact chain of custody (REVISION_3 §2).

    Distinct from the audit log: the audit log answers "what did the system do?",
    this answers "what happened to *this artifact*?". An examiner challenging a
    screenshot asks the second question, and reconstructing it by filtering the
    audit log would be fragile.

    Every transformation writes a new row. Nothing is ever overwritten.
    """

    __tablename__ = "custody_events"
    __table_args__ = (
        UniqueConstraint(
            "artifact_type", "artifact_id", "sequence", name="uq_custody_artifact_sequence"
        ),
        Index("ix_custody_artifact", "artifact_type", "artifact_id", "sequence"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    # No FK, for the same reason as audit_log: a custody chain must survive the
    # purge of the artifact it documents. Proving *what was deleted and when*
    # is precisely what it is for.
    investigation_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    artifact_type: Mapped[str] = enum_column(
        tuple(ArtifactType), constraint_name="artifact_type_valid"
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)

    action: Mapped[str] = enum_column(tuple(CustodyAction), constraint_name="action_valid")
    actor_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    actor_kind: Mapped[str] = enum_column(
        tuple(ActorKind), constraint_name="actor_kind_valid", default=ActorKind.SYSTEM
    )

    source_uri: Mapped[str] = mapped_column(Text, nullable=False, default="")
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_location: Mapped[str] = mapped_column(Text, nullable=False, default="")
    transformation: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)

    # Makes derivation explicit, so download → hash → screenshot → report is a
    # walkable chain rather than four unrelated rows.
    derived_from_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("custody_events.id"), nullable=True
    )

    previous_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    entry_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(nullable=False)

    HASHED_FIELDS = (
        "investigation_id", "artifact_type", "artifact_id", "sequence", "action",
        "actor_id", "actor_kind", "source_uri", "content_hash", "storage_location",
        "transformation", "derived_from_id", "occurred_at",
    )


# ------------------------------------------------------------------ audit ----


class AuditLogEntry(Base):
    """Append-only, hash-chained system audit trail.

    `BigInteger` autoincrement rather than a UUID primary key: the chain has a
    total order and the primary key should express it, so verification can walk
    rows in insertion order without relying on a timestamp that may tie.

    Immutability is enforced by privilege in the migration
    (`REVOKE UPDATE, DELETE`), not by application discipline.
    """

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_investigation", "investigation_id", "id"),
        Index("ix_audit_log_action", "action"),
    )

    # SQLite only auto-increments a column declared exactly `INTEGER PRIMARY
    # KEY`; a BIGINT primary key stays NULL and the insert fails. The variant
    # gives PostgreSQL its BIGINT and SQLite its INTEGER — which is 64-bit
    # internally anyway, so nothing is lost.
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    # Deliberately NOT a foreign key. Two reasons, both load-bearing:
    #   1. `ON DELETE SET NULL` is an UPDATE, which the append-only REVOKE
    #      forbids — the FK would make deleting an investigation impossible.
    #   2. An audit log must outlive what it describes. After a retention purge
    #      the investigation row is a tombstone and its history survives; a
    #      CASCADE would erase the record of what was deleted, defeating the
    #      entire purpose of having a retention regime.
    investigation_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    actor_label: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    outcome: Mapped[str] = mapped_column(String(80), nullable=False)
    lawful_basis: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    resource_type: Mapped[str] = mapped_column(String(60), nullable=False, default="")
    resource_id: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    detail: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)

    previous_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    entry_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)

    HASHED_FIELDS = (
        "investigation_id", "actor_id", "actor_label", "action", "outcome",
        "lawful_basis", "resource_type", "resource_id", "detail", "created_at",
    )
    """Columns covered by the chain hash. The surrogate `id` is excluded so the
    hash depends only on content, and `previous_hash`/`entry_hash` are excluded
    because they are the chain itself."""


__all__ = [
    "AuditLogEntry",
    "ConfigSnapshot",
    "CustodyEvent",
    "EvidenceLifecycleEvent",
    "Investigation",
    "InvestigationStatusEvent",
    "PipelineRun",
    "PipelineStageRow",
    "RetentionHold",
    "RetentionPolicy",
    "User",
]
