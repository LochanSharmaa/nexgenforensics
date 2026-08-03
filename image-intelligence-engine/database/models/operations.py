"""Monitoring, reporting, copilot and search.

The pieces that turn a one-shot investigation into a longitudinal one, plus the
output surfaces.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from shared.enums import (
    ChangeType,
    CopilotRole,
    MonitorCadence,
    ReportFormat,
    SearchObjectType,
    ValidationStatus,
)
from shared.ids import new_id

from ..base import Base, enum_column

# ------------------------------------------------------------- monitoring --


class Monitor(Base):
    """A standing watch on one image (REVISION_3 §13)."""

    __tablename__ = "monitors"
    __table_args__ = (Index("ix_monitors_due", "enabled", "next_run_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    cadence: Mapped[str] = enum_column(
        tuple(MonitorCadence), constraint_name="cadence_valid"
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    next_run_at: Mapped[datetime] = mapped_column(nullable=False)
    last_run_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)


class MonitorRun(Base):
    __tablename__ = "monitor_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    monitor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("monitors.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    changes_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="RUNNING")


class MonitorChange(Base):
    """One difference a sweep found against the prior state.

    `PAGE_REMOVED` and `PAGE_UNOBSERVABLE` are distinct change types, and
    `confirming_runs` gates the former: a removal is asserted only after
    consecutive 404/410 responses. A single Cloudflare block must never
    manufacture a takedown event.
    """

    __tablename__ = "monitor_changes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    monitor_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("monitor_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    change_type: Mapped[str] = enum_column(
        tuple(ChangeType), constraint_name="change_type_valid"
    )
    page_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pages.id", ondelete="SET NULL"), nullable=True
    )
    image_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("images.id", ondelete="SET NULL"), nullable=True
    )
    detail: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    confirming_runs: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    timeline_event_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("timeline_events.id", ondelete="SET NULL"), nullable=True
    )
    detected_at: Mapped[datetime] = mapped_column(nullable=False)


# ---------------------------------------------------------------- reports --


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    format: Mapped[str] = enum_column(tuple(ReportFormat), constraint_name="format_valid")
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    # Lets a reviewer confirm the file they hold is the file that was generated.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    generated_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(nullable=False)


# ---------------------------------------------------------------- copilot --


class CopilotSession(Base):
    __tablename__ = "copilot_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(nullable=False)


class CopilotMessage(Base):
    """One turn, with its citations and the outcome of validating them.

    `validation_status = REJECTED` is a normal, persisted result rather than an
    error: the assistant's failures are auditable too, and a rising rejection
    rate is a correctness signal worth seeing.
    """

    __tablename__ = "copilot_messages"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("copilot_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = enum_column(tuple(CopilotRole), constraint_name="role_valid")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    validation_status: Mapped[str] = enum_column(
        tuple(ValidationStatus), constraint_name="validation_valid",
        default=ValidationStatus.PENDING,
    )
    rejected_spans: Mapped[list[Any]] = mapped_column(nullable=False, default=list)
    model: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(nullable=False)


# ----------------------------------------------------------------- search --


class SearchDocument(Base):
    """One row per searchable object, across every type.

    A single table with one index gives comparable ranking across cases, pages,
    entities and notes alike, and avoids a second datastore until measurement
    justifies one. The PostgreSQL `tsvector` column and its GIN index are added
    by a dialect-guarded step in the migration, since SQLite has no equivalent.
    """

    __tablename__ = "search_documents"
    __table_args__ = (
        UniqueConstraint("object_type", "object_id", name="uq_search_object"),
        Index("ix_search_investigation", "investigation_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=True
    )
    object_type: Mapped[str] = enum_column(
        tuple(SearchObjectType), constraint_name="object_type_valid"
    )
    object_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime] = mapped_column(nullable=False)


__all__ = [
    "CopilotMessage",
    "CopilotSession",
    "Monitor",
    "MonitorChange",
    "MonitorRun",
    "Report",
    "SearchDocument",
]
