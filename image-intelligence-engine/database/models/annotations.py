"""Human annotation: notes, checklists, bookmarks.

**Invariant 3 (human side).** `notes.origin` is CHECK-constrained to `HUMAN`,
mirroring the `EXTRACTED`-only constraint on `observations`. Together the two
constraints make the separation physical rather than procedural: a note *cites*
evidence through `note_links`; it can never become evidence.

That matters because six months later, under challenge, an investigator's
working hypothesis must still be distinguishable from something a page actually
said.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from shared.enums import ObservationOrigin, TargetType
from shared.ids import new_id

from ..base import Base, TimestampMixin, enum_column


class Note(Base, TimestampMixin):
    """A rich-text investigator note."""

    __tablename__ = "notes"
    __table_args__ = (
        CheckConstraint(
            f"origin = '{ObservationOrigin.HUMAN.value}'", name="human_origin_only"
        ),
        Index("ix_notes_investigation", "investigation_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    author_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    origin: Mapped[str] = mapped_column(
        String(24), nullable=False, default=ObservationOrigin.HUMAN.value,
        server_default=ObservationOrigin.HUMAN.value,
    )

    title: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    body_richtext: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    # Plain projection kept alongside the structured document purely so the
    # note is searchable without the search layer having to parse TipTap JSON.
    body_plain: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tags: Mapped[list[str]] = mapped_column(nullable=False, default=list)


class NoteLink(Base):
    """A note citing a piece of evidence. One-directional, by design."""

    __tablename__ = "note_links"

    # Composite key across all three columns: one note may cite many targets,
    # and the same target through different note.
    note_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("notes.id", ondelete="CASCADE"), primary_key=True
    )
    target_type: Mapped[str] = mapped_column(String(40), primary_key=True)
    target_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)

    __table_args__ = (
        CheckConstraint(
            "target_type IN (" + ", ".join(f"'{t.value}'" for t in TargetType) + ")",
            name="ck_note_links_target_type_valid",
        ),
    )


class NoteAttachment(Base):
    __tablename__ = "note_attachments"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    note_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("notes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(300), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(nullable=False)


class Checklist(Base):
    __tablename__ = "checklists"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)


class ChecklistItem(Base):
    __tablename__ = "checklist_items"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    checklist_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("checklists.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(Text, nullable=False)
    done: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Bookmark(Base):
    __tablename__ = "bookmarks"
    __table_args__ = (
        UniqueConstraint(
            "owner_id", "target_type", "target_id", name="uq_bookmark_target"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    target_type: Mapped[str] = enum_column(
        tuple(TargetType), constraint_name="target_type_valid"
    )
    target_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    label: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(nullable=False)


__all__ = [
    "Bookmark",
    "Checklist",
    "ChecklistItem",
    "Note",
    "NoteAttachment",
    "NoteLink",
]
