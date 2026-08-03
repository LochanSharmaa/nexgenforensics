"""The evidence chain: observations, entities, facts, and the review queue.

Four concepts kept deliberately separate (ARCHITECTURE §7):

    Page ──< Observation >── Mention ──> Entity
                   │                       │
                   └──────> Fact <─────────┘

* **Observation** — one immutable extraction event. The atom of the system.
* **Mention** — a surface form recognised as referring to something.
* **Entity** — a resolved canonical thing. Mutable and *re-derivable*.
* **Fact** — an assertion about an entity, supported by observations.

Recomputability is the payoff: when correlation or scoring improves, the later
stages re-run over stored observations with no re-crawling and no new API spend,
and `extractor_version` records which logic produced each row so old reports
stay reproducible.

Two of the five database-level invariants are enforced here (§3 and §5 of
DATA_MODEL's invariant list).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from shared.enums import (
    ConfidenceTier,
    EntityType,
    FactClassification,
    FactStatus,
    ObservationMethod,
    ObservationOrigin,
    ReviewKind,
    ReviewStatus,
    VerificationState,
)
from shared.ids import new_id

from ..base import Base, enum_column


class Observation(Base):
    """One immutable extraction event.

    **Invariant 3 (machine side).** `origin` is CHECK-constrained to
    `EXTRACTED`, so human annotation physically cannot land in this table. The
    complementary constraint sits on `notes`, which accepts only `HUMAN`. An
    investigator's hypothesis must never become indistinguishable from a crawled
    fact when the report is challenged months later.
    """

    __tablename__ = "observations"
    __table_args__ = (
        CheckConstraint(
            f"origin = '{ObservationOrigin.EXTRACTED.value}'",
            name="machine_origin_only",
        ),
        Index("ix_observations_investigation_page", "investigation_id", "page_id"),
        Index("ix_observations_normalized", "normalized_value"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    page_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pages.id", ondelete="CASCADE"), nullable=True
    )
    image_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=True
    )

    origin: Mapped[str] = mapped_column(
        String(24), nullable=False, default=ObservationOrigin.EXTRACTED.value,
        server_default=ObservationOrigin.EXTRACTED.value,
    )
    method: Mapped[str] = enum_column(
        tuple(ObservationMethod), constraint_name="method_valid"
    )

    raw_value: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_value: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Character offsets into the stored text content, so a finding can be
    # located in the source rather than merely asserted about it.
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    context_snippet: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Which code produced this. When spaCy or a regex changes, we must be able
    # to say which findings came from which version — reports get defended
    # months after they are written.
    extractor_version: Mapped[str] = mapped_column(String(64), nullable=False)
    method_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extracted_at: Mapped[datetime] = mapped_column(nullable=False)


class Entity(Base):
    """A resolved canonical thing, produced by correlation.

    `possible_duplicate_of` implements the conservative-merge rule: uncertain
    pairs stay separate and flagged rather than merged. Over-merging attributes
    one person's facts to another; under-merging is a visible inconvenience an
    investigator can resolve.
    """

    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint(
            "investigation_id", "type", "normalized_key", name="uq_entities_identity"
        ),
        Index("ix_entities_investigation_type", "investigation_id", "type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = enum_column(tuple(EntityType), constraint_name="type_valid")
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_key: Mapped[str] = mapped_column(String(500), nullable=False)

    source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    independent_source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Machine proposal versus human ruling (REVISION_3 §4).
    verification_state: Mapped[str] = enum_column(
        tuple(VerificationState), constraint_name="verification_valid",
        default=VerificationState.MACHINE_PROPOSED,
    )
    possible_duplicate_of: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entities.id", ondelete="SET NULL"), nullable=True
    )
    first_seen_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(nullable=True)


class EntityAlias(Base):
    """A surface form seen for an entity, with its occurrence count."""

    __tablename__ = "entity_aliases"
    __table_args__ = (
        UniqueConstraint("entity_id", "surface_form", name="uq_entity_alias"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    surface_form: Mapped[str] = mapped_column(String(500), nullable=False)
    occurrences: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class Mention(Base):
    """Links one observation to the entity it refers to."""

    __tablename__ = "mentions"
    __table_args__ = (Index("ix_mentions_entity", "entity_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    observation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("observations.id", ondelete="CASCADE"), nullable=False
    )
    entity_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entities.id", ondelete="SET NULL"), nullable=True
    )
    entity_type: Mapped[str] = enum_column(
        tuple(EntityType), constraint_name="entity_type_valid"
    )
    surface_form: Mapped[str] = mapped_column(String(500), nullable=False)


class Fact(Base):
    """An assertion about an entity, supported by observations.

    **Two independent status axes** (REVISION_3 §5). `status` is evidential and
    machine-authored; `classification` is investigative and human-authored. A
    fact can legitimately be COMMON (three independent sources agree) *and*
    DISPUTED (the investigator has off-platform reason to doubt all three). One
    column could not hold both without silently overwriting either the evidence
    or the judgement.

    **Invariant 5.** A confidence value may never be stored without its
    explanation. Expressed as `confidence_factor_count >= 1` rather than a
    "JSON is not empty" check: PostgreSQL's `json` type has no equality
    operator, so `<> '{}'` is not portable, whereas "at least one explanatory
    factor" is both checkable everywhere and a truer statement of the rule.
    """

    __tablename__ = "facts"
    __table_args__ = (
        CheckConstraint("confidence_factor_count >= 1", name="explanation_required"),
        Index("ix_facts_entity", "entity_id"),
        Index("ix_facts_investigation_status", "investigation_id", "status"),
        Index("ix_facts_conflict_group", "conflict_group_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )

    attribute: Mapped[str] = mapped_column(String(120), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_value: Mapped[str] = mapped_column(Text, nullable=False, default="")

    status: Mapped[str] = enum_column(tuple(FactStatus), constraint_name="status_valid")
    # Competing values share a group. All are retained with their own evidence;
    # none is deleted. The investigator adjudicates — a tool that picks a winner
    # is hiding evidence.
    conflict_group_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)

    classification: Mapped[str] = enum_column(
        tuple(FactClassification), constraint_name="classification_valid",
        default=FactClassification.UNVERIFIED,
    )
    classified_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    classified_at: Mapped[datetime | None] = mapped_column(nullable=True)
    classification_note: Mapped[str] = mapped_column(Text, nullable=False, default="")

    confidence: Mapped[str] = enum_column(
        tuple(ConfidenceTier), constraint_name="confidence_valid"
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence_explanation: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    confidence_factor_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    independent_source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_asserted_at: Mapped[datetime | None] = mapped_column(nullable=True)

    computed_at: Mapped[datetime] = mapped_column(nullable=False)
    scorer_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")


class FactEvidence(Base):
    """Join table: which observations support which fact."""

    __tablename__ = "fact_evidence"

    fact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("facts.id", ondelete="CASCADE"), primary_key=True
    )
    observation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("observations.id", ondelete="CASCADE"), primary_key=True
    )


class ReviewItem(Base):
    """A machine proposal awaiting a human ruling (REVISION_3 §4).

    Extraction never writes a confirmed entity. It writes an observation plus a
    review item; a human decision promotes it. A rejection never deletes the
    observation — "the machine saw this and a human disagreed" is itself a
    finding worth preserving.
    """

    __tablename__ = "review_items"
    __table_args__ = (
        Index("ix_review_queue", "investigation_id", "status", "priority"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = enum_column(tuple(ReviewKind), constraint_name="kind_valid")
    subject_type: Mapped[str] = mapped_column(String(60), nullable=False)
    subject_id: Mapped[uuid.UUID] = mapped_column(nullable=False)

    proposal: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    # Why the machine proposes this, including supporting observation ids — so
    # the reviewer rules on evidence rather than on a bare suggestion.
    rationale: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[str] = enum_column(
        tuple(ReviewStatus), constraint_name="status_valid", default=ReviewStatus.PENDING
    )
    decided_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(nullable=True)
    decision_note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(nullable=False)


__all__ = [
    "Entity",
    "EntityAlias",
    "Fact",
    "FactEvidence",
    "Mention",
    "Observation",
    "ReviewItem",
]
