"""Relationship graph and timeline.

Modelled in PostgreSQL rather than a graph database (ARCHITECTURE §3.2): an
investigation graph is thousands of nodes, which recursive CTEs traverse in
milliseconds, and a second datastore would bring a second query language, a
synchronisation problem between evidence and graph, and a second backup path.

Two of the five database-level invariants live here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from shared.enums import (
    ConfidenceTier,
    EdgeType,
    NodeType,
    TimelineKind,
    TimePrecision,
)
from shared.ids import new_id

from ..base import Base, enum_column


class GraphNode(Base):
    """A node in the investigation graph, projecting an underlying record.

    **Invariant 2.** A `PERSON` node cannot exist without `asserted_by_page_id`
    — a page that explicitly named them. There is therefore no schema-level path
    from an image to a person: identity enters the graph only as a quoted claim
    with a citation, never as an inference.
    """

    __tablename__ = "graph_nodes"
    __table_args__ = (
        CheckConstraint(
            f"node_type <> '{NodeType.PERSON.value}' OR asserted_by_page_id IS NOT NULL",
            name="person_requires_assertion",
        ),
        UniqueConstraint(
            "investigation_id", "node_type", "ref_id", name="uq_graph_node_identity"
        ),
        Index("ix_graph_nodes_investigation_type", "investigation_id", "node_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    node_type: Mapped[str] = enum_column(tuple(NodeType), constraint_name="node_type_valid")

    # Polymorphic reference to the record this node projects. Not a foreign key:
    # the target table varies by node_type, and the graph is a rebuildable view
    # over evidence rather than an owner of it.
    ref_table: Mapped[str] = mapped_column(String(60), nullable=False)
    ref_id: Mapped[uuid.UUID] = mapped_column(nullable=False)

    label: Mapped[str] = mapped_column(Text, nullable=False)
    asserted_by_page_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pages.id", ondelete="CASCADE"), nullable=True
    )
    attributes: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)


class GraphEdge(Base):
    """An evidence-backed relationship between two nodes.

    **Invariant 1.** `evidence_observation_ids` must contain at least one entry.
    An unsupported edge fails to insert, which is what stops the graph drifting
    into an unsourced parallel truth alongside the evidence store.

    The check uses `json_array_length`, which both PostgreSQL and SQLite
    implement, so the guarantee holds in tests and in production alike.
    `derivation` records the rule that produced the edge, so a reviewer can ask
    not just "what supports this?" but "why was this edge drawn at all?".
    """

    __tablename__ = "graph_edges"
    __table_args__ = (
        CheckConstraint(
            "json_array_length(evidence_observation_ids) >= 1",
            name="edge_requires_evidence",
        ),
        UniqueConstraint(
            "from_node_id", "to_node_id", "edge_type", name="uq_graph_edge_identity"
        ),
        Index("ix_graph_edges_from", "from_node_id"),
        Index("ix_graph_edges_to", "to_node_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    from_node_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False
    )
    to_node_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False
    )
    edge_type: Mapped[str] = enum_column(tuple(EdgeType), constraint_name="edge_type_valid")

    derivation: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_observation_ids: Mapped[list[str]] = mapped_column(nullable=False, default=list)
    confidence: Mapped[str] = enum_column(
        tuple(ConfidenceTier), constraint_name="confidence_valid",
        default=ConfidenceTier.LOW,
    )
    created_at: Mapped[datetime] = mapped_column(nullable=False)


class TimelineEvent(Base):
    """One chronological event with its supporting evidence.

    `precision` is stored rather than inferred from the timestamp's shape: an
    event known only to the year must render as "2019", never
    "1 Jan 2019 00:00". Rendering a guess as exact would be a lie of formatting.
    """

    __tablename__ = "timeline_events"
    __table_args__ = (
        Index("ix_timeline_investigation_time", "investigation_id", "occurred_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    occurred_at: Mapped[datetime] = mapped_column(nullable=False)
    precision: Mapped[str] = enum_column(
        tuple(TimePrecision), constraint_name="precision_valid"
    )
    kind: Mapped[str] = enum_column(tuple(TimelineKind), constraint_name="kind_valid")
    description: Mapped[str] = mapped_column(Text, nullable=False)

    page_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pages.id", ondelete="CASCADE"), nullable=True
    )
    image_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=True
    )
    evidence_observation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("observations.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(nullable=False)


__all__ = ["GraphEdge", "GraphNode", "TimelineEvent"]
