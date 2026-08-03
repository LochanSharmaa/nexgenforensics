"""Sources: domains, images, appearances, pages and duplicate-content clusters.

Portability note on hashes. DATA_MODEL specifies `bit(64)` for perceptual
hashes and simhashes. Those are stored here as 16-character hex `String`
instead, for two reasons: `bit(64)` has no SQLite equivalent so the test suite
could not exercise these tables, and a 64-bit unsigned hash does not fit a
signed `BIGINT` without sign-juggling that is easy to get subtly wrong. Hamming
comparison happens in-process anyway — ARCHITECTURE §17 specifies an in-process
BK-tree for pHash lookup — so nothing is lost. If PostgreSQL-side bit operations
are ever needed, a generated `bit(64)` column can be added without touching this
model.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
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
    CopyRole,
    DomainClassification,
    ImageRelationship,
    ImageRole,
    ProgressState,
    ProviderCapability,
    RetentionState,
    VerificationResult,
)
from shared.ids import new_id

from ..base import Base, enum_column

HASH_HEX_LENGTH = 16
"""16 hex characters = 64 bits."""


class Domain(Base):
    """A registrable domain (eTLD+1), the unit of source independence.

    Computed once at write time rather than derived per query, because
    independence scoring reads it constantly (ARCHITECTURE §9.3).
    """

    __tablename__ = "domains"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    registrable_domain: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    classification: Mapped[str] = enum_column(
        tuple(DomainClassification),
        constraint_name="classification_valid",
        default=DomainClassification.UNKNOWN,
    )
    # Which signals produced the label, so the classification is auditable.
    classification_basis: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    first_seen_at: Mapped[datetime] = mapped_column(nullable=False)


class Image(Base):
    """Every image is an evidence object — the uploaded probe and every image
    discovered during the investigation alike (REVISION_3 §1)."""

    __tablename__ = "images"
    __table_args__ = (
        Index("ix_images_investigation_role", "investigation_id", "role"),
        Index("ix_images_sha256", "sha256"),
        Index("ix_images_phash", "phash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = enum_column(tuple(ImageRole), constraint_name="role_valid")

    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    phash: Mapped[str] = mapped_column(String(HASH_HEX_LENGTH), nullable=False)
    dhash: Mapped[str | None] = mapped_column(String(HASH_HEX_LENGTH), nullable=True)
    whash: Mapped[str | None] = mapped_column(String(HASH_HEX_LENGTH), nullable=True)

    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    mime_type: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    exif: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)

    storage_key: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_page_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pages.id", ondelete="SET NULL"), nullable=True
    )
    source_image_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    archive_url: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # `downloaded_at` is when *we* fetched it. `first_seen_at`/`last_seen_at`
    # are observed publication bounds derived from provider dates and archive
    # snapshots. Conflating them would make every image look newly published.
    discovered_at: Mapped[datetime | None] = mapped_column(nullable=True)
    downloaded_at: Mapped[datetime | None] = mapped_column(nullable=True)
    first_seen_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Two independent lifecycle axes (REVISION_3 §1).
    progress_state: Mapped[str] = enum_column(
        tuple(ProgressState), constraint_name="progress_valid",
        default=ProgressState.DISCOVERED,
    )
    retention_state: Mapped[str] = enum_column(
        tuple(RetentionState), constraint_name="retention_valid",
        default=RetentionState.ACTIVE,
    )
    created_at: Mapped[datetime] = mapped_column(nullable=False)


class ImageRelationshipRow(Base):
    """A typed, justified provenance edge between two images."""

    __tablename__ = "image_relationships"
    __table_args__ = (
        UniqueConstraint("from_image_id", "to_image_id", name="uq_image_rel_pair"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    from_image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    to_image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    relationship: Mapped[str] = enum_column(
        tuple(ImageRelationship), constraint_name="relationship_valid"
    )
    phash_distance: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The actual measurements behind the call, so a classification can be
    # re-checked months later without re-fetching the image.
    justification: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    classifier_version: Mapped[str] = mapped_column(String(64), nullable=False)
    classified_at: Mapped[datetime] = mapped_column(nullable=False)


class DiscoveryRequest(Base):
    """Search provenance: how evidence was found is itself evidence
    (REVISION_3 §8)."""

    __tablename__ = "discovery_requests"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True
    )
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    provider_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    capability: Mapped[str] = enum_column(
        tuple(ProviderCapability), constraint_name="capability_valid"
    )
    probe_image_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("images.id", ondelete="SET NULL"), nullable=True
    )
    query_parameters: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)

    requested_at: Mapped[datetime] = mapped_column(nullable=False)
    responded_at: Mapped[datetime | None] = mapped_column(nullable=True)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Retained because when a classification is challenged the question is often
    # "what did the provider actually return?", not "what did we conclude?".
    raw_response_key: Mapped[str] = mapped_column(Text, nullable=False, default="")

    results_returned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    results_accepted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    results_rejected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejection_reasons: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")
    cost_units: Mapped[float | None] = mapped_column(Float, nullable=True)


class Appearance(Base):
    """A provider's claim that an image appears at a URL, plus the outcome of
    verifying that claim locally."""

    __tablename__ = "appearances"
    __table_args__ = (Index("ix_appearances_probe", "probe_image_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    probe_image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    discovery_request_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("discovery_requests.id", ondelete="SET NULL"), nullable=True
    )
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    page_url: Mapped[str] = mapped_column(Text, nullable=False)
    image_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    provider_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    provider_reported_date: Mapped[datetime | None] = mapped_column(nullable=True)
    thumbnail_key: Mapped[str] = mapped_column(Text, nullable=False, default="")
    archive_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    discovered_at: Mapped[datetime] = mapped_column(nullable=False)
    verified_at: Mapped[datetime | None] = mapped_column(nullable=True)
    verification_result: Mapped[str] = enum_column(
        tuple(VerificationResult), constraint_name="verification_valid",
        default=VerificationResult.PENDING,
    )


class Page(Base):
    """A fetched web page and its captured artifacts."""

    __tablename__ = "pages"
    __table_args__ = (
        UniqueConstraint("investigation_id", "url", name="uq_pages_investigation_url"),
        Index("ix_pages_domain", "domain_id"),
        Index("ix_pages_simhash", "content_simhash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    domain_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("domains.id"), nullable=False)

    url: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)

    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    site_name: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="")
    author_raw: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    published_at: Mapped[datetime | None] = mapped_column(nullable=True)
    updated_at_source: Mapped[datetime | None] = mapped_column(nullable=True)

    content_simhash: Mapped[str | None] = mapped_column(
        String(HASH_HEX_LENGTH), nullable=True
    )
    raw_html_key: Mapped[str] = mapped_column(Text, nullable=False, default="")
    screenshot_key: Mapped[str] = mapped_column(Text, nullable=False, default="")
    text_content_key: Mapped[str] = mapped_column(Text, nullable=False, default="")
    outbound_links: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)

    fetched_at: Mapped[datetime | None] = mapped_column(nullable=True)
    fetch_error: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Distinguishes "we could not read this page" from "this page is gone".
    # A 403 is not evidence of removal; conflating them would fabricate a
    # takedown event (REVISION_3 §7).
    observable: Mapped[bool] = mapped_column(nullable=False, default=True)

    progress_state: Mapped[str] = enum_column(
        tuple(ProgressState), constraint_name="progress_valid",
        default=ProgressState.DISCOVERED,
    )
    retention_state: Mapped[str] = enum_column(
        tuple(RetentionState), constraint_name="retention_valid",
        default=RetentionState.ACTIVE,
    )


class ContentCluster(Base):
    """A group of pages that copy one another (ARCHITECTURE §10.2).

    Collapsed to its ORIGINAL when counting independent sources, which is why
    SCORE must run after CLUSTER.
    """

    __tablename__ = "content_clusters"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_id)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    original_page_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("pages.id", ondelete="SET NULL"), nullable=True
    )
    member_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(nullable=False)


class ContentClusterMember(Base):
    __tablename__ = "content_cluster_members"

    cluster_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("content_clusters.id", ondelete="CASCADE"), primary_key=True
    )
    page_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pages.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = enum_column(tuple(CopyRole), constraint_name="role_valid")
    similarity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # TRANSLATION is the least reliable classification here; flagging it keeps
    # an over-confident guess from corrupting independence counts.
    role_confidence: Mapped[str] = mapped_column(String(24), nullable=False, default="LOW")


__all__ = [
    "HASH_HEX_LENGTH",
    "Appearance",
    "ContentCluster",
    "ContentClusterMember",
    "DiscoveryRequest",
    "Domain",
    "Image",
    "ImageRelationshipRow",
    "Page",
]
