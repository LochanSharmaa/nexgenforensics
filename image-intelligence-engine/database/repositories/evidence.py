"""Evidence repositories: sources, observations, entities, facts, graph.

These enforce at the application layer the same rules the database enforces at
the storage layer. The duplication is deliberate — the constraint is the
guarantee, the repository is the *good error message*. A caller who forgets
evidence on an edge should read "an edge must cite at least one observation",
not a raw `IntegrityError` naming a constraint.

Every write path here is also the place recomputability is preserved: nothing
mutates an observation, ever. Correlation and scoring rewrite `entities`,
`facts` and the graph from observations that stay untouched, which is what lets
stages 9-13 re-run without re-crawling.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.clock import Clock, SystemClock
from shared.enums import (
    ConfidenceTier,
    EntityType,
    FactClassification,
    FactStatus,
    NodeType,
    ObservationMethod,
    ReviewStatus,
    VerificationState,
)
from shared.errors import NotFoundError, ValidationError
from shared.logging import get_logger

from ..models import (
    Domain,
    Entity,
    EntityAlias,
    Fact,
    FactEvidence,
    GraphEdge,
    GraphNode,
    Image,
    Mention,
    Observation,
    Page,
    ReviewItem,
)

logger = get_logger(__name__)


# ----------------------------------------------------------------- sources --


class SourceRepository:
    """Domains and pages."""

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    async def upsert_domain(
        self, registrable_domain: str, *, classification: str | None = None,
        basis: dict[str, Any] | None = None,
    ) -> Domain:
        """Domains are global, not per-investigation: independence counting
        compares registrable domains across every case the operator runs."""
        normalized = registrable_domain.strip().lower().rstrip(".")
        if not normalized:
            raise ValidationError("registrable_domain must not be empty.")

        existing = (
            await self.session.execute(
                select(Domain).where(Domain.registrable_domain == normalized)
            )
        ).scalar_one_or_none()

        if existing is not None:
            if classification and existing.classification == "UNKNOWN":
                existing.classification = classification
                existing.classification_basis = basis or {}
            return existing

        domain = Domain(
            registrable_domain=normalized,
            classification=classification or "UNKNOWN",
            classification_basis=basis or {},
            first_seen_at=self.clock.now(),
        )
        self.session.add(domain)
        await self.session.flush()
        return domain

    async def add_page(
        self, *, investigation_id: uuid.UUID, domain_id: uuid.UUID, url: str, **fields: Any
    ) -> Page:
        page = Page(
            investigation_id=investigation_id, domain_id=domain_id, url=url, **fields
        )
        self.session.add(page)
        await self.session.flush()
        return page

    async def get_page(self, page_id: uuid.UUID) -> Page:
        page = (
            await self.session.execute(select(Page).where(Page.id == page_id))
        ).scalar_one_or_none()
        if page is None:
            raise NotFoundError(f"Page {page_id} not found.")
        return page

    async def independent_domain_count(self, observation_ids: Sequence[uuid.UUID]) -> int:
        """Distinct registrable domains behind a set of observations.

        The raw count, before duplicate-content clusters are collapsed. Phase 10
        adds that collapse; until then this deliberately over-counts rather than
        guessing, because silently under-reporting independence would inflate
        confidence.
        """
        if not observation_ids:
            return 0
        result = await self.session.execute(
            select(func.count(func.distinct(Page.domain_id)))
            .select_from(Observation)
            .join(Page, Observation.page_id == Page.id)
            .where(Observation.id.in_(list(observation_ids)))
        )
        return int(result.scalar_one() or 0)


# ------------------------------------------------------------ observations --


class ObservationRepository:
    """Immutable extraction events. Append-only by discipline and by design —
    there is no update method, and there will not be one."""

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    async def record(
        self,
        *,
        investigation_id: uuid.UUID,
        method: ObservationMethod | str,
        raw_value: str,
        extractor_version: str,
        page_id: uuid.UUID | None = None,
        image_id: uuid.UUID | None = None,
        normalized_value: str = "",
        char_start: int | None = None,
        char_end: int | None = None,
        context_snippet: str = "",
        method_confidence: float | None = None,
    ) -> Observation:
        if not raw_value.strip():
            raise ValidationError("An observation must carry a non-empty value.")
        if page_id is None and image_id is None:
            raise ValidationError(
                "An observation must be anchored to a page or an image. "
                "An extraction with no source is not evidence."
            )

        observation = Observation(
            investigation_id=investigation_id,
            page_id=page_id,
            image_id=image_id,
            method=str(method),
            raw_value=raw_value,
            normalized_value=normalized_value or raw_value.strip().casefold(),
            char_start=char_start,
            char_end=char_end,
            context_snippet=context_snippet[:2000],
            extractor_version=extractor_version,
            method_confidence=method_confidence,
            extracted_at=self.clock.now(),
        )
        self.session.add(observation)
        await self.session.flush()
        return observation

    async def for_page(self, page_id: uuid.UUID) -> Sequence[Observation]:
        result = await self.session.execute(
            select(Observation)
            .where(Observation.page_id == page_id)
            .order_by(Observation.id)
        )
        return result.scalars().all()

    async def by_method(
        self, investigation_id: uuid.UUID, method: str
    ) -> Sequence[Observation]:
        """Every observation produced by one extraction method, in collection
        order. UUIDv7 ids sort chronologically, so no timestamp join is needed."""
        result = await self.session.execute(
            select(Observation)
            .where(
                Observation.investigation_id == investigation_id,
                Observation.method == str(method),
            )
            .order_by(Observation.id)
        )
        return result.scalars().all()

    async def get_many(self, ids: Sequence[uuid.UUID]) -> Sequence[Observation]:
        if not ids:
            return []
        result = await self.session.execute(
            select(Observation).where(Observation.id.in_(list(ids)))
        )
        return result.scalars().all()


# ---------------------------------------------------------------- entities --


class EntityRepository:
    """Resolved entities and their aliases.

    Merging is deliberately conservative: an uncertain pair is flagged with
    `possible_duplicate_of` rather than merged. Over-merging attributes one
    person's facts to another and is close to unrecoverable once reports have
    been issued; under-merging is a visible inconvenience an investigator fixes
    in one click.
    """

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    @staticmethod
    def normalize_key(value: str) -> str:
        import unicodedata

        collapsed = " ".join(unicodedata.normalize("NFKC", value).split())
        return collapsed.casefold()

    async def upsert(
        self,
        *,
        investigation_id: uuid.UUID,
        entity_type: EntityType | str,
        canonical_name: str,
        surface_form: str | None = None,
    ) -> Entity:
        key = self.normalize_key(canonical_name)
        if not key:
            raise ValidationError("An entity must have a non-empty name.")

        entity = (
            await self.session.execute(
                select(Entity).where(
                    Entity.investigation_id == investigation_id,
                    Entity.type == str(entity_type),
                    Entity.normalized_key == key,
                )
            )
        ).scalar_one_or_none()

        now = self.clock.now()
        if entity is None:
            entity = Entity(
                investigation_id=investigation_id,
                type=str(entity_type),
                canonical_name=canonical_name.strip(),
                normalized_key=key,
                verification_state=VerificationState.MACHINE_PROPOSED,
                first_seen_at=now,
                last_seen_at=now,
            )
            self.session.add(entity)
            await self.session.flush()
        else:
            entity.last_seen_at = now

        await self._touch_alias(entity, surface_form or canonical_name)
        return entity

    async def _touch_alias(self, entity: Entity, surface_form: str) -> None:
        form = surface_form.strip()[:500]
        if not form:
            return
        alias = (
            await self.session.execute(
                select(EntityAlias).where(
                    EntityAlias.entity_id == entity.id, EntityAlias.surface_form == form
                )
            )
        ).scalar_one_or_none()
        if alias is None:
            self.session.add(EntityAlias(entity_id=entity.id, surface_form=form))
        else:
            alias.occurrences += 1
        await self.session.flush()

    async def link_mention(
        self, *, observation: Observation, entity: Entity, surface_form: str
    ) -> Mention:
        mention = Mention(
            observation_id=observation.id,
            entity_id=entity.id,
            entity_type=entity.type,
            surface_form=surface_form.strip()[:500],
        )
        self.session.add(mention)
        await self.session.flush()
        return mention

    async def flag_possible_duplicate(self, entity: Entity, other: Entity) -> None:
        """Record a suspected duplicate without merging.

        The pair stays separate and visible; a human rules on it through the
        review queue.
        """
        if entity.id == other.id:
            raise ValidationError("An entity cannot duplicate itself.")
        entity.possible_duplicate_of = other.id
        await self.session.flush()

    async def set_verification(
        self, entity: Entity, state: VerificationState | str
    ) -> Entity:
        entity.verification_state = str(state)
        await self.session.flush()
        return entity


# ------------------------------------------------------------------- facts --


class FactRepository:
    """Facts, their evidence links and their two independent status axes."""

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    async def assert_fact(
        self,
        *,
        investigation_id: uuid.UUID,
        entity: Entity,
        attribute: str,
        value: str,
        observations: Sequence[Observation],
        status: FactStatus | str,
        confidence: ConfidenceTier | str,
        confidence_score: float,
        confidence_explanation: dict[str, Any],
        independent_source_count: int,
        scorer_version: str,
        conflict_group_id: uuid.UUID | None = None,
    ) -> Fact:
        """Write a fact together with its evidence and its explanation.

        The explanation is a required argument rather than an optional one:
        the database refuses a fact whose confidence cannot be explained, and
        making it optional here would just move that failure later.
        """
        if not observations:
            raise ValidationError(
                f"Fact {attribute}={value!r} has no supporting observations. "
                "The system never outputs an unsupported conclusion."
            )

        factors = confidence_explanation.get("factors") or []
        if not factors:
            raise ValidationError(
                "A confidence value must ship with at least one explanatory factor. "
                "A number without an explanation is never displayed, so it is never stored."
            )

        fact = Fact(
            investigation_id=investigation_id,
            entity_id=entity.id,
            attribute=attribute.strip(),
            value=value.strip(),
            normalized_value=value.strip().casefold(),
            status=str(status),
            conflict_group_id=conflict_group_id,
            classification=FactClassification.UNVERIFIED,
            confidence=str(confidence),
            confidence_score=confidence_score,
            confidence_explanation=confidence_explanation,
            confidence_factor_count=len(factors),
            independent_source_count=independent_source_count,
            observation_count=len(observations),
            first_asserted_at=min(o.extracted_at for o in observations),
            computed_at=self.clock.now(),
            scorer_version=scorer_version,
        )
        self.session.add(fact)
        await self.session.flush()

        for observation in observations:
            self.session.add(
                FactEvidence(fact_id=fact.id, observation_id=observation.id)
            )
        await self.session.flush()
        return fact

    async def get(self, fact_id: uuid.UUID) -> Fact:
        fact = (
            await self.session.execute(select(Fact).where(Fact.id == fact_id))
        ).scalar_one_or_none()
        if fact is None:
            raise NotFoundError(f"Fact {fact_id} not found.")
        return fact

    async def evidence_chain(self, fact_id: uuid.UUID) -> Sequence[Observation]:
        """Every observation supporting a fact, in collection order.

        UUIDv7 ids sort chronologically, so this is ordered without a join to a
        timestamp column.
        """
        result = await self.session.execute(
            select(Observation)
            .join(FactEvidence, FactEvidence.observation_id == Observation.id)
            .where(FactEvidence.fact_id == fact_id)
            .order_by(Observation.id)
        )
        return result.scalars().all()

    async def classify(
        self,
        fact: Fact,
        classification: FactClassification | str,
        *,
        user_id: uuid.UUID,
        note: str = "",
    ) -> Fact:
        """Record a human judgement.

        Touches only the investigative axis. `status` — the evidential axis — is
        machine-derived and is never overwritten here, because a human doubting
        a fact does not change how many sources asserted it.
        """
        fact.classification = str(classification)
        fact.classified_by = user_id
        fact.classified_at = self.clock.now()
        fact.classification_note = note.strip()
        await self.session.flush()
        return fact

    async def conflicting(self, conflict_group_id: uuid.UUID) -> Sequence[Fact]:
        """All competing values in a conflict group.

        Every variant is retained with its own evidence; none is deleted. The
        investigator adjudicates — a tool that silently picks a winner is hiding
        evidence.
        """
        result = await self.session.execute(
            select(Fact)
            .where(Fact.conflict_group_id == conflict_group_id)
            .order_by(Fact.confidence_score.desc())
        )
        return result.scalars().all()

    async def clear_derived(self, investigation_id: uuid.UUID) -> int:
        """Drop computed facts ahead of a re-score.

        Safe precisely because facts are derived: observations are untouched, so
        stages 10-13 rebuild from the same evidence. This is what makes an
        algorithm improvement replayable without re-crawling or new API spend.
        """
        result = await self.session.execute(
            delete(Fact).where(Fact.investigation_id == investigation_id)
        )
        await self.session.flush()
        return int(result.rowcount or 0)


# ------------------------------------------------------------------- graph --


class GraphRepository:
    """Nodes and evidence-backed edges."""

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    async def upsert_node(
        self,
        *,
        investigation_id: uuid.UUID,
        node_type: NodeType | str,
        ref_table: str,
        ref_id: uuid.UUID,
        label: str,
        asserted_by_page_id: uuid.UUID | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> GraphNode:
        if str(node_type) == NodeType.PERSON and asserted_by_page_id is None:
            raise ValidationError(
                "A PERSON node requires the page that explicitly named them. "
                "Identity enters the graph only as a quoted claim with a citation, "
                "never as an inference."
            )

        existing = (
            await self.session.execute(
                select(GraphNode).where(
                    GraphNode.investigation_id == investigation_id,
                    GraphNode.node_type == str(node_type),
                    GraphNode.ref_id == ref_id,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing

        node = GraphNode(
            investigation_id=investigation_id,
            node_type=str(node_type),
            ref_table=ref_table,
            ref_id=ref_id,
            label=label[:500],
            asserted_by_page_id=asserted_by_page_id,
            attributes=attributes or {},
        )
        self.session.add(node)
        await self.session.flush()
        return node

    async def connect(
        self,
        *,
        investigation_id: uuid.UUID,
        from_node: GraphNode,
        to_node: GraphNode,
        edge_type: str,
        derivation: str,
        observations: Sequence[Observation | uuid.UUID],
        confidence: ConfidenceTier | str = ConfidenceTier.LOW,
    ) -> GraphEdge:
        """Draw an edge. Requires at least one supporting observation."""
        ids = [
            str(o.id if isinstance(o, Observation) else o) for o in observations
        ]
        if not ids:
            raise ValidationError(
                f"Edge {edge_type} from {from_node.label!r} to {to_node.label!r} cites no "
                "observation. Every edge must be evidence-backed, or the graph becomes "
                "a second, unsourced source of truth."
            )
        if not derivation.strip():
            raise ValidationError(
                "An edge must record the rule that produced it, so a reviewer can ask "
                "not only what supports it but why it was drawn."
            )

        edge = GraphEdge(
            investigation_id=investigation_id,
            from_node_id=from_node.id,
            to_node_id=to_node.id,
            edge_type=str(edge_type),
            derivation=derivation.strip(),
            evidence_observation_ids=ids,
            confidence=str(confidence),
            created_at=self.clock.now(),
        )
        self.session.add(edge)
        await self.session.flush()
        return edge

    async def neighbours(self, node_id: uuid.UUID) -> Sequence[GraphEdge]:
        result = await self.session.execute(
            select(GraphEdge).where(
                (GraphEdge.from_node_id == node_id) | (GraphEdge.to_node_id == node_id)
            )
        )
        return result.scalars().all()


# ------------------------------------------------------------------ review --


class ReviewRepository:
    """The queue that separates machine observation from human interpretation."""

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    async def propose(
        self,
        *,
        investigation_id: uuid.UUID,
        kind: str,
        subject_type: str,
        subject_id: uuid.UUID,
        proposal: dict[str, Any],
        rationale: dict[str, Any],
        priority: int = 0,
    ) -> ReviewItem:
        if not rationale:
            raise ValidationError(
                "A review item must carry its rationale, including supporting "
                "observation ids. A reviewer rules on evidence, not on a bare suggestion."
            )
        item = ReviewItem(
            investigation_id=investigation_id,
            kind=str(kind),
            subject_type=subject_type,
            subject_id=subject_id,
            proposal=proposal,
            rationale=rationale,
            priority=priority,
            status=ReviewStatus.PENDING,
            created_at=self.clock.now(),
        )
        self.session.add(item)
        await self.session.flush()
        return item

    async def get(self, item_id: uuid.UUID) -> ReviewItem:
        item = (
            await self.session.execute(select(ReviewItem).where(ReviewItem.id == item_id))
        ).scalar_one_or_none()
        if item is None:
            raise NotFoundError(f"Review item {item_id} not found.")
        return item

    async def decide(
        self,
        item: ReviewItem,
        status: ReviewStatus | str,
        *,
        user_id: uuid.UUID,
        note: str = "",
    ) -> ReviewItem:
        """Record a ruling.

        A rejection never deletes the underlying observation: "the machine saw
        this and a human disagreed" is itself a finding worth preserving.
        """
        if str(status) == ReviewStatus.PENDING:
            raise ValidationError("A decision cannot set the item back to PENDING.")
        item.status = str(status)
        item.decided_by = user_id
        item.decided_at = self.clock.now()
        item.decision_note = note.strip()
        await self.session.flush()
        return item

    async def pending(
        self, investigation_id: uuid.UUID, limit: int = 50
    ) -> Sequence[ReviewItem]:
        result = await self.session.execute(
            select(ReviewItem)
            .where(
                ReviewItem.investigation_id == investigation_id,
                ReviewItem.status == ReviewStatus.PENDING,
            )
            .order_by(ReviewItem.priority.desc(), ReviewItem.id)
            .limit(limit)
        )
        return result.scalars().all()

    async def pending_count(self, investigation_id: uuid.UUID) -> int:
        """Gates `UNDER_REVIEW → COMPLETED`: a case cannot be completed with
        unreviewed machine output sitting in it (REVISION_3 §3)."""
        result = await self.session.execute(
            select(func.count())
            .select_from(ReviewItem)
            .where(
                ReviewItem.investigation_id == investigation_id,
                ReviewItem.status == ReviewStatus.PENDING,
            )
        )
        return int(result.scalar_one() or 0)


__all__ = [
    "EntityRepository",
    "ImageRepository",
    "FactRepository",
    "GraphRepository",
    "ObservationRepository",
    "ReviewRepository",
    "SourceRepository",
]


# ------------------------------------------------------------------ images --


class ImageRepository:
    """Images as evidence objects — the uploaded probe and every discovered
    copy alike."""

    def __init__(self, session: AsyncSession, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    async def add(
        self,
        *,
        investigation_id: uuid.UUID,
        role: str,
        sha256: str,
        phash: str,
        storage_key: str,
        dhash: str | None = None,
        whash: str | None = None,
        width: int | None = None,
        height: int | None = None,
        file_size: int | None = None,
        mime_type: str = "",
        exif: dict[str, Any] | None = None,
        source_page_id: uuid.UUID | None = None,
        source_image_url: str = "",
    ) -> Image:
        now = self.clock.now()
        image = Image(
            investigation_id=investigation_id,
            role=role,
            sha256=sha256,
            phash=phash,
            dhash=dhash,
            whash=whash,
            width=width,
            height=height,
            file_size=file_size,
            mime_type=mime_type,
            exif=exif or {},
            storage_key=storage_key,
            source_page_id=source_page_id,
            source_image_url=source_image_url,
            discovered_at=now,
            downloaded_at=now,
            created_at=now,
        )
        self.session.add(image)
        await self.session.flush()
        return image

    async def get(self, image_id: uuid.UUID) -> Image:
        image = (
            await self.session.execute(select(Image).where(Image.id == image_id))
        ).scalar_one_or_none()
        if image is None:
            raise NotFoundError(f"Image {image_id} not found.")
        return image

    async def find_by_sha256(
        self, investigation_id: uuid.UUID, sha256: str
    ) -> Image | None:
        """Detect a re-upload of identical bytes.

        Returned rather than rejected: uploading the same file twice is a
        harmless mistake, and silently creating a second probe would double-count
        it in every later stage.
        """
        result = await self.session.execute(
            select(Image).where(
                Image.investigation_id == investigation_id, Image.sha256 == sha256
            )
        )
        return result.scalars().first()

    async def list_for_investigation(
        self, investigation_id: uuid.UUID, *, role: str | None = None
    ) -> Sequence[Image]:
        statement = select(Image).where(Image.investigation_id == investigation_id)
        if role:
            statement = statement.where(Image.role == role)
        result = await self.session.execute(statement.order_by(Image.id))
        return result.scalars().all()
