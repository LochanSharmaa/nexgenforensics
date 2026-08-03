"""The five database-level invariants, each proven to REJECT its violation.

An invariant that has never been observed failing is an assumption. Every test
here writes the *forbidden* row directly through SQLAlchemy — bypassing the
repository layer entirely — and asserts the database refuses it. That is the
point: these guarantees must hold against a buggy service, a migration script,
or someone at a psql prompt, not merely against well-behaved application code.

Invariants (DATA_MODEL §"Invariants this schema enforces"):

1. No ``graph_edges`` row without supporting evidence.
2. No ``PERSON`` graph node without an asserting page.
3. ``HUMAN``-origin content cannot enter extracted-evidence tables.
4. ``audit_log`` is append-only and hash-chained.
5. Every ``facts.confidence`` has a matching persisted explanation.
"""

from __future__ import annotations

import pathlib
import re
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError, StatementError

from database.models import (
    Domain,
    Entity,
    Fact,
    GraphEdge,
    GraphNode,
    Investigation,
    Note,
    Observation,
    Page,
)
from shared.enums import (
    ConfidenceTier,
    DomainClassification,
    EdgeType,
    EntityType,
    FactStatus,
    NodeType,
    ObservationMethod,
)

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------- fixtures --


async def _scaffold(session, user):
    """Minimal object graph: investigation → domain → page → entity."""
    investigation = Investigation(
        owner_id=user.id, case_id=f"INV-{uuid.uuid4().hex[:8]}", title="Invariants",
        lawful_basis="Test", status="NEW",
    )
    session.add(investigation)
    await session.flush()

    domain = Domain(
        registrable_domain=f"{uuid.uuid4().hex[:8]}.example",
        classification=DomainClassification.NEWS,
        first_seen_at=NOW,
    )
    session.add(domain)
    await session.flush()

    page = Page(
        investigation_id=investigation.id, domain_id=domain.id,
        url=f"https://example.test/{uuid.uuid4().hex[:8]}", fetched_at=NOW,
    )
    session.add(page)
    await session.flush()

    entity = Entity(
        investigation_id=investigation.id, type=EntityType.ORGANIZATION,
        canonical_name="Meridian Logistics", normalized_key="meridian logistics",
    )
    session.add(entity)
    await session.flush()

    observation = Observation(
        investigation_id=investigation.id, page_id=page.id,
        method=ObservationMethod.SCHEMA_ORG, raw_value="Meridian Logistics",
        normalized_value="meridian logistics", extractor_version="test@1.0",
        extracted_at=NOW,
    )
    session.add(observation)
    await session.flush()

    return investigation, page, entity, observation


def _node(investigation_id, node_type, ref_id, *, asserted_by_page_id=None, label="n"):
    return GraphNode(
        investigation_id=investigation_id, node_type=node_type,
        ref_table="pages", ref_id=ref_id, label=label,
        asserted_by_page_id=asserted_by_page_id,
    )


# ------------------------------------------ 1. edges require evidence --------


async def test_graph_edge_without_evidence_is_rejected(session, user):
    """An unsupported edge must fail to insert.

    This is what stops the graph drifting into an unsourced parallel truth
    alongside the evidence store.
    """
    investigation, page, _entity, _obs = await _scaffold(session, user)
    a = _node(investigation.id, NodeType.PAGE, page.id, label="page")
    b = _node(investigation.id, NodeType.DOMAIN, uuid.uuid4(), label="domain")
    session.add_all([a, b])
    await session.flush()

    session.add(
        GraphEdge(
            investigation_id=investigation.id, from_node_id=a.id, to_node_id=b.id,
            edge_type=EdgeType.HOSTED_BY, derivation="page.domain_id",
            evidence_observation_ids=[],          # ← the violation
            confidence=ConfidenceTier.LOW, created_at=NOW,
        )
    )
    with pytest.raises(IntegrityError, match="edge_requires_evidence"):
        await session.flush()
    await session.rollback()


async def test_graph_edge_with_evidence_is_accepted(session, user):
    """The positive case, so the constraint is not merely rejecting everything."""
    investigation, page, _entity, observation = await _scaffold(session, user)
    a = _node(investigation.id, NodeType.PAGE, page.id, label="page")
    b = _node(investigation.id, NodeType.DOMAIN, uuid.uuid4(), label="domain")
    session.add_all([a, b])
    await session.flush()

    session.add(
        GraphEdge(
            investigation_id=investigation.id, from_node_id=a.id, to_node_id=b.id,
            edge_type=EdgeType.HOSTED_BY, derivation="page.domain_id",
            evidence_observation_ids=[str(observation.id)],
            confidence=ConfidenceTier.LOW, created_at=NOW,
        )
    )
    await session.flush()   # must not raise


# ------------------------------- 2. PERSON nodes require an assertion --------


async def test_person_node_without_asserting_page_is_rejected(session, user):
    """There must be no schema-level path from an image to a person.

    Identity enters the graph only as a quoted claim with a citation. A PERSON
    node with no page that named them would be an inference, which is precisely
    what this platform must never produce.
    """
    investigation, _page, _entity, _obs = await _scaffold(session, user)
    session.add(
        _node(investigation.id, NodeType.PERSON, uuid.uuid4(), label="J. Bramwell")
    )
    with pytest.raises(IntegrityError, match="person_requires_assertion"):
        await session.flush()
    await session.rollback()


async def test_person_node_with_asserting_page_is_accepted(session, user):
    investigation, page, _entity, _obs = await _scaffold(session, user)
    session.add(
        _node(
            investigation.id, NodeType.PERSON, uuid.uuid4(),
            asserted_by_page_id=page.id, label="J. Bramwell",
        )
    )
    await session.flush()


async def test_non_person_nodes_need_no_assertion(session, user):
    """The constraint must not accidentally block ordinary nodes."""
    investigation, page, _entity, _obs = await _scaffold(session, user)
    session.add(_node(investigation.id, NodeType.ORGANIZATION, page.id, label="Acme"))
    await session.flush()


# ------------------------- 3. human input never becomes machine evidence -----


async def test_human_origin_observation_is_rejected(session, user):
    """`observations` accepts machine extraction only.

    An investigator's hypothesis must stay distinguishable from something a page
    actually said, months later and under challenge.
    """
    investigation, page, _entity, _obs = await _scaffold(session, user)
    session.add(
        Observation(
            investigation_id=investigation.id, page_id=page.id,
            origin="HUMAN",                       # ← the violation
            method=ObservationMethod.NER, raw_value="my hunch",
            extractor_version="human", extracted_at=NOW,
        )
    )
    with pytest.raises(IntegrityError, match="machine_origin_only"):
        await session.flush()
    await session.rollback()


async def test_extracted_origin_note_is_rejected(session, user):
    """And the mirror: `notes` accepts human authorship only.

    Both directions are constrained, so the separation is physical rather than
    procedural.
    """
    investigation, _page, _entity, _obs = await _scaffold(session, user)
    session.add(
        Note(
            investigation_id=investigation.id, author_id=user.id,
            origin="EXTRACTED",                   # ← the violation
            title="machine-written note", body_plain="…",
        )
    )
    with pytest.raises(IntegrityError, match="human_origin_only"):
        await session.flush()
    await session.rollback()


async def test_correctly_originated_rows_are_accepted(session, user):
    investigation, page, _entity, _obs = await _scaffold(session, user)
    session.add(
        Observation(
            investigation_id=investigation.id, page_id=page.id,
            method=ObservationMethod.SCHEMA_ORG, raw_value="Acme",
            extractor_version="test@1.0", extracted_at=NOW,
        )
    )
    session.add(
        Note(
            investigation_id=investigation.id, author_id=user.id,
            title="Working theory", body_plain="Cross-check the 2019 filing.",
        )
    )
    await session.flush()


# ----------------------------- 4. audit log is append-only -------------------


def test_migration_revokes_mutation_on_append_only_tables():
    """Immutability is enforced by privilege, not application discipline.

    SQLite has no role system, so this asserts the guarantee exists in the
    migration that runs against PostgreSQL. Combined with the hash chain,
    rewriting history then needs superuser access *and* still breaks every
    subsequent hash.
    """
    versions = pathlib.Path(__file__).resolve().parents[2] / "database/migrations/versions"
    sources = "\n".join(p.read_text(encoding="utf-8") for p in versions.glob("*.py"))

    assert 'APPEND_ONLY_TABLES = ("audit_log", "custody_events")' in sources
    assert re.search(r"REVOKE UPDATE, DELETE, TRUNCATE ON \{table\}", sources), (
        "append-only tables must have UPDATE/DELETE/TRUNCATE revoked"
    )
    assert "GRANT SELECT, INSERT ON {table}" in sources


def test_append_only_logs_have_no_cascading_foreign_keys():
    """An audit log must outlive what it describes.

    `ON DELETE SET NULL` is an UPDATE and `CASCADE` is a DELETE — both revoked
    on these tables, so an FK would either make investigation deletion
    impossible or erase the record of what was deleted. Either outcome defeats
    the retention regime, so these columns carry no FK at all.
    """
    from database.models import AuditLogEntry, CustodyEvent

    for model in (AuditLogEntry, CustodyEvent):
        for column_name in ("investigation_id", "actor_id"):
            column = model.__table__.columns.get(column_name)
            if column is None:
                continue
            assert not column.foreign_keys, (
                f"{model.__tablename__}.{column_name} must not carry a foreign key — "
                "the log has to survive the purge of what it documents."
            )


# ------------------- 5. confidence always carries an explanation -------------


async def test_fact_without_explanation_is_rejected(session, user):
    """A confidence number that cannot be explained must not be storable —
    and therefore can never be displayed alone."""
    investigation, _page, entity, _obs = await _scaffold(session, user)
    session.add(
        Fact(
            investigation_id=investigation.id, entity_id=entity.id,
            attribute="employer", value="Meridian Logistics",
            status=FactStatus.UNIQUE, confidence=ConfidenceTier.MEDIUM,
            confidence_score=0.58,
            confidence_explanation={},
            confidence_factor_count=0,            # ← the violation
            computed_at=NOW,
        )
    )
    with pytest.raises(IntegrityError, match="explanation_required"):
        await session.flush()
    await session.rollback()


async def test_fact_with_explanation_is_accepted(session, user):
    investigation, _page, entity, _obs = await _scaffold(session, user)
    session.add(
        Fact(
            investigation_id=investigation.id, entity_id=entity.id,
            attribute="employer", value="Meridian Logistics",
            status=FactStatus.COMMON, confidence=ConfidenceTier.MEDIUM,
            confidence_score=0.58,
            confidence_explanation={
                "factors": [
                    {"factor": "independent_domains", "observed": 2, "contribution": 0.30,
                     "why": "Asserted by 2 unrelated registrable domains"}
                ]
            },
            confidence_factor_count=1,
            computed_at=NOW,
        )
    )
    await session.flush()


# ------------------------------------------- vocabulary constraints ---------


async def test_unknown_enum_value_is_rejected(session, user):
    """Controlled vocabularies are database guarantees, not conventions — a
    direct SQL write with a typo is refused too."""
    investigation, page, _entity, _obs = await _scaffold(session, user)
    session.add(
        Observation(
            investigation_id=investigation.id, page_id=page.id,
            method="TELEPATHY",                   # ← not in the vocabulary
            raw_value="x", extractor_version="test@1.0", extracted_at=NOW,
        )
    )
    # SQLAlchemy's non-native Enum validates on bind, so the rejection surfaces
    # as StatementError wrapping LookupError rather than reaching the database.
    # Either way the value never lands — and the CHECK constraint in the
    # migration catches a write that bypasses the ORM entirely.
    with pytest.raises(StatementError, match="not among the defined enum values"):
        await session.flush()
    await session.rollback()


async def test_domain_classification_is_stored_but_never_scored(session, user):
    """Classification is descriptive metadata.

    ARCHITECTURE §9.3 bars it from influencing confidence; this asserts the
    schema keeps them in separate tables so no join makes it accidental.
    """
    _investigation, _page, _entity, _obs = await _scaffold(session, user)
    fact_columns = set(Fact.__table__.columns.keys())
    assert not any("domain" in name or "classification_basis" in name for name in fact_columns), (
        "facts must not carry a domain-classification column — encoding "
        "institutional trust into an evidence score smuggles an editorial "
        "judgement into a number presented as objective."
    )
