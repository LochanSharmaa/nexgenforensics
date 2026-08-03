"""Evidence repositories: the write paths that build and rebuild findings."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from database.repositories import (
    EntityRepository,
    FactRepository,
    GraphRepository,
    InvestigationRepository,
    ObservationRepository,
    ReviewRepository,
    SourceRepository,
)
from shared.enums import (
    ConfidenceTier,
    EdgeType,
    EntityType,
    FactClassification,
    FactStatus,
    NodeType,
    ObservationMethod,
    ReviewKind,
    ReviewStatus,
    VerificationState,
)
from shared.errors import ValidationError

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)

EXPLANATION = {
    "factors": [
        {
            "factor": "independent_domains",
            "observed": 2,
            "contribution": 0.30,
            "why": "Asserted by 2 unrelated registrable domains",
        }
    ]
}


@pytest.fixture
async def case(session, clock, user):
    """An investigation with two pages on genuinely different domains."""
    investigations = InvestigationRepository(session, clock)
    sources = SourceRepository(session, clock)

    investigation = await investigations.create(
        owner_id=user.id, case_id=f"EV-{uuid.uuid4().hex[:6]}",
        title="Evidence", lawful_basis="Test engagement",
    )
    first_domain = await sources.upsert_domain("meridian.example", classification="COMPANY")
    second_domain = await sources.upsert_domain("dailynews.example", classification="NEWS")
    first = await sources.add_page(
        investigation_id=investigation.id, domain_id=first_domain.id,
        url="https://meridian.example/leadership", title="Leadership",
    )
    second = await sources.add_page(
        investigation_id=investigation.id, domain_id=second_domain.id,
        url="https://dailynews.example/2026/award", title="Award",
    )
    return investigation, first, second


# ----------------------------------------------------------------- sources --


async def test_domains_are_deduplicated_and_normalized(session, clock):
    sources = SourceRepository(session, clock)
    first = await sources.upsert_domain("Example.COM")
    second = await sources.upsert_domain("example.com.")
    assert first.id == second.id
    assert first.registrable_domain == "example.com"


async def test_domain_classification_fills_in_once_known(session, clock):
    sources = SourceRepository(session, clock)
    domain = await sources.upsert_domain("late.example")
    assert domain.classification == "UNKNOWN"
    again = await sources.upsert_domain("late.example", classification="NEWS")
    assert again.classification == "NEWS"


async def test_independent_domain_count_counts_domains_not_pages(session, clock, case):
    """Two pages on one domain are one source, not two."""
    investigation, first, _second = case
    sources = SourceRepository(session, clock)
    observations = ObservationRepository(session, clock)

    same_domain_page = await sources.add_page(
        investigation_id=investigation.id, domain_id=first.domain_id,
        url="https://meridian.example/about",
    )
    a = await observations.record(
        investigation_id=investigation.id, page_id=first.id,
        method=ObservationMethod.SCHEMA_ORG, raw_value="Meridian", extractor_version="t@1",
    )
    b = await observations.record(
        investigation_id=investigation.id, page_id=same_domain_page.id,
        method=ObservationMethod.META, raw_value="Meridian", extractor_version="t@1",
    )
    assert await sources.independent_domain_count([a.id, b.id]) == 1


async def test_independent_domain_count_sees_distinct_domains(session, clock, case):
    investigation, first, second = case
    sources = SourceRepository(session, clock)
    observations = ObservationRepository(session, clock)
    a = await observations.record(
        investigation_id=investigation.id, page_id=first.id,
        method=ObservationMethod.SCHEMA_ORG, raw_value="Meridian", extractor_version="t@1",
    )
    b = await observations.record(
        investigation_id=investigation.id, page_id=second.id,
        method=ObservationMethod.NER, raw_value="Meridian", extractor_version="t@1",
    )
    assert await sources.independent_domain_count([a.id, b.id]) == 2


# ------------------------------------------------------------ observations --


async def test_observation_must_be_anchored_to_a_source(session, clock, case):
    """An extraction with no source is not evidence."""
    investigation, _first, _second = case
    observations = ObservationRepository(session, clock)
    with pytest.raises(ValidationError, match="anchored"):
        await observations.record(
            investigation_id=investigation.id,
            method=ObservationMethod.NER, raw_value="orphan", extractor_version="t@1",
        )


async def test_observation_records_offsets_and_version(session, clock, case):
    investigation, first, _second = case
    observations = ObservationRepository(session, clock)
    observation = await observations.record(
        investigation_id=investigation.id, page_id=first.id,
        method=ObservationMethod.SCHEMA_ORG, raw_value="Meridian Logistics",
        char_start=1204, char_end=1222, extractor_version="extract@2.1.0",
        context_snippet="…worksFor Meridian Logistics…",
    )
    assert (observation.char_start, observation.char_end) == (1204, 1222)
    assert observation.extractor_version == "extract@2.1.0"
    assert observation.normalized_value == "meridian logistics"
    assert observation.origin == "EXTRACTED"


async def test_repository_exposes_no_update_path():
    """Observations are immutable. The absence of a mutator is the guarantee."""
    forbidden = {"update", "edit", "modify", "set_value", "amend"}
    methods = {name for name in dir(ObservationRepository) if not name.startswith("_")}
    leaked = methods & forbidden
    assert not leaked, f"ObservationRepository exposes mutators: {leaked}"


# ---------------------------------------------------------------- entities --


async def test_entity_upsert_is_idempotent_and_tracks_aliases(session, clock, case):
    investigation, _first, _second = case
    entities = EntityRepository(session, clock)

    first = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.PERSON,
        canonical_name="John Smith", surface_form="John Smith",
    )
    second = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.PERSON,
        canonical_name="JOHN SMITH", surface_form="JOHN SMITH",
    )
    assert first.id == second.id, "case variants must resolve to one entity"


async def test_different_people_do_not_collapse(session, clock, case):
    """The failure that matters most: never merge two different people."""
    investigation, _first, _second = case
    entities = EntityRepository(session, clock)
    john = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.PERSON,
        canonical_name="John Smith",
    )
    jane = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.PERSON,
        canonical_name="Jane Smith",
    )
    assert john.id != jane.id


async def test_same_name_different_type_stays_separate(session, clock, case):
    investigation, _first, _second = case
    entities = EntityRepository(session, clock)
    person = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.PERSON,
        canonical_name="Meridian",
    )
    org = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.ORGANIZATION,
        canonical_name="Meridian",
    )
    assert person.id != org.id


async def test_uncertain_pairs_are_flagged_not_merged(session, clock, case):
    """Conservative by default: over-merging attributes one person's facts to
    another and is close to unrecoverable once a report has been issued."""
    investigation, _first, _second = case
    entities = EntityRepository(session, clock)
    a = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.PERSON,
        canonical_name="J. Smith",
    )
    b = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.PERSON,
        canonical_name="John Smith",
    )
    await entities.flag_possible_duplicate(a, b)
    assert a.possible_duplicate_of == b.id
    assert a.id != b.id, "flagging must not merge"


async def test_entities_start_machine_proposed(session, clock, case):
    investigation, _first, _second = case
    entities = EntityRepository(session, clock)
    entity = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.ORGANIZATION,
        canonical_name="Meridian Logistics",
    )
    assert entity.verification_state == VerificationState.MACHINE_PROPOSED


# ------------------------------------------------------------------- facts --


async def _two_observations(session, clock, case):
    investigation, first, second = case
    observations = ObservationRepository(session, clock)
    a = await observations.record(
        investigation_id=investigation.id, page_id=first.id,
        method=ObservationMethod.SCHEMA_ORG, raw_value="Meridian Logistics",
        extractor_version="t@1",
    )
    b = await observations.record(
        investigation_id=investigation.id, page_id=second.id,
        method=ObservationMethod.NER, raw_value="Meridian Logistics",
        extractor_version="t@1",
    )
    return investigation, [a, b]


async def test_fact_requires_supporting_observations(session, clock, case):
    investigation, _first, _second = case
    entities = EntityRepository(session, clock)
    facts = FactRepository(session, clock)
    entity = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.PERSON,
        canonical_name="Jordan Bramwell",
    )
    with pytest.raises(ValidationError, match="unsupported conclusion"):
        await facts.assert_fact(
            investigation_id=investigation.id, entity=entity, attribute="employer",
            value="Meridian", observations=[], status=FactStatus.UNIQUE,
            confidence=ConfidenceTier.LOW, confidence_score=0.1,
            confidence_explanation=EXPLANATION, independent_source_count=0,
            scorer_version="s@1",
        )


async def test_fact_requires_an_explanation(session, clock, case):
    investigation, observations = await _two_observations(session, clock, case)
    entities = EntityRepository(session, clock)
    facts = FactRepository(session, clock)
    entity = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.PERSON,
        canonical_name="Jordan Bramwell",
    )
    with pytest.raises(ValidationError, match="explanatory factor"):
        await facts.assert_fact(
            investigation_id=investigation.id, entity=entity, attribute="employer",
            value="Meridian", observations=observations, status=FactStatus.COMMON,
            confidence=ConfidenceTier.MEDIUM, confidence_score=0.58,
            confidence_explanation={"factors": []},
            independent_source_count=2, scorer_version="s@1",
        )


async def test_evidence_chain_round_trips(session, clock, case):
    """The traceability guarantee: a fact leads back to its observations."""
    investigation, observations = await _two_observations(session, clock, case)
    entities = EntityRepository(session, clock)
    facts = FactRepository(session, clock)
    entity = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.PERSON,
        canonical_name="Jordan Bramwell",
    )
    fact = await facts.assert_fact(
        investigation_id=investigation.id, entity=entity, attribute="employer",
        value="Meridian Logistics", observations=observations, status=FactStatus.COMMON,
        confidence=ConfidenceTier.MEDIUM, confidence_score=0.58,
        confidence_explanation=EXPLANATION, independent_source_count=2,
        scorer_version="s@1",
    )
    chain = await facts.evidence_chain(fact.id)
    assert {o.id for o in chain} == {o.id for o in observations}
    assert fact.observation_count == 2
    assert fact.confidence_factor_count == 1
    assert fact.first_asserted_at == min(o.extracted_at for o in observations)


async def test_conflicting_values_are_all_retained(session, clock, case):
    """The system never silently picks a winner."""
    investigation, observations = await _two_observations(session, clock, case)
    entities = EntityRepository(session, clock)
    facts = FactRepository(session, clock)
    entity = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.PERSON,
        canonical_name="Jordan Bramwell",
    )
    group = uuid.uuid4()
    competing_values = ("Meridian Logistics", "Northwind Freight")
    for value, observation in zip(competing_values, observations, strict=True):
        await facts.assert_fact(
            investigation_id=investigation.id, entity=entity, attribute="employer",
            value=value, observations=[observation], status=FactStatus.CONFLICTED,
            confidence=ConfidenceTier.LOW, confidence_score=0.2,
            confidence_explanation=EXPLANATION, independent_source_count=1,
            scorer_version="s@1", conflict_group_id=group,
        )
    competing = await facts.conflicting(group)
    assert {f.value for f in competing} == {"Meridian Logistics", "Northwind Freight"}


async def test_human_classification_leaves_evidential_status_alone(
    session, clock, case, user
):
    """A fact can be COMMON *and* DISPUTED — the two axes are independent."""
    investigation, observations = await _two_observations(session, clock, case)
    entities = EntityRepository(session, clock)
    facts = FactRepository(session, clock)
    entity = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.PERSON,
        canonical_name="Jordan Bramwell",
    )
    fact = await facts.assert_fact(
        investigation_id=investigation.id, entity=entity, attribute="employer",
        value="Meridian Logistics", observations=observations, status=FactStatus.COMMON,
        confidence=ConfidenceTier.MEDIUM, confidence_score=0.58,
        confidence_explanation=EXPLANATION, independent_source_count=2,
        scorer_version="s@1",
    )
    await facts.classify(
        fact, FactClassification.DISPUTED, user_id=user.id,
        note="Contradicted by a 2024 filing held off-platform",
    )
    assert fact.classification == FactClassification.DISPUTED
    assert fact.status == FactStatus.COMMON, "human judgement must not rewrite the evidence"


async def test_rescoring_clears_facts_but_never_observations(session, clock, case):
    """Recomputability: stages re-run over stored evidence, no re-crawl."""
    investigation, observations = await _two_observations(session, clock, case)
    entities = EntityRepository(session, clock)
    facts = FactRepository(session, clock)
    observation_repo = ObservationRepository(session, clock)

    entity = await entities.upsert(
        investigation_id=investigation.id, entity_type=EntityType.PERSON,
        canonical_name="Jordan Bramwell",
    )
    await facts.assert_fact(
        investigation_id=investigation.id, entity=entity, attribute="employer",
        value="Meridian Logistics", observations=observations, status=FactStatus.COMMON,
        confidence=ConfidenceTier.MEDIUM, confidence_score=0.58,
        confidence_explanation=EXPLANATION, independent_source_count=2,
        scorer_version="s@1",
    )
    removed = await facts.clear_derived(investigation.id)
    assert removed == 1

    surviving = await observation_repo.get_many([o.id for o in observations])
    assert len(surviving) == 2, "observations must outlive the facts derived from them"


# ------------------------------------------------------------------- graph --


async def test_edge_without_evidence_is_refused_with_a_clear_message(session, clock, case):
    investigation, first, second = case
    graph = GraphRepository(session, clock)
    a = await graph.upsert_node(
        investigation_id=investigation.id, node_type=NodeType.PAGE,
        ref_table="pages", ref_id=first.id, label="Leadership",
    )
    b = await graph.upsert_node(
        investigation_id=investigation.id, node_type=NodeType.PAGE,
        ref_table="pages", ref_id=second.id, label="Award",
    )
    with pytest.raises(ValidationError, match="evidence-backed"):
        await graph.connect(
            investigation_id=investigation.id, from_node=a, to_node=b,
            edge_type=EdgeType.LINKS_TO, derivation="outbound link", observations=[],
        )


async def test_person_node_requires_an_asserting_page(session, clock, case):
    investigation, _first, _second = case
    graph = GraphRepository(session, clock)
    with pytest.raises(ValidationError, match="quoted claim"):
        await graph.upsert_node(
            investigation_id=investigation.id, node_type=NodeType.PERSON,
            ref_table="entities", ref_id=uuid.uuid4(), label="Jordan Bramwell",
        )


async def test_edge_records_its_derivation_and_evidence(session, clock, case):
    investigation, observations = await _two_observations(session, clock, case)
    _inv, first, second = case
    graph = GraphRepository(session, clock)
    a = await graph.upsert_node(
        investigation_id=investigation.id, node_type=NodeType.PAGE,
        ref_table="pages", ref_id=first.id, label="Leadership",
    )
    b = await graph.upsert_node(
        investigation_id=investigation.id, node_type=NodeType.PAGE,
        ref_table="pages", ref_id=second.id, label="Award",
    )
    edge = await graph.connect(
        investigation_id=investigation.id, from_node=a, to_node=b,
        edge_type=EdgeType.LINKS_TO, derivation="page.outbound_links",
        observations=observations,
    )
    assert len(edge.evidence_observation_ids) == 2
    assert edge.derivation == "page.outbound_links"
    assert len(await graph.neighbours(a.id)) == 1


async def test_node_upsert_is_idempotent(session, clock, case):
    investigation, first, _second = case
    graph = GraphRepository(session, clock)
    kwargs = {
        "investigation_id": investigation.id, "node_type": NodeType.PAGE,
        "ref_table": "pages", "ref_id": first.id, "label": "Leadership",
    }
    assert (await graph.upsert_node(**kwargs)).id == (await graph.upsert_node(**kwargs)).id


# ------------------------------------------------------------------ review --


async def test_review_item_requires_a_rationale(session, clock, case):
    investigation, _first, _second = case
    review = ReviewRepository(session, clock)
    with pytest.raises(ValidationError, match="rationale"):
        await review.propose(
            investigation_id=investigation.id, kind=ReviewKind.ENTITY_CANDIDATE,
            subject_type="entity", subject_id=uuid.uuid4(),
            proposal={"name": "Acme"}, rationale={},
        )


async def test_pending_count_gates_case_completion(session, clock, case, user):
    """A case cannot be completed with unreviewed machine output in it."""
    investigation, _first, _second = case
    review = ReviewRepository(session, clock)

    item = await review.propose(
        investigation_id=investigation.id, kind=ReviewKind.ENTITY_CANDIDATE,
        subject_type="entity", subject_id=uuid.uuid4(),
        proposal={"name": "Acme"}, rationale={"observations": [str(uuid.uuid4())]},
    )
    assert await review.pending_count(investigation.id) == 1

    await review.decide(item, ReviewStatus.REJECTED, user_id=user.id, note="Not the subject")
    assert await review.pending_count(investigation.id) == 0
    assert item.status == ReviewStatus.REJECTED


async def test_a_decision_cannot_revert_to_pending(session, clock, case, user):
    investigation, _first, _second = case
    review = ReviewRepository(session, clock)
    item = await review.propose(
        investigation_id=investigation.id, kind=ReviewKind.CONFLICT,
        subject_type="fact", subject_id=uuid.uuid4(),
        proposal={}, rationale={"observations": []},
    )
    with pytest.raises(ValidationError, match="PENDING"):
        await review.decide(item, ReviewStatus.PENDING, user_id=user.id)
