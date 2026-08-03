"""Vision analysis: the observation stage, and the guardrail on it.

The prohibition — never identify a person from appearance — cannot rest on
prompt wording, so most of this file tests the *control* rather than the
request: output that asserts identity must be dropped no matter how the model
phrases it.
"""

from __future__ import annotations

import io
import random

import pytest
from PIL import Image as PILImage

from shared.config import Settings
from shared.enums import ObservationMethod, VisionCategory
from vision.base import SearchClue, VisionAnalysis, VisionObservation
from vision.gemini.provider import RESPONSE_SCHEMA, SYSTEM_PROMPT, GeminiVisionProvider
from vision.guardrails import clue_is_safe, filter_observations, inspect
from vision.registry import provider_status

CASE = {"case_id": "VIS-1", "title": "Vision", "lawful_basis": "Engagement"}


def _png(seed: int = 11) -> bytes:
    rng = random.Random(seed)  # noqa: S311 - test fixture, not cryptography
    image = PILImage.new("RGB", (64, 64))
    for bx in range(8):
        for by in range(8):
            colour = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for x in range(bx * 8, bx * 8 + 8):
                for y in range(by * 8, by * 8 + 8):
                    image.putpixel((x, y), colour)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _obs(value: str, category=VisionCategory.VISUAL_CLUE, detail: str = ""):
    return VisionObservation(category=category, value=value, detail=detail)


# ------------------------------------------------- the prohibition, layer 1 --


def test_the_response_schema_has_no_place_for_an_identity():
    """The outermost defence: a well-behaved model cannot even express one."""
    text = str(RESPONSE_SCHEMA).lower()
    assert "identity" not in text
    assert "who" not in text
    # People may be counted, never named.
    assert RESPONSE_SCHEMA["properties"]["people_present"]["type"] == "integer"


def test_the_prompt_states_the_boundary():
    assert "Identify any person" in SYSTEM_PROMPT
    assert "only count them" in SYSTEM_PROMPT


# ------------------------------------------------- the prohibition, layer 2 --


@pytest.mark.parametrize(
    "value",
    [
        "This person is Jordan Bramwell",
        "The man appears to be a senior executive",
        "The woman resembles the subject of the enquiry",
        "Individual identified as J. Bramwell",
        "Subject believed to be the account holder",
    ],
)
def test_identity_assertions_are_dropped(value):
    verdict = inspect(_obs(value))
    assert verdict is not None
    assert verdict.rule == "identity_assertion"


@pytest.mark.parametrize(
    "value",
    [
        "Based on his facial features, likely the same man",
        "Judging by their appearance, a company employee",
    ],
)
def test_appearance_inference_is_dropped(value):
    assert inspect(_obs(value)) is not None


@pytest.mark.parametrize(
    "value",
    [
        "Facial recognition suggests a match",
        "Biometric comparison with the second figure",
        "Face match against the other photograph",
    ],
)
def test_any_facial_analysis_claim_is_dropped(value):
    verdict = inspect(_obs(value))
    assert verdict is not None
    assert verdict.rule == "facial_analysis"


@pytest.mark.parametrize(
    "value",
    [
        "The man appears to be around 40 years old",
        "Woman appears to be in her thirties",
        "ethnicity: South Asian",
    ],
)
def test_demographic_profiling_is_dropped(value):
    assert inspect(_obs(value)) is not None


def test_a_name_attached_to_a_person_is_dropped():
    """A name read off a sign is evidence. A name attached to a face is an
    identification."""
    verdict = inspect(
        _obs("Man in blue jacket, Jordan Bramwell", category=VisionCategory.VISUAL_CLUE)
    )
    assert verdict is not None
    assert verdict.rule == "name_attached_to_person"


# ------------------------------------------------ what must NOT be dropped --


@pytest.mark.parametrize(
    ("value", "category"),
    [
        ("MERIDIAN LOGISTICS", VisionCategory.SIGN),
        ("Jordan Bramwell", VisionCategory.TEXT),           # a nameplate
        ("Director of Operations — J. Bramwell", VisionCategory.DOCUMENT),
        ("Two people are visible near the entrance", VisionCategory.VISUAL_CLUE),
        ("Red delivery van, registration DL 3C AB 1234", VisionCategory.VEHICLE),
        ("Devanagari script on the shopfront", VisionCategory.LOCATION_CUE),
    ],
)
def test_legitimate_observations_survive(value, category):
    """Over-blocking would gut the feature. A name printed on a nameplate is
    *text visible in the image* — exactly what this stage exists to capture."""
    assert inspect(_obs(value, category=category)) is None


def test_filtering_splits_kept_from_refused():
    kept, rejected = filter_observations(
        [
            _obs("MERIDIAN LOGISTICS", VisionCategory.SIGN),
            _obs("This person is Jordan Bramwell"),
            _obs("Blue awning over the entrance", VisionCategory.OBJECT),
        ]
    )
    assert [o.value for o in kept] == ["MERIDIAN LOGISTICS", "Blue awning over the entrance"]
    assert len(rejected) == 1


def test_empty_observations_are_dropped():
    assert inspect(_obs("   ")) is not None


# ---------------------------------------------------------- search clues ----


@pytest.mark.parametrize(
    "query",
    [
        "who is the man in the red jacket",
        "identify the person in this photo",
        "name of the woman standing on the left",
    ],
)
def test_identification_queries_never_leave_the_system(query):
    """A clue is where an observation becomes a search. It is the last place to
    stop an identification attempt."""
    assert clue_is_safe(query) is False


@pytest.mark.parametrize(
    "query",
    [
        "Meridian Logistics head office",
        "DL 3C AB 1234 vehicle registration",
        "Meridian Logistics annual report 2024",
    ],
)
def test_legitimate_queries_are_allowed(query):
    assert clue_is_safe(query) is True


# ------------------------------------------------------ provider behaviour --


def test_provider_reports_unconfigured_rather_than_failing():
    status = provider_status(Settings())
    assert status["configured"] is False
    assert "IIE_GEMINI_API_KEY" in status["config_keys"]


async def test_analysis_without_a_key_is_unavailable_not_empty():
    """"No key" and "saw nothing" must never look the same."""
    result = await GeminiVisionProvider(api_key="").analyse(_png(), "image/png")
    assert result.available is False
    assert "IIE_GEMINI_API_KEY" in result.error
    assert result.observations == ()


def test_parsing_applies_the_guardrails():
    """The provider's own parse step must filter, not just the service."""
    provider = GeminiVisionProvider(api_key="x")
    analysis = provider._parse(
        {
            "observations": [
                {"category": "SIGN", "value": "MERIDIAN LOGISTICS", "confidence": 0.9},
                {"category": "VISUAL_CLUE", "value": "This person is Jordan Bramwell"},
            ],
            "search_clues": [
                {"query": "Meridian Logistics", "rationale": "read from the sign"},
                {"query": "who is the man in the photo", "rationale": "identify him"},
            ],
            "people_present": 2,
        },
        started=0.0,
    )

    assert [o.value for o in analysis.observations] == ["MERIDIAN LOGISTICS"]
    assert [c.query for c in analysis.clues] == ["Meridian Logistics"]
    assert len(analysis.rejected) == 2
    assert {r["rule"] for r in analysis.rejected} == {"identity_assertion", "unsafe_clue"}
    # People are counted; nothing more.
    assert analysis.people_present == 2


def test_unknown_categories_are_kept_not_discarded():
    """A vocabulary mismatch must not silently lose evidence."""
    provider = GeminiVisionProvider(api_key="x")
    analysis = provider._parse(
        {"observations": [{"category": "SOMETHING_NEW", "value": "a brass plaque"}]},
        started=0.0,
    )
    assert len(analysis.observations) == 1
    assert analysis.observations[0].category == VisionCategory.VISUAL_CLUE


# ------------------------------------------------------------- the service --


class _StubVision:
    """A model that behaves, and one that misbehaves, on demand."""

    name = "stub"
    model = "stub-1"

    def __init__(self, analysis: VisionAnalysis):
        self._analysis = analysis

    def available(self) -> bool:
        return True

    async def analyse(self, image: bytes, mime_type: str) -> VisionAnalysis:
        return self._analysis


def _analysis(**kwargs) -> VisionAnalysis:
    return VisionAnalysis(
        provider="stub", model="stub-1", available=True, **kwargs
    )


async def test_observations_are_stored_against_the_image(session, clock, user):
    from database.models import Image, Investigation
    from database.repositories import ObservationRepository
    from shared.clock import SystemClock
    from vision.service import VisionService

    investigation = Investigation(
        owner_id=user.id, case_id="VIS-STORE", title="V", lawful_basis="t", status="NEW"
    )
    session.add(investigation)
    await session.flush()
    image = Image(
        investigation_id=investigation.id, role="PROBE", sha256="a" * 64,
        phash="b" * 16, storage_key="k", created_at=SystemClock().now(),
    )
    session.add(image)
    await session.flush()

    service = VisionService(
        session,
        Settings(environment="test", database_url="sqlite+aiosqlite:///:memory:",
                 secret_key="k" * 40),
        _StubVision(
            _analysis(
                observations=(
                    VisionObservation(
                        category=VisionCategory.SIGN, value="MERIDIAN LOGISTICS",
                        detail="above the door", confidence=0.92,
                    ),
                ),
                clues=(
                    SearchClue(
                        query="Meridian Logistics", rationale="read from the sign",
                        source_category=VisionCategory.SIGN, priority=2,
                    ),
                ),
                people_present=1,
            )
        ),
        clock,
    )

    result = await service.analyse(
        investigation_id=investigation.id, image=image, image_bytes=_png()
    )

    stored = await ObservationRepository(session, clock).by_method(
        investigation.id, ObservationMethod.VISION
    )
    assert len(stored) == 1
    record = stored[0]
    assert record.raw_value == "MERIDIAN LOGISTICS"
    # Sourced to the image, so a reviewer can check it by looking.
    assert record.image_id == image.id
    assert record.page_id is None
    assert record.method == ObservationMethod.VISION
    assert "[SIGN]" in record.context_snippet
    assert "stub-1" in record.extractor_version
    assert result.analysis.people_present == 1


async def test_vision_writes_observations_never_facts(session, clock, user):
    """The separation the whole design rests on: the model observes, it does
    not conclude. A fact needs corroborating sources."""
    from sqlalchemy import func, select

    from database.models import Fact, Image, Investigation
    from shared.clock import SystemClock
    from vision.service import VisionService

    investigation = Investigation(
        owner_id=user.id, case_id="VIS-NOFACT", title="V", lawful_basis="t", status="NEW"
    )
    session.add(investigation)
    await session.flush()
    image = Image(
        investigation_id=investigation.id, role="PROBE", sha256="c" * 64,
        phash="d" * 16, storage_key="k", created_at=SystemClock().now(),
    )
    session.add(image)
    await session.flush()

    service = VisionService(
        session,
        Settings(environment="test", database_url="sqlite+aiosqlite:///:memory:",
                 secret_key="k" * 40),
        _StubVision(
            _analysis(
                observations=(
                    VisionObservation(category=VisionCategory.SIGN, value="MERIDIAN"),
                )
            )
        ),
        clock,
    )
    await service.analyse(
        investigation_id=investigation.id, image=image, image_bytes=_png()
    )

    facts = (await session.execute(select(func.count()).select_from(Fact))).scalar_one()
    assert facts == 0, "vision must never write a fact"


async def test_clues_are_derived_from_transcriptions_too(session, clock, user):
    """Searchable text the model transcribed but did not think to propose is
    still a lead."""
    from database.models import Image, Investigation
    from shared.clock import SystemClock
    from vision.service import VisionService

    investigation = Investigation(
        owner_id=user.id, case_id="VIS-CLUES", title="V", lawful_basis="t", status="NEW"
    )
    session.add(investigation)
    await session.flush()
    image = Image(
        investigation_id=investigation.id, role="PROBE", sha256="e" * 64,
        phash="f" * 16, storage_key="k", created_at=SystemClock().now(),
    )
    session.add(image)
    await session.flush()

    service = VisionService(
        session,
        Settings(environment="test", database_url="sqlite+aiosqlite:///:memory:",
                 secret_key="k" * 40),
        _StubVision(
            _analysis(
                observations=(
                    VisionObservation(category=VisionCategory.SIGN, value="MERIDIAN LOGISTICS"),
                    VisionObservation(category=VisionCategory.OBJECT, value="blue van"),
                )
            )
        ),
        clock,
    )
    result = await service.analyse(
        investigation_id=investigation.id, image=image, image_bytes=_png()
    )

    queries = [c["query"] for c in result.clues]
    assert "MERIDIAN LOGISTICS" in queries
    # "blue van" is an OBJECT — not searchable, and running it would spend a
    # paid provider call on noise.
    assert "blue van" not in queries


# ------------------------------------------------------------------- API ----


async def _case_with_image(auth_client) -> str:
    case_id = (await auth_client.post("/api/v1/investigations", json=CASE)).json()["id"]
    await auth_client.post(
        f"/api/v1/investigations/{case_id}/images",
        files={"file": ("probe.png", _png(), "image/png")},
    )
    return case_id


async def test_analyze_without_a_key_reports_unavailable(auth_client):
    case_id = await _case_with_image(auth_client)
    response = await auth_client.post(f"/api/v1/investigations/{case_id}/analyze")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["available"] is False
    assert "IIE_GEMINI_API_KEY" in body["error"]
    assert body["observation_count"] == 0


async def test_analyze_requires_a_probe_image(auth_client):
    case_id = (await auth_client.post("/api/v1/investigations", json=CASE)).json()["id"]
    response = await auth_client.post(f"/api/v1/investigations/{case_id}/analyze")
    assert response.status_code == 422
    assert "Upload a probe image" in response.json()["detail"]


async def test_analysis_is_audited_even_when_unavailable(auth_client):
    case_id = await _case_with_image(auth_client)
    await auth_client.post(f"/api/v1/investigations/{case_id}/analyze")
    entries = (await auth_client.get(f"/api/v1/audit?investigation_id={case_id}")).json()
    runs = [e for e in entries if e["action"] == "vision.analyze"]
    assert runs
    assert runs[-1]["outcome"] == "unavailable"


async def test_stored_analysis_reads_back_from_the_evidence_table(auth_client):
    """The evidence table is the source of truth, not a cached response."""
    case_id = await _case_with_image(auth_client)
    response = await auth_client.get(f"/api/v1/investigations/{case_id}/analysis")
    assert response.status_code == 200
    assert response.json()["observation_count"] == 0
