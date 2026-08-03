"""The narrative layer's guards, and the report's independence from it.

Every test here runs without network access. The provider call is the one thing
that is always stubbed; everything else -- payload construction, validation,
persistence, reuse, and PDF rendering -- is exercised for real, because those
are the parts that decide whether a generated sentence reaches a reader.
"""

from __future__ import annotations

import json
from io import BytesIO

import pytest
from PIL import Image
from sqlmodel import Session, SQLModel, create_engine, select

from imatch_api.core.config import Settings
from imatch_api.db.models import ReportNarrative
from imatch_api.services.narrative_service import (
    NarrativeService,
    build_payload,
    payload_digest,
    validate_sections,
)
from imatch_api.services.report_pdf import render_case_report_pdf
from imatch_api.services.report_service import mode_label

# Identifiers that must never cross the boundary to a third-party API.
SUBJECT_NAME = "Priya Raghunathan"
OPERATOR_EMAIL = "dc.hargreaves@example.police.uk"
CASE_REFERENCE = "OP-NIGHTJAR-2291"
PROBE_SHA = "a" * 64
TEMPLATE_SHA = "b" * 64


def _report(**overrides) -> dict:
    report = {
        "notice": "Investigative lead only.",
        "generated_at": "2026-08-03T09:15:00+00:00",
        "generated_by": OPERATOR_EMAIL,
        "case": {
            "id": "case-1",
            "reference": CASE_REFERENCE,
            "title": "Operation Nightjar",
            "description": "Retail robbery series.",
            "status": "open",
            "lawful_basis": "PACE s.64A, prevention and detection of crime",
            "opened_at": "2026-07-01T10:00:00+00:00",
            "closed_at": None,
        },
        "summary": {
            "searches_run": 1,
            "candidates_returned": 2,
            "confirmed_by_examiner": 0,
            "awaiting_adjudication": 2,
            "searches_with_no_result": 0,
        },
        "searches": [{
            "search_id": "run-1",
            "performed_at": "2026-08-02T14:30:00+00:00",
            "mode": "single",
            "mode_label": mode_label("single"),
            "operator": OPERATOR_EMAIL,
            "lawful_basis": "PACE s.64A, prevention and detection of crime",
            "purpose": "Identify a suspect from CCTV.",
            "probe_sha256": PROBE_SHA,
            "probe_path": "",
            "decision": "review",
            "explanation": "Top candidate fell between the review and match thresholds.",
            "top_score": 0.7421,
            "margin": 0.0912,
            "gallery_size": 4820,
            "probe_quality": 0.61,
            "probe_liveness": 0.88,
            "probe_liveness_certified": False,
            "review_required": True,
            "recognition_capable": True,
            "reasons": [],
            "model": {"backend": "insightface", "pack": "buffalo_l"},
            "thresholds": {"match": 0.20, "review": 0.15},
            "audit_hash": "c" * 64,
            "candidates": [
                {
                    "rank": 1,
                    "subject_id": "subj-1",
                    "subject_name": SUBJECT_NAME,
                    "external_ref": "PNC-9931",
                    "template_id": "tpl-1",
                    "template_image_sha256": TEMPLATE_SHA,
                    "template_image_path": "",
                    "similarity": 0.7421,
                    "normalized_score": 0.8110,
                    "adjudication": "pending",
                    "adjudicated_by": None,
                    "adjudicated_at": None,
                    "examiner_notes": "",
                },
                {
                    "rank": 2,
                    "subject_id": "subj-2",
                    "subject_name": "Другой Субъект",
                    "external_ref": "",
                    "template_id": "tpl-2",
                    "template_image_sha256": "d" * 64,
                    "template_image_path": "",
                    "similarity": 0.6509,
                    "normalized_score": 0.7002,
                    "adjudication": "pending",
                    "adjudicated_by": None,
                    "adjudicated_at": None,
                    "examiner_notes": "",
                },
            ],
        }],
        "standard_sections": {
            "methodology": ["Templates are compared by cosine similarity."],
            "limitations": ["Automated face recognition produces investigative leads."],
            "conclusion": ["No candidate has been confirmed by an examiner."],
        },
        "audit_trail": [],
    }
    report.update(overrides)
    return report


# ---------------------------------------------------------------- mode label


def test_single_is_not_reported_as_a_one_to_one_verification():
    """/search records mode="single" for a 1:N gallery search.

    "single" describes the submission, not the comparison. Labelling it as a
    verification would understate the gallery every score was ranked against,
    on every ordinary search in the system.
    """
    assert "1:N" in mode_label("single")
    assert "1:1" not in mode_label("single")


def test_pair_is_the_one_to_one_case():
    assert "1:1" in mode_label("pair")


def test_unknown_mode_is_passed_through_rather_than_guessed():
    assert mode_label("some_future_mode") == "some_future_mode"
    assert mode_label("") == "(not recorded)"


# ------------------------------------------------------------------- payload


def test_payload_carries_no_identifiers():
    """Nothing that names a person, a case, or an operator may leave the system."""
    serialised = json.dumps(build_payload(_report()))

    for identifier in (SUBJECT_NAME, OPERATOR_EMAIL, CASE_REFERENCE, "PNC-9931", "Nightjar"):
        assert identifier not in serialised, f"{identifier!r} reached the outbound payload"

    # The free-text lawful basis can name officers, powers and operations. Only
    # the fact that one is recorded crosses the boundary.
    assert "PACE" not in serialised
    assert json.loads(serialised)["lawful_basis_recorded"] is True

    # A full SHA-256 identifies the exact file. Twelve characters is enough for
    # prose to refer to an exhibit.
    assert PROBE_SHA not in serialised
    assert PROBE_SHA[:12] in serialised


def test_payload_keeps_the_figures_the_narrative_must_cite():
    payload = build_payload(_report())
    search = payload["searches"][0]

    assert search["top_score"] == 0.7421
    assert search["gallery_size"] == 4820
    assert search["thresholds"] == {"match": 0.20, "review": 0.15}
    assert [c["label"] for c in search["candidates"]] == ["Candidate 1", "Candidate 2"]


def test_digest_is_stable_and_findings_sensitive():
    baseline = payload_digest(build_payload(_report()))
    assert baseline == payload_digest(build_payload(_report()))

    changed = _report()
    changed["searches"][0]["candidates"][0]["adjudication"] = "confirmed"
    assert payload_digest(build_payload(changed)) != baseline


# ----------------------------------------------------------------- validator


GOOD = {
    "executive_summary": (
        "One search was run against this case and returned 2 candidates, both of which "
        "remain awaiting adjudication by an examiner. The system recorded a decision of "
        "review for the search."
    ),
    "findings": (
        "The search compared the probe template against a gallery of 4820 templates. The "
        "highest similarity recorded was 0.7421, against a match threshold of 0.2. Two "
        "candidates were returned and neither has yet been adjudicated."
    ),
    "similarity_explanation": (
        "The figures above express likeness between images as measured by the recognition "
        "model. They are not probabilities and cannot be read as a level of certainty. "
        "Ranking is relative to the gallery searched, so the person sought may not be "
        "enrolled in it at all."
    ),
}


def test_validator_accepts_grounded_text():
    assert validate_sections(GOOD, build_payload(_report())) == []


def test_validator_rejects_score_to_percentage_conversion():
    """The single most quotable error available to this layer."""
    draft = dict(GOOD)
    draft["findings"] = "The system reported 74% confidence in the top candidate."

    complaints = validate_sections(draft, build_payload(_report()))
    assert complaints
    assert any("74" in c for c in complaints)


def test_validator_rejects_invented_figures():
    draft = dict(GOOD)
    draft["executive_summary"] = "A gallery of 9999 templates was searched."

    complaints = validate_sections(draft, build_payload(_report()))
    assert any("9999" in c for c in complaints)


def test_validator_rejects_described_appearance():
    """The model receives no imagery, so any such description is fabricated."""
    draft = dict(GOOD)
    draft["similarity_explanation"] = (
        "The nose and jawline show a strong correspondence between the two images."
    )

    complaints = validate_sections(draft, build_payload(_report()))
    assert any("physical appearance" in c for c in complaints)


def test_validator_rejects_identification_assertion():
    draft = dict(GOOD)
    draft["findings"] = "The results show the two images are of the same person."

    complaints = validate_sections(draft, build_payload(_report()))
    assert any("asserts an identification" in c for c in complaints)


def test_validator_rejects_empty_section():
    draft = dict(GOOD)
    draft["findings"] = "   "
    assert validate_sections(draft, build_payload(_report()))


# ------------------------------------------------------------------- service


def _settings(**overrides) -> Settings:
    # GEMINI_API_KEY, not gemini_api_key: the field carries a validation_alias
    # so it reads that exact environment variable rather than the NEXGEN_
    # prefixed one, and initialisation has to use the alias too.
    base = {
        "narrative_enabled": True,
        "GEMINI_API_KEY": "test-key-never-sent",
        "narrative_max_attempts": 2,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_disabled_service_never_calls_the_provider(db_session, monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("provider called while narrative generation is disabled")

    monkeypatch.setattr(NarrativeService, "_call", explode)

    report = _report()
    result = NarrativeService(_settings(narrative_enabled=False)).attach(
        db_session, report, tenant_id="t1", case_id="case-1", generated_by="tester"
    )

    assert result.available is False
    assert "disabled" in result.reason
    assert report["narrative"]["withheld_notice"]


def test_unreachable_provider_does_not_break_the_report(db_session, monkeypatch):
    from imatch_api.services.narrative_service import NarrativeUnavailable

    def unreachable(*_args, **_kwargs):
        raise NarrativeUnavailable("Narrative provider unreachable: connection refused")

    monkeypatch.setattr(NarrativeService, "_call", unreachable)

    report = _report()
    result = NarrativeService(_settings()).attach(
        db_session, report, tenant_id="t1", case_id="case-1", generated_by="tester"
    )

    assert result.available is False
    assert report["standard_sections"]["limitations"], "factual sections must survive"
    assert render_case_report_pdf(report).startswith(b"%PDF")


def test_generated_narrative_is_persisted_then_reused(db_session, monkeypatch):
    """Two exports of one case must read identically. See ReportNarrative."""
    calls = {"count": 0}

    def stub(_self, _prompt):
        calls["count"] += 1
        return json.dumps(GOOD)

    monkeypatch.setattr(NarrativeService, "_call", stub)
    service = NarrativeService(_settings())

    first = service.attach(db_session, _report(), tenant_id="t1", case_id="case-1", generated_by="a")
    db_session.commit()
    second = service.attach(db_session, _report(), tenant_id="t1", case_id="case-1", generated_by="b")

    assert first.available and second.available
    assert calls["count"] == 1, "the provider was called a second time for identical findings"
    assert second.reused is True
    assert first.sections == second.sections

    rows = db_session.exec(select(ReportNarrative)).all()
    assert len(rows) == 1
    assert rows[0].validator_status == "passed"


def test_changed_findings_force_regeneration(db_session, monkeypatch):
    calls = {"count": 0}
    monkeypatch.setattr(
        NarrativeService, "_call", lambda _s, _p: (calls.__setitem__("count", calls["count"] + 1), json.dumps(GOOD))[1]
    )
    service = NarrativeService(_settings())

    service.attach(db_session, _report(), tenant_id="t1", case_id="case-1", generated_by="a")
    db_session.commit()

    adjudicated = _report()
    adjudicated["searches"][0]["candidates"][0]["adjudication"] = "confirmed"
    service.attach(db_session, adjudicated, tenant_id="t1", case_id="case-1", generated_by="a")

    assert calls["count"] == 2


def test_rejected_generation_is_withheld_but_recorded(db_session, monkeypatch):
    bad = dict(GOOD)
    bad["findings"] = "The system reported 74% confidence that this is the same person."
    monkeypatch.setattr(NarrativeService, "_call", lambda _s, _p: json.dumps(bad))

    report = _report()
    result = NarrativeService(_settings()).attach(
        db_session, report, tenant_id="t1", case_id="case-1", generated_by="tester"
    )

    assert result.available is False
    assert result.validator_status == "rejected"
    assert report["narrative"]["sections"] == {}

    # Kept, not discarded: "asked and refused" is different from "never asked".
    row = db_session.exec(select(ReportNarrative)).one()
    assert row.validator_status == "rejected"
    assert row.validator_notes
    assert row.attempts == 2, "a rejected draft should have been retried once"


# ----------------------------------------------------------------------- pdf


def _jpeg(colour=(140, 120, 110), size=240) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (size, size), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_pdf_renders_with_no_narrative_and_no_image_loader():
    pdf = render_case_report_pdf(_report())
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 2000


def test_pdf_renders_images_when_a_loader_is_supplied():
    report = _report()
    report["searches"][0]["probe_path"] = "probes/t1/aa/bb/probe.jpg"
    report["searches"][0]["candidates"][0]["template_image_path"] = "enrol/t1/cc/dd/tpl.jpg"

    store = {
        "probes/t1/aa/bb/probe.jpg": _jpeg((200, 180, 160)),
        "enrol/t1/cc/dd/tpl.jpg": _jpeg((90, 100, 130)),
    }
    without = render_case_report_pdf(report)
    with_images = render_case_report_pdf(report, image_loader=store.get)

    assert with_images.startswith(b"%PDF")
    assert len(with_images) > len(without), "image plates did not reach the document"


def test_pdf_survives_a_purged_image():
    """A retired probe must print as a stated absence, not fail the export."""
    report = _report()
    report["searches"][0]["probe_path"] = "probes/t1/gone.jpg"

    pdf = render_case_report_pdf(report, image_loader=lambda _path: None)
    assert pdf.startswith(b"%PDF")


def test_pdf_survives_a_corrupt_image():
    report = _report()
    report["searches"][0]["probe_path"] = "probes/t1/corrupt.jpg"

    pdf = render_case_report_pdf(report, image_loader=lambda _path: b"not an image at all")
    assert pdf.startswith(b"%PDF")


def test_pdf_prints_candidate_similarity(monkeypatch):
    """Regression: the renderer read a key the report has never contained.

    ``_candidate_section`` emits ``similarity``; the candidate table read
    ``score`` and fell back to "-", so every exported PDF showed a dash where
    the JSON and Markdown exports showed the real figure.
    """
    from reportlab import rl_config

    # Uncompressed page streams so the drawn text is greppable in the bytes.
    monkeypatch.setattr(rl_config, "pageCompression", 0)

    pdf = render_case_report_pdf(_report())
    assert b"0.7421" in pdf, "rank-1 similarity is missing from the rendered PDF"
    assert b"0.6509" in pdf, "rank-2 similarity is missing from the rendered PDF"
