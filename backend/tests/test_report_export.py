"""Case report export, end to end through the HTTP API against the real engine.

The narrative unit tests in test_narrative_report.py work from a synthetic
report dict. Nothing there proves that ReportService.build() produces that shape
from real database rows, that the storage paths it emits resolve to real files,
or that the PDF renderer can embed them. This does, by enrolling a subject,
searching for them, adjudicating the result and exporting the case.

The narrative layer stays disabled throughout. These tests assert the property
that matters most about it: with no provider configured, every factual section
still renders and the export still succeeds.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from imatch_api.db.models import Role


@pytest.fixture
def supervisor(tenant_factory, user_factory):
    tenant = tenant_factory()
    user_factory(tenant, email="supervisor@example.com", role=Role.SUPERVISOR)
    return tenant


@pytest.fixture
def headers(supervisor, auth_headers):
    return auth_headers("supervisor@example.com")


@pytest.fixture
def adjudicated_case(client: TestClient, headers, face_paths) -> str:
    """A case with one enrolled subject, one search against it, one adjudication."""
    name, paths = next(iter(face_paths.items()))

    case = client.post(
        "/api/cases",
        headers=headers,
        json={
            "reference": "OP-EXPORT-1",
            "title": "Report export coverage",
            "lawful_basis": "Warrant 2026/114",
        },
    )
    assert case.status_code == 201, case.text
    case_id = case.json()["id"]

    enrol = client.post(
        "/api/subjects",
        headers=headers,
        json={
            "display_name": name,
            "external_ref": name,
            "image_base64": base64.b64encode(paths[0].read_bytes()).decode(),
            "lawful_basis": "Warrant 2026/114",
        },
    )
    assert enrol.status_code == 201, enrol.text

    search = client.post(
        "/api/imatch/search",
        headers=headers,
        json={
            "image_base64": base64.b64encode(paths[1].read_bytes()).decode(),
            "lawful_basis": "Warrant 2026/114",
            "case_id": case_id,
            "mode": "single",
            "top_k": 5,
        },
    )
    assert search.status_code == 200, search.text
    candidates = search.json()["candidates"]
    assert candidates, "no candidate returned; cannot exercise the report"

    adjudicate = client.post(
        f"/api/imatch/candidates/{candidates[0]['id']}/adjudicate",
        headers=headers,
        json={"adjudication": "confirmed", "examiner_notes": "Verified side by side."},
    )
    assert adjudicate.status_code == 200, adjudicate.text
    return case_id


class TestJsonExport:
    def test_report_carries_the_deterministic_sections(
        self, client: TestClient, headers, adjudicated_case
    ):
        report = client.get(
            f"/api/cases/{adjudicated_case}/report", headers=headers, params={"fmt": "json"}
        )
        assert report.status_code == 200, report.text
        body = report.json()

        sections = body["standard_sections"]
        assert sections["methodology"], "methodology is empty"
        assert sections["limitations"], "limitations are empty"
        assert sections["conclusion"], "conclusion is empty"

        # The fixed caveats must survive into every export unconditionally.
        joined = " ".join(sections["limitations"])
        assert "investigative leads, not identifications" in joined
        assert "not the probability" in joined

        # An examiner confirmed a candidate, so the conclusion must say so
        # rather than reporting the score as the finding.
        assert any("confirmed" in line for line in sections["conclusion"])

    def test_report_points_at_the_images_it_was_computed_from(
        self, client: TestClient, headers, adjudicated_case
    ):
        body = client.get(
            f"/api/cases/{adjudicated_case}/report", headers=headers, params={"fmt": "json"}
        ).json()

        search = body["searches"][0]
        assert search["probe_sha256"], "probe hash missing"
        assert search["probe_path"], "probe path missing; the PDF cannot show the image"

        candidate = search["candidates"][0]
        assert candidate["template_image_sha256"], "enrolment image hash missing"
        assert candidate["template_image_path"], "enrolment image path missing"

    def test_search_mode_is_labelled_as_one_to_many(
        self, client: TestClient, headers, adjudicated_case
    ):
        """mode="single" describes the submission, not a 1:1 comparison."""
        body = client.get(
            f"/api/cases/{adjudicated_case}/report", headers=headers, params={"fmt": "json"}
        ).json()

        search = body["searches"][0]
        assert search["mode"] == "single"
        assert "1:N" in search["mode_label"]
        assert "1:1" not in search["mode_label"]

    def test_narrative_absence_is_stated_rather_than_silent(
        self, client: TestClient, headers, adjudicated_case
    ):
        body = client.get(
            f"/api/cases/{adjudicated_case}/report", headers=headers, params={"fmt": "json"}
        ).json()

        narrative = body["narrative"]
        assert narrative["available"] is False
        assert narrative["withheld_notice"], "a reader must be told why the prose is missing"
        assert "disabled" in narrative["reason"]

    def test_export_is_audited(self, client: TestClient, headers, adjudicated_case):
        client.get(f"/api/cases/{adjudicated_case}/report", headers=headers, params={"fmt": "json"})
        body = client.get(
            f"/api/cases/{adjudicated_case}/report", headers=headers, params={"fmt": "json"}
        ).json()

        actions = [entry["action"] for entry in body["audit_trail"]]
        assert "case.export" in actions
        # Nothing left the system, so nothing may claim it did.
        assert "case.narrative_generate" not in actions


class TestMarkdownExport:
    def test_markdown_contains_every_deterministic_section(
        self, client: TestClient, headers, adjudicated_case
    ):
        response = client.get(
            f"/api/cases/{adjudicated_case}/report", headers=headers, params={"fmt": "markdown"}
        )
        assert response.status_code == 200
        text = response.text

        assert "## Methodology" in text
        assert "## Limitations" in text
        assert "## Conclusion" in text
        assert "## Audit trail" in text
        # Not generated, so it must not appear.
        assert "## Executive summary" not in text


class TestPdfExport:
    def test_pdf_embeds_the_probe_and_candidate_images(
        self, client: TestClient, headers, adjudicated_case
    ):
        response = client.get(
            f"/api/cases/{adjudicated_case}/report", headers=headers, params={"fmt": "pdf"}
        )
        assert response.status_code == 200
        pdf = response.content

        assert pdf.startswith(b"%PDF")
        assert response.headers["content-type"] == "application/pdf"
        # An embedded raster is the proof that the storage path resolved and the
        # bytes were decoded. Without it the plates fell back to "image not
        # available" and the export would still have returned 200.
        assert b"/Image" in pdf, "no image XObject; the plates did not render"
        assert len(pdf) > 20_000, "PDF is too small to contain two embedded images"

    def test_pdf_prints_the_real_similarity_score(
        self, client: TestClient, headers, adjudicated_case, monkeypatch
    ):
        """Regression: the candidate table read a key the report never had."""
        from reportlab import rl_config

        monkeypatch.setattr(rl_config, "pageCompression", 0)

        body = client.get(
            f"/api/cases/{adjudicated_case}/report", headers=headers, params={"fmt": "json"}
        ).json()
        similarity = body["searches"][0]["candidates"][0]["similarity"]

        pdf = client.get(
            f"/api/cases/{adjudicated_case}/report", headers=headers, params={"fmt": "pdf"}
        ).content

        assert f"{similarity:.4f}".encode() in pdf, (
            f"similarity {similarity:.4f} is missing from the rendered PDF"
        )

    def test_pdf_renders_for_a_case_with_no_searches(self, client: TestClient, headers):
        case = client.post(
            "/api/cases",
            headers=headers,
            json={"reference": "OP-EMPTY-1", "title": "Nothing run yet",
                  "lawful_basis": "Warrant 2026/114"},
        )
        assert case.status_code == 201

        response = client.get(
            f"/api/cases/{case.json()['id']}/report", headers=headers, params={"fmt": "pdf"}
        )
        assert response.status_code == 200
        assert response.content.startswith(b"%PDF")
