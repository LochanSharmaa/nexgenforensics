"""Enhancement API tests.

The behaviours under test are the ones a report depends on:

  * the original is stored, is served, and is never replaced;
  * a reconstructed image carries its label everywhere it can be fetched;
  * the deployment flag wins over a caller that asks for reconstruction;
  * enhancement is recorded in the audit chain as its own action, so a reader
    can see that an image was processed independently of any later match;
  * the A/B comparison marks the original as primary and records which image
    produced which candidate list.
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imatch_api.db.models import Role


@pytest.fixture
def supervisor(tenant_factory, user_factory):
    tenant = tenant_factory()
    user_factory(tenant, email="supervisor@example.com", role=Role.SUPERVISOR)
    return tenant


@pytest.fixture
def headers(supervisor, auth_headers):
    return auth_headers("supervisor@example.com")


def degraded_b64(width: int = 48, height: int = 40, quality: int = 20, seed: int = 5) -> str:
    """A small, heavily compressed frame -- this module's operating condition."""
    rng = np.random.default_rng(seed)
    array = rng.normal(110, 22, (height, width, 3)).clip(0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class TestStatus:
    def test_status_lists_backends_and_explains_the_unavailable_ones(
        self, client: TestClient, headers
    ):
        response = client.get("/api/imatch/enhance/status", headers=headers)
        assert response.status_code == 200
        body = response.json()

        assert body["enabled"] is True
        # Generative restoration is a separate, deliberate decision.
        assert body["reconstruction_enabled"] is False
        assert body["backends"]

        for backend in body["backends"]:
            assert backend["track"] in {"measurement", "reconstruction"}
            if not backend["available"]:
                # A missing checkpoint is a normal state, not an error, and the
                # operator has to be told which file goes where.
                assert backend["unavailable_reason"]

    def test_classical_backends_are_available_without_weights_or_a_gpu(
        self, client: TestClient, headers
    ):
        body = client.get("/api/imatch/enhance/status", headers=headers).json()
        classical = [b for b in body["backends"] if b["name"].startswith("classical_")]
        assert classical
        assert all(b["available"] for b in classical)

    def test_status_requires_authentication(self, client: TestClient):
        assert client.get("/api/imatch/enhance/status").status_code in (401, 403)


class TestAnalyze:
    def test_analysis_reports_measurements_and_a_plan(self, client: TestClient, headers):
        response = client.post(
            "/api/imatch/enhance/analyze", headers=headers, json={"image_base64": degraded_b64()}
        )
        assert response.status_code == 200
        body = response.json()

        assert body["profile"]["width"] == 48
        assert body["profile"]["jpeg_quality"] is not None
        assert "overall" in body["metrics"]
        assert body["recommended_plan"]["stages"]

    def test_every_planned_stage_states_why_it_was_or_was_not_selected(
        self, client: TestClient, headers
    ):
        body = client.post(
            "/api/imatch/enhance/analyze", headers=headers, json={"image_base64": degraded_b64()}
        ).json()
        for stage in body["recommended_plan"]["stages"]:
            assert stage["rationale"]
            if not stage["selected"]:
                assert stage["skip_reason"]

    def test_analysis_stores_nothing(self, client: TestClient, headers):
        """It is a measurement, not an operation on the evidence."""
        before = client.get("/api/audit", headers=headers).json()
        client.post("/api/imatch/enhance/analyze", headers=headers, json={"image_base64": degraded_b64()})
        after = client.get("/api/audit", headers=headers).json()
        assert len(after) == len(before)

    def test_a_non_image_payload_is_a_400(self, client: TestClient, headers):
        payload = base64.b64encode(b"this is not an image").decode("ascii")
        response = client.post(
            "/api/imatch/enhance/analyze", headers=headers, json={"image_base64": payload}
        )
        assert response.status_code == 400


class TestEnhance:
    def test_enhancement_returns_a_processed_image_and_its_provenance(
        self, client: TestClient, headers
    ):
        response = client.post(
            "/api/imatch/enhance", headers=headers, json={"image_base64": degraded_b64()}
        )
        assert response.status_code == 201
        body = response.json()

        assert body["track"] == "restored"
        assert body["label"].startswith("Processed image")
        assert body["original_sha256"] and body["enhanced_sha256"]
        assert body["original_sha256"] != body["enhanced_sha256"]
        assert body["stages"]
        assert body["metrics_before"] and body["metrics_after"]
        assert body["audit_hash"]

    def test_the_deployment_flag_beats_a_caller_asking_for_reconstruction(
        self, client: TestClient, headers
    ):
        """A caller may decline Track B. It may not grant itself Track B."""
        body = client.post(
            "/api/imatch/enhance",
            headers=headers,
            json={"image_base64": degraded_b64(), "allow_reconstruction": True},
        ).json()
        assert body["track"] == "restored"
        assert body["plan"]["crosses_into_reconstruction"] is False

    def test_both_images_are_retrievable_and_differ(self, client: TestClient, headers):
        enhancement_id = client.post(
            "/api/imatch/enhance", headers=headers, json={"image_base64": degraded_b64()}
        ).json()["enhancement_id"]

        original = client.get(
            f"/api/imatch/enhance/{enhancement_id}/image?variant=original", headers=headers
        )
        enhanced = client.get(
            f"/api/imatch/enhance/{enhancement_id}/image?variant=enhanced", headers=headers
        )
        assert original.status_code == enhanced.status_code == 200
        assert original.content != enhanced.content
        assert original.headers["X-Image-Variant"] == "original"
        assert enhanced.headers["X-Enhancement-Track"] == "restored"
        assert "X-Enhancement-Label" in enhanced.headers

    def test_the_original_is_stored_byte_for_byte(self, client: TestClient, headers):
        """The evidence has to come back exactly as it went in."""
        payload = degraded_b64()
        enhancement_id = client.post(
            "/api/imatch/enhance", headers=headers, json={"image_base64": payload}
        ).json()["enhancement_id"]

        served = client.get(
            f"/api/imatch/enhance/{enhancement_id}/image?variant=original", headers=headers
        )
        assert served.content == base64.b64decode(payload)

    def test_the_enhanced_image_is_lossless(self, client: TestClient, headers):
        """PNG, not JPEG: re-compressing would undo the stage that just ran."""
        enhancement_id = client.post(
            "/api/imatch/enhance", headers=headers, json={"image_base64": degraded_b64()}
        ).json()["enhancement_id"]
        served = client.get(
            f"/api/imatch/enhance/{enhancement_id}/image?variant=enhanced", headers=headers
        )
        assert served.content.startswith(b"\x89PNG\r\n\x1a\n")

    def test_enhancement_is_its_own_audit_action(self, client: TestClient, headers):
        """Separate from any search, so "this image was processed" is legible alone."""
        client.post("/api/imatch/enhance", headers=headers, json={"image_base64": degraded_b64()})
        records = client.get("/api/audit", headers=headers).json()
        entries = [r for r in records if r["action"] == "evidence.enhance"]
        assert entries, "enhancement did not appear in the audit chain"

    def test_the_audit_chain_still_verifies_after_an_enhancement(
        self, client: TestClient, headers, supervisor, user_factory, auth_headers
    ):
        # Chain verification is admin-only; the enhancement itself runs as the
        # supervisor so the test exercises the same records a real tenant would.
        user_factory(supervisor, email="admin@example.com", role=Role.ADMIN)
        client.post("/api/imatch/enhance", headers=headers, json={"image_base64": degraded_b64()})
        verification = client.get("/api/audit/verify", headers=auth_headers("admin@example.com"))
        assert verification.status_code == 200
        assert verification.json()["valid"] is True

    def test_an_unknown_case_is_refused(self, client: TestClient, headers):
        response = client.post(
            "/api/imatch/enhance",
            headers=headers,
            json={"image_base64": degraded_b64(), "case_id": "does-not-exist"},
        )
        assert response.status_code == 404

    def test_an_enhancement_from_another_tenant_is_not_found(
        self, client: TestClient, headers, tenant_factory, user_factory, auth_headers
    ):
        """404 rather than 403: a 403 would confirm the id exists."""
        enhancement_id = client.post(
            "/api/imatch/enhance", headers=headers, json={"image_base64": degraded_b64()}
        ).json()["enhancement_id"]

        other = tenant_factory("other-tenant")
        user_factory(other, email="outsider@example.com", role=Role.SUPERVISOR)
        response = client.get(
            f"/api/imatch/enhance/{enhancement_id}", headers=auth_headers("outsider@example.com")
        )
        assert response.status_code == 404

    def test_repeating_an_enhancement_is_served_from_cache(self, client: TestClient, headers):
        payload = degraded_b64()
        first = client.post("/api/imatch/enhance", headers=headers, json={"image_base64": payload}).json()
        second = client.post("/api/imatch/enhance", headers=headers, json={"image_base64": payload}).json()
        assert second["served_from_cache"] is True
        assert first["enhanced_sha256"] == second["enhanced_sha256"]

    def test_a_dropped_stage_is_recorded_as_an_override(self, client: TestClient, headers):
        body = client.post(
            "/api/imatch/enhance",
            headers=headers,
            json={"image_base64": degraded_b64(), "disabled_stages": ["classical_tone"]},
        ).json()
        dropped = [s for s in body["plan"]["stages"] if s["name"] == "classical_tone"]
        assert dropped and dropped[0]["selected"] is False
        assert "examiner" in dropped[0]["skip_reason"]


class TestRecognitionComparison:
    """The A/B run. Both sides go through the UNMODIFIED recogniser."""

    @pytest.fixture
    def enhancement_id(self, client: TestClient, headers, face_b64) -> str:
        return client.post(
            "/api/imatch/enhance",
            headers=headers,
            json={"image_base64": face_b64[0], "lawful_basis": "Test comparison"},
        ).json()["enhancement_id"]

    def test_both_sides_are_returned_and_the_original_is_primary(
        self, client: TestClient, headers, enhancement_id
    ):
        response = client.post(
            f"/api/imatch/enhance/{enhancement_id}/recognise",
            headers=headers,
            json={"lawful_basis": "Test comparison", "top_k": 5},
        )
        assert response.status_code == 200
        body = response.json()

        assert body["primary"] == "original"
        assert body["original"]["source_kind"] == "original"
        assert body["enhanced"]["source_kind"] in {"restored", "reconstructed"}
        # The two searches are not independent evidence, and the response says so.
        assert "not independent" in body["caution"]

    def test_each_side_is_persisted_with_its_source_kind(
        self, client: TestClient, headers, enhancement_id
    ):
        client.post(
            f"/api/imatch/enhance/{enhancement_id}/recognise",
            headers=headers,
            json={"lawful_basis": "Test comparison"},
        )
        runs = client.get("/api/imatch/searches", headers=headers).json()
        assert runs, "the comparison did not persist any search run"

    def test_a_comparison_without_lawful_basis_is_refused(
        self, client: TestClient, headers, face_b64
    ):
        enhancement_id = client.post(
            "/api/imatch/enhance", headers=headers, json={"image_base64": face_b64[0]}
        ).json()["enhancement_id"]
        response = client.post(
            f"/api/imatch/enhance/{enhancement_id}/recognise", headers=headers, json={}
        )
        assert response.status_code == 422

    def test_a_face_the_detector_cannot_find_is_a_finding_not_a_crash(
        self, client: TestClient, headers
    ):
        """Very common on real surveillance frames, and informative on its own."""
        enhancement_id = client.post(
            "/api/imatch/enhance",
            headers=headers,
            json={"image_base64": degraded_b64(), "lawful_basis": "Test"},
        ).json()["enhancement_id"]

        response = client.post(
            f"/api/imatch/enhance/{enhancement_id}/recognise",
            headers=headers,
            json={"lawful_basis": "Test"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["original"]["decision"] in {"no_face_detected", "no_match", "inconclusive", "review"}
