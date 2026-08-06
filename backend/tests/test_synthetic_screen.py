"""API-level tests for the synthetic-media screen on the comparison endpoints.

The complaint that motivated the screen: a deepfaked photograph could be
submitted to 1:1 verify or reference-vs-set batch and the system would compare
it and present the similarity as if both inputs were genuine photographs.
These tests pin the contract that every comparison response now carries the
screen's verdict, and that a generator-tagged image cannot pass silently.

Needs the real recognition models and AgeDB (the screened image must still
contain a detectable face); skips cleanly without them, like every other
recognition test.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

from imatch_api.db.models import Role

BANDS = {"minimal", "moderate", "elevated", "high"}


@pytest.fixture
def supervisor(tenant_factory, user_factory):
    tenant = tenant_factory()
    user_factory(tenant, email="supervisor@example.com", role=Role.SUPERVISOR)
    return tenant


@pytest.fixture
def headers(supervisor, auth_headers):
    return auth_headers("supervisor@example.com")


def tagged_deepfake_b64(image_path) -> str:
    """A real face photograph re-saved as a PNG carrying a generation record.

    The pixels stay a genuine face so detection still succeeds; the metadata
    is what a Stable Diffusion export actually writes, which the provenance
    signal treats as decisive.
    """
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "portrait, Steps: 30, Sampler: DPM++ 2M, Seed: 4242")
    buffer = BytesIO()
    Image.open(image_path).convert("RGB").save(buffer, "PNG", pnginfo=info)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class TestVerifyScreening:
    def test_genuine_pair_reports_a_clean_screen(self, client: TestClient, headers, face_paths):
        paths = next(iter(face_paths.values()))
        response = client.post(
            "/api/imatch/verify",
            headers=headers,
            json={
                "reference_image_base64": base64.b64encode(paths[0].read_bytes()).decode(),
                "probe_image_base64": base64.b64encode(paths[1].read_bytes()).decode(),
                "lawful_basis": "Warrant 2026/114",
            },
        )
        assert response.status_code == 200
        body = response.json()

        screen = body["synthetic_screen"]
        assert screen["flagged"] is False
        assert screen["probe"]["band"] in BANDS
        assert screen["reference"]["band"] in BANDS
        # The full per-signal breakdown rides on each assessment.
        assert body["probe"]["deepfake"]["signals"]
        assert body["reference"]["deepfake"]["method"].startswith("multi_signal")

    def test_generator_tagged_probe_cannot_pass_silently(
        self, client: TestClient, headers, face_paths
    ):
        paths = next(iter(face_paths.values()))
        response = client.post(
            "/api/imatch/verify",
            headers=headers,
            json={
                "reference_image_base64": base64.b64encode(paths[0].read_bytes()).decode(),
                "probe_image_base64": tagged_deepfake_b64(paths[1]),
                "lawful_basis": "Warrant 2026/114",
            },
        )
        assert response.status_code == 200
        body = response.json()

        screen = body["synthetic_screen"]
        assert screen["flagged"] is True
        assert screen["probe"]["flagged"] is True
        assert screen["reference"]["flagged"] is False
        assert body["review_required"] is True
        assert "generative_metadata_present" in screen["probe"]["reasons"]
        # The caution is spelled out in the human-readable explanation too.
        assert "synthetic-media screen flagged" in body["explanation"]

    def test_flagged_comparison_is_recorded_in_the_audit_trail(
        self, client: TestClient, headers, face_paths
    ):
        paths = next(iter(face_paths.values()))
        client.post(
            "/api/imatch/verify",
            headers=headers,
            json={
                "reference_image_base64": base64.b64encode(paths[0].read_bytes()).decode(),
                "probe_image_base64": tagged_deepfake_b64(paths[1]),
                "lawful_basis": "Warrant 2026/114",
            },
        )
        entries = client.get("/api/audit", headers=headers).json()
        verify_entries = [e for e in entries if e["action"] == "biometric.verify"]
        assert verify_entries, "verify action missing from the audit trail"
        detail = json.loads(verify_entries[0]["detail"])
        assert detail["synthetic_media_flagged"] is True
        assert detail["probe_deepfake_band"] == "high"


class TestBatchScreening:
    def test_one_to_many_screens_the_shared_reference(
        self, client: TestClient, headers, face_paths
    ):
        """The exact reported drawback: a deepfaked reference in reference-vs-set
        ran the whole batch without a word. The response must now carry the
        reference's screening verdict."""
        paths = next(iter(face_paths.values()))
        response = client.post(
            "/api/imatch/batch",
            headers=headers,
            json={
                "mode": "one_to_many",
                "reference_image_base64": tagged_deepfake_b64(paths[0]),
                "items": [
                    {"label": "still-1", "probe_image_base64": base64.b64encode(paths[1].read_bytes()).decode()},
                ],
                "lawful_basis": "Warrant 2026/114",
            },
        )
        assert response.status_code == 200
        body = response.json()

        assert body["reference"] is not None
        assert body["reference"]["deepfake"]["flagged"] is True
        assert body["reference"]["deepfake"]["band"] == "high"

        row = body["results"][0]
        assert row["status"] == "ok"
        assert row["probe_deepfake_band"] in BANDS

    def test_flagged_probe_is_marked_on_its_own_row(
        self, client: TestClient, headers, face_paths
    ):
        paths = next(iter(face_paths.values()))
        response = client.post(
            "/api/imatch/batch",
            headers=headers,
            json={
                "mode": "one_to_many",
                "reference_image_base64": base64.b64encode(paths[0].read_bytes()).decode(),
                "items": [
                    {"label": "genuine", "probe_image_base64": base64.b64encode(paths[1].read_bytes()).decode()},
                    {"label": "tagged", "probe_image_base64": tagged_deepfake_b64(paths[1])},
                ],
                "lawful_basis": "Warrant 2026/114",
            },
        )
        assert response.status_code == 200
        rows = {row["label"]: row for row in response.json()["results"]}

        assert rows["tagged"]["probe_deepfake_band"] == "high"
        assert "synthetic_media_risk" in rows["tagged"]["probe_flags"]
        assert rows["genuine"]["probe_deepfake_band"] not in {"high"}
        assert "synthetic_media_risk" not in rows["genuine"]["probe_flags"]
