from __future__ import annotations

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


class TestCases:
    def test_create_and_read_a_case(self, client: TestClient, headers):
        created = client.post(
            "/api/cases",
            headers=headers,
            json={
                "reference": "OP-2026-0001",
                "title": "Retail theft series",
                "lawful_basis": "Investigation under warrant 2026/114",
            },
        )
        assert created.status_code == 201
        case_id = created.json()["id"]

        detail = client.get(f"/api/cases/{case_id}", headers=headers)
        assert detail.status_code == 200
        assert detail.json()["reference"] == "OP-2026-0001"
        assert detail.json()["search_count"] == 0

    def test_duplicate_reference_is_rejected(self, client: TestClient, headers):
        payload = {"reference": "OP-DUP", "title": "First"}
        assert client.post("/api/cases", headers=headers, json=payload).status_code == 201
        assert client.post("/api/cases", headers=headers, json=payload).status_code == 409

    def test_case_from_another_tenant_is_not_found(
        self, client: TestClient, headers, tenant_factory, user_factory, auth_headers
    ):
        """A 403 here would confirm the id exists; 404 reveals nothing."""
        created = client.post("/api/cases", headers=headers, json={"reference": "OP-X", "title": "Secret"})
        case_id = created.json()["id"]

        other_tenant = tenant_factory("other-tenant")
        user_factory(other_tenant, email="outsider@example.com", role=Role.SUPERVISOR)

        response = client.get(f"/api/cases/{case_id}", headers=auth_headers("outsider@example.com"))
        assert response.status_code == 404

    def test_closing_a_case_stamps_the_time(self, client: TestClient, headers):
        case_id = client.post(
            "/api/cases", headers=headers, json={"reference": "OP-CLOSE", "title": "Closing"}
        ).json()["id"]

        response = client.patch(f"/api/cases/{case_id}", headers=headers, json={"status": "closed"})
        assert response.status_code == 200
        assert response.json()["closed_at"] is not None


class TestEnrolment:
    def test_low_quality_enrolment_is_refused(self, client: TestClient, headers):
        """A weak enrolment degrades every future search against that subject."""
        import base64
        from io import BytesIO

        from PIL import Image

        buffer = BytesIO()
        Image.new("RGB", (200, 200), (128, 128, 128)).save(buffer, format="JPEG")
        flat = base64.b64encode(buffer.getvalue()).decode()

        response = client.post(
            "/api/subjects",
            headers=headers,
            json={"display_name": "Flat", "image_base64": flat, "lawful_basis": "test"},
        )
        assert response.status_code == 422

    def test_malformed_base64_is_rejected(self, client: TestClient, headers):
        response = client.post(
            "/api/subjects",
            headers=headers,
            json={"display_name": "Bad", "image_base64": "!!!not base64!!!", "lawful_basis": "test"},
        )
        assert response.status_code == 400

    def test_non_image_payload_is_rejected(self, client: TestClient, headers):
        import base64

        payload = base64.b64encode(b"this is a text file, not an image").decode()
        response = client.post(
            "/api/subjects",
            headers=headers,
            json={"display_name": "Text", "image_base64": payload, "lawful_basis": "test"},
        )
        assert response.status_code == 400

    def test_template_vectors_are_never_returned(self, client: TestClient, headers, face_b64):
        """A template can be inverted into an approximation of the face, so it is
        as sensitive as the photograph and must not leave the server."""
        response = client.post(
            "/api/subjects",
            headers=headers,
            json={"display_name": "Subject", "image_base64": face_b64[1], "lawful_basis": "test"},
        )
        body = response.json()
        serialized = str(body)
        assert "ciphertext" not in serialized
        assert "embedding" not in serialized
        assert "nonce" not in serialized


class TestSearchGovernance:
    def test_search_without_lawful_basis_is_refused(self, client: TestClient, headers, face_b64):
        """The system cannot judge lawfulness, but it can require that someone
        states a basis and preserve that statement."""
        response = client.post(
            "/api/imatch/search",
            headers=headers,
            json={"image_base64": face_b64[0], "mode": "single"},
        )
        assert response.status_code == 422
        assert "lawful basis" in response.json()["detail"].lower()

    def test_search_requires_authentication(self, client: TestClient, face_b64):
        response = client.post(
            "/api/imatch/search",
            json={"image_base64": face_b64[0], "lawful_basis": "test"},
        )
        assert response.status_code == 401

    def test_server_side_url_import_is_disabled(self, client: TestClient, headers):
        """Fetching a caller-supplied URL server-side is an SSRF primitive."""
        response = client.post(
            "/api/imatch/search",
            headers=headers,
            json={
                "source_url": "https://169.254.169.254/latest/meta-data/",
                "lawful_basis": "test",
                "mode": "url",
            },
        )
        assert response.status_code == 501

    def test_http_source_url_is_rejected_at_validation(self, client: TestClient, headers):
        response = client.post(
            "/api/imatch/search",
            headers=headers,
            json={"source_url": "http://example.com/face.jpg", "lawful_basis": "test"},
        )
        assert response.status_code == 422

    def test_search_on_empty_gallery_reports_no_match(self, client: TestClient, headers, face_b64):
        response = client.post(
            "/api/imatch/search",
            headers=headers,
            json={"image_base64": face_b64[0], "lawful_basis": "Warrant 2026/114", "mode": "single"},
        )
        body = response.json()
        assert body["gallery_size"] == 0
        assert body["candidates"] == []
        assert "investigative leads, not identifications" in body["notice"]


class TestEndToEndIdentification:
    """The complete workflow through the HTTP API, against the real engine.

    Enrol two different people, then search with a *second* photograph of the
    first person. The correct subject must come back ranked first, with a real
    similarity score.
    """

    def test_enrol_then_search_finds_the_right_person(
        self, client: TestClient, headers, face_paths
    ):
        identities = list(face_paths.items())[:4]
        if len(identities) < 2:
            pytest.skip("Need at least two identities with two images each.")

        import base64

        enrolled: dict[str, str] = {}
        for name, paths in identities:
            response = client.post(
                "/api/subjects",
                headers=headers,
                json={
                    "display_name": name,
                    "external_ref": name,
                    "image_base64": base64.b64encode(paths[0].read_bytes()).decode(),
                    "lawful_basis": "Warrant 2026/114",
                },
            )
            assert response.status_code == 201, response.text
            enrolled[name] = response.json()["subject"]["id"]

        target, target_paths = identities[0]
        probe = base64.b64encode(target_paths[1].read_bytes()).decode()

        response = client.post(
            "/api/imatch/search",
            headers=headers,
            json={
                "image_base64": probe,
                "lawful_basis": "Warrant 2026/114",
                "mode": "single",
                "top_k": 5,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["gallery_size"] == len(enrolled)
        assert body["candidates"], "search returned no candidates at all"
        top = body["candidates"][0]
        assert top["subject_id"] == enrolled[target], (
            f"expected {target} ranked first, got {top['subject_name']!r} at {top['score']:.4f}"
        )
        assert 0.0 < top["score"] <= 1.0
        assert body["duration_ms"] > 0
        assert body["model"]["backend"] == "insightface_arcface"

    def test_candidates_can_be_adjudicated(self, client: TestClient, headers, face_paths):
        import base64

        name, paths = next(iter(face_paths.items()))
        client.post(
            "/api/subjects",
            headers=headers,
            json={
                "display_name": name,
                "image_base64": base64.b64encode(paths[0].read_bytes()).decode(),
                "lawful_basis": "Warrant 2026/114",
            },
        )
        search = client.post(
            "/api/imatch/search",
            headers=headers,
            json={
                "image_base64": base64.b64encode(paths[1].read_bytes()).decode(),
                "lawful_basis": "Warrant 2026/114",
            },
        ).json()

        assert search["candidates"], "no candidate to adjudicate"
        candidate_id = search["candidates"][0]["id"]

        response = client.post(
            f"/api/imatch/candidates/{candidate_id}/adjudicate",
            headers=headers,
            json={"adjudication": "confirmed", "examiner_notes": "Verified side by side."},
        )
        assert response.status_code == 200
        assert response.json()["adjudication"] == "confirmed"

    def test_verify_compares_two_images(self, client: TestClient, headers, face_paths):
        import base64

        items = list(face_paths.items())
        same = items[0][1]
        other = items[1][1]

        def verify(a, b):
            return client.post(
                "/api/imatch/verify",
                headers=headers,
                json={
                    "reference_image_base64": base64.b64encode(a.read_bytes()).decode(),
                    "probe_image_base64": base64.b64encode(b.read_bytes()).decode(),
                    "lawful_basis": "Warrant 2026/114",
                },
            ).json()

        genuine = verify(same[0], same[1])
        impostor = verify(same[0], other[0])

        assert genuine["similarity"] > impostor["similarity"], (
            f"same-person {genuine['similarity']:.4f} did not exceed "
            f"different-person {impostor['similarity']:.4f}"
        )


class TestEngineStatus:
    def test_status_reports_the_loaded_model(self, client: TestClient, headers):
        response = client.get("/api/imatch/engine/status", headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert body["recognition_capable"] is True
        assert body["recognizer"]["backend"] == "insightface_arcface"
        assert body["recognizer"]["embedding_dim"] == 512
        assert body["device"]["effective"] in {"cpu", "cuda"}
        assert "match" in body["thresholds"]

    def test_health_is_public(self, client: TestClient):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] in {"ok", "degraded"}


class TestSecurityHeaders:
    def test_responses_are_not_cacheable(self, client: TestClient):
        """Biometric findings must not sit in a shared or browser cache."""
        response = client.get("/api/health")
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"

    def test_every_response_carries_a_request_id(self, client: TestClient):
        assert client.get("/api/health").headers.get("X-Request-ID")
