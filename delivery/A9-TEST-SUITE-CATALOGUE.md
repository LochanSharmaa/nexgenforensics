# A9 — Test Suite Catalogue

**Generated:** 2026-07-31 19:29 UTC ·
**Repository state:** `1e2e5d84de07`

| | |
|---|---|
| Test modules | 11 |
| Test functions | 197 |
| Test classes | 38 |
| Assertion statements | 362 |

Each entry below reproduces the test's complete source. A catalogue that
describes tests in prose can drift from what they actually check, and a test
that has silently stopped asserting anything still reads as coverage. These are
the tests themselves.

---

## What a passing suite does and does not demonstrate

Recorded in A2 Failure and Recovery Log: **the majority of this system's most
serious defects were not caught by these tests.** The silent-wrong-result and
methodological classes — the failures that produce confidently incorrect output
— were found by independent controls, by disbelieving results that looked
correct, and in two cases only by opening a browser and watching a real request
fail.

Specific examples, each of which passed the entire suite at the time:

| Defect | Why the suite could not catch it |
|---|---|
| GPU silently ran on CPU (F-01, F-02) | Results were correct, only slower. Nothing asserts on throughput. |
| Threshold drift causing a false match (F-03) | The test hard-coded the same stale constant the code used. |
| Training proxy reported a gain while the model got worse (F-04) | No test compares a checkpoint against the deployed model. |
| CSRF guard outside CORS (F-24) | `TestClient` speaks ASGI directly and never exercises CORS. |
| SameSite blocked the cookie (F-25) | No test uses a browser, and `localhost` vs `127.0.0.1` only differs in one. |

This catalogue is evidence of what is checked automatically. It is not evidence
that the system is correct.

---

## Index

| Module | Tests | Assertions |
|---|---|---|
| `backend/tests/test_api_auth.py` | 17 | 26 |
| `backend/tests/test_api_workflow.py` | 20 | 47 |
| `backend/tests/test_audit.py` | 14 | 23 |
| `backend/tests/test_auth_flows.py` | 29 | 61 |
| `backend/tests/test_engine.py` | 35 | 48 |
| `backend/tests/test_gallery_index.py` | 20 | 36 |
| `backend/tests/test_recognition_engine.py` | 22 | 33 |
| `backend/tests/test_security_headers.py` | 19 | 33 |
| `backend/tests_engine/test_adversarial_input.py` | 11 | 14 |
| `backend/tests_engine/test_persistence.py` | 7 | 21 |
| `backend/tests_engine/test_service_durability.py` | 3 | 20 |

---

# API, authentication, governance and workflow


## `backend/tests/test_api_auth.py`

**17 tests · 26 assertions**

| Class | Test | Asserts | What it checks |
|---|---|---|---|
| TestLogin | `test_login_returns_tokens` | 3 | *(no docstring)* |
| TestLogin | `test_wrong_password_is_rejected` | 1 | *(no docstring)* |
| TestLogin | `test_unknown_and_known_emails_give_the_same_error` | 2 | Distinguishable errors would let anyone enumerate registered accounts. |
| TestLogin | `test_inactive_user_cannot_log_in` | 1 | *(no docstring)* |
| TestLogin | `test_same_email_in_two_tenants_requires_a_tenant_slug` | 2 | *(no docstring)* |
| TestTokens | `test_protected_route_requires_a_token` | 1 | *(no docstring)* |
| TestTokens | `test_garbage_token_is_rejected` | 1 | *(no docstring)* |
| TestTokens | `test_refresh_token_is_not_accepted_as_an_access_token` | 1 | Otherwise a long-lived refresh token silently becomes a session. |
| TestTokens | `test_refresh_issues_a_new_access_token` | 2 | *(no docstring)* |
| TestTokens | `test_me_returns_the_signed_in_user` | 2 | *(no docstring)* |
| TestRoles | `test_investigator_cannot_enrol` | 1 | Enrolment decides who the system can find, so it needs supervisor rights. |
| TestRoles | `test_investigator_cannot_create_users` | 1 | *(no docstring)* |
| TestRoles | `test_admin_can_create_users_in_their_own_tenant` | 2 | *(no docstring)* |
| TestRoles | `test_weak_passwords_are_rejected` | 1 | *(no docstring)* |
| TestApiKeys | `test_admin_can_mint_and_use_a_key` | 2 | *(no docstring)* |
| TestApiKeys | `test_revoked_key_stops_working` | 2 | *(no docstring)* |
| TestApiKeys | `test_key_plaintext_is_never_returned_again` | 1 | *(no docstring)* |

### `TestLogin.test_login_returns_tokens`

```python
    def test_login_returns_tokens(self, client: TestClient, tenant_factory, user_factory):
        user_factory(tenant_factory())
        response = client.post(
            "/api/auth/login", json={"email": "investigator@example.com", "password": TEST_PASSWORD}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["access_token"] and body["refresh_token"]
        assert body["user"]["role"] == "investigator"
```

### `TestLogin.test_wrong_password_is_rejected`

```python
    def test_wrong_password_is_rejected(self, client: TestClient, tenant_factory, user_factory):
        user_factory(tenant_factory())
        response = client.post(
            "/api/auth/login", json={"email": "investigator@example.com", "password": "wrong-password-1!"}
        )
        assert response.status_code == 401
```

### `TestLogin.test_unknown_and_known_emails_give_the_same_error`

**Rationale as recorded in the test**

```text
Distinguishable errors would let anyone enumerate registered accounts.
```

```python
    def test_unknown_and_known_emails_give_the_same_error(
        self, client: TestClient, tenant_factory, user_factory
    ):
        """Distinguishable errors would let anyone enumerate registered accounts."""
        user_factory(tenant_factory())
        known = client.post(
            "/api/auth/login", json={"email": "investigator@example.com", "password": "wrong-password-1!"}
        )
        unknown = client.post(
            "/api/auth/login", json={"email": "nobody@example.com", "password": "wrong-password-1!"}
        )
        assert known.status_code == unknown.status_code == 401
        assert known.json()["detail"] == unknown.json()["detail"]
```

### `TestLogin.test_inactive_user_cannot_log_in`

```python
    def test_inactive_user_cannot_log_in(self, client: TestClient, tenant_factory, user_factory):
        from sqlmodel import Session

        from imatch_api.db.session import get_engine

        user = user_factory(tenant_factory())
        with Session(get_engine()) as session:
            stored = session.get(type(user), user.id)
            stored.active = False
            session.add(stored)
            session.commit()

        response = client.post(
            "/api/auth/login", json={"email": user.email, "password": TEST_PASSWORD}
        )
        assert response.status_code == 401
```

### `TestLogin.test_same_email_in_two_tenants_requires_a_tenant_slug`

```python
    def test_same_email_in_two_tenants_requires_a_tenant_slug(
        self, client: TestClient, tenant_factory, user_factory
    ):
        user_factory(tenant_factory("tenant-one"), email="shared@example.com")
        user_factory(tenant_factory("tenant-two"), email="shared@example.com")

        ambiguous = client.post(
            "/api/auth/login", json={"email": "shared@example.com", "password": TEST_PASSWORD}
        )
        assert ambiguous.status_code == 400

        resolved = client.post(
            "/api/auth/login",
            json={"email": "shared@example.com", "password": TEST_PASSWORD, "tenant": "tenant-two"},
        )
        assert resolved.status_code == 200
```

### `TestTokens.test_protected_route_requires_a_token`

```python
    def test_protected_route_requires_a_token(self, client: TestClient):
        assert client.get("/api/cases").status_code == 401
```

### `TestTokens.test_garbage_token_is_rejected`

```python
    def test_garbage_token_is_rejected(self, client: TestClient):
        response = client.get("/api/cases", headers={"Authorization": "Bearer not-a-real-token"})
        assert response.status_code == 401
```

### `TestTokens.test_refresh_token_is_not_accepted_as_an_access_token`

**Rationale as recorded in the test**

```text
Otherwise a long-lived refresh token silently becomes a session.
```

```python
    def test_refresh_token_is_not_accepted_as_an_access_token(
        self, client: TestClient, tenant_factory, user_factory
    ):
        """Otherwise a long-lived refresh token silently becomes a session."""
        user_factory(tenant_factory())
        login = client.post(
            "/api/auth/login", json={"email": "investigator@example.com", "password": TEST_PASSWORD}
        ).json()

        response = client.get(
            "/api/cases", headers={"Authorization": f"Bearer {login['refresh_token']}"}
        )
        assert response.status_code == 401
```

### `TestTokens.test_refresh_issues_a_new_access_token`

```python
    def test_refresh_issues_a_new_access_token(self, client: TestClient, tenant_factory, user_factory):
        user_factory(tenant_factory())
        login = client.post(
            "/api/auth/login", json={"email": "investigator@example.com", "password": TEST_PASSWORD}
        ).json()

        refreshed = client.post("/api/auth/refresh", json={"refresh_token": login["refresh_token"]})
        assert refreshed.status_code == 200
        assert refreshed.json()["access_token"]
```

### `TestTokens.test_me_returns_the_signed_in_user`

```python
    def test_me_returns_the_signed_in_user(self, client: TestClient, tenant_factory, user_factory, auth_headers):
        user_factory(tenant_factory())
        response = client.get("/api/auth/me", headers=auth_headers())
        assert response.status_code == 200
        assert response.json()["email"] == "investigator@example.com"
```

### `TestRoles.test_investigator_cannot_enrol`

**Rationale as recorded in the test**

```text
Enrolment decides who the system can find, so it needs supervisor rights.
```

```python
    def test_investigator_cannot_enrol(self, client: TestClient, tenant_factory, user_factory, auth_headers, face_b64):
        """Enrolment decides who the system can find, so it needs supervisor rights."""
        user_factory(tenant_factory())
        response = client.post(
            "/api/subjects",
            headers=auth_headers(),
            json={"display_name": "Test", "image_base64": face_b64[0], "lawful_basis": "test"},
        )
        assert response.status_code == 403
```

### `TestRoles.test_investigator_cannot_create_users`

```python
    def test_investigator_cannot_create_users(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        user_factory(tenant_factory())
        response = client.post(
            "/api/auth/users",
            headers=auth_headers(),
            json={"email": "new@example.com", "password": "Another-Strong-1!", "role": "admin"},
        )
        assert response.status_code == 403
```

### `TestRoles.test_admin_can_create_users_in_their_own_tenant`

```python
    def test_admin_can_create_users_in_their_own_tenant(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        tenant = tenant_factory()
        user_factory(tenant, email="admin@example.com", role=Role.ADMIN)
        response = client.post(
            "/api/auth/users",
            headers=auth_headers("admin@example.com"),
            json={"email": "new@example.com", "password": "Another-Strong-1!", "role": "investigator"},
        )
        assert response.status_code == 201
        assert response.json()["tenant_id"] == tenant.id
```

### `TestRoles.test_weak_passwords_are_rejected`

```python
    def test_weak_passwords_are_rejected(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        user_factory(tenant_factory(), email="admin@example.com", role=Role.ADMIN)
        response = client.post(
            "/api/auth/users",
            headers=auth_headers("admin@example.com"),
            json={"email": "weak@example.com", "password": "password", "role": "investigator"},
        )
        assert response.status_code == 422
```

### `TestApiKeys.test_admin_can_mint_and_use_a_key`

```python
    def test_admin_can_mint_and_use_a_key(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        user_factory(tenant_factory(), email="admin@example.com", role=Role.ADMIN)
        headers = auth_headers("admin@example.com")

        created = client.post("/api/admin/api-keys", headers=headers, json={"name": "integration"})
        assert created.status_code == 201
        plaintext = created.json()["api_key"]

        assert client.get("/api/cases", headers={"X-API-Key": plaintext}).status_code == 200
```

### `TestApiKeys.test_revoked_key_stops_working`

```python
    def test_revoked_key_stops_working(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        user_factory(tenant_factory(), email="admin@example.com", role=Role.ADMIN)
        headers = auth_headers("admin@example.com")

        created = client.post("/api/admin/api-keys", headers=headers, json={"name": "temp"}).json()
        assert client.get("/api/cases", headers={"X-API-Key": created["api_key"]}).status_code == 200

        client.delete(f"/api/admin/api-keys/{created['id']}", headers=headers)
        assert client.get("/api/cases", headers={"X-API-Key": created["api_key"]}).status_code == 401
```

### `TestApiKeys.test_key_plaintext_is_never_returned_again`

```python
    def test_key_plaintext_is_never_returned_again(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        user_factory(tenant_factory(), email="admin@example.com", role=Role.ADMIN)
        headers = auth_headers("admin@example.com")
        client.post("/api/admin/api-keys", headers=headers, json={"name": "listed"})

        listed = client.get("/api/admin/api-keys", headers=headers).json()
        assert listed and all("api_key" not in entry for entry in listed)
```


## `backend/tests/test_api_workflow.py`

**20 tests · 47 assertions**

| Class | Test | Asserts | What it checks |
|---|---|---|---|
| TestCases | `test_create_and_read_a_case` | 4 | *(no docstring)* |
| TestCases | `test_duplicate_reference_is_rejected` | 2 | *(no docstring)* |
| TestCases | `test_case_from_another_tenant_is_not_found` | 1 | A 403 here would confirm the id exists; 404 reveals nothing. |
| TestCases | `test_closing_a_case_stamps_the_time` | 2 | *(no docstring)* |
| TestEnrolment | `test_low_quality_enrolment_is_refused` | 1 | A weak enrolment degrades every future search against that subject. |
| TestEnrolment | `test_malformed_base64_is_rejected` | 1 | *(no docstring)* |
| TestEnrolment | `test_non_image_payload_is_rejected` | 1 | *(no docstring)* |
| TestEnrolment | `test_template_vectors_are_never_returned` | 3 | A template can be inverted into an approximation of the face, so it is as sensitive as the photograph and must not leave the server. |
| TestSearchGovernance | `test_search_without_lawful_basis_is_refused` | 2 | The system cannot judge lawfulness, but it can require that someone states a basis and preserve that statement. |
| TestSearchGovernance | `test_search_requires_authentication` | 1 | *(no docstring)* |
| TestSearchGovernance | `test_server_side_url_import_is_disabled` | 1 | Fetching a caller-supplied URL server-side is an SSRF primitive. |
| TestSearchGovernance | `test_http_source_url_is_rejected_at_validation` | 1 | *(no docstring)* |
| TestSearchGovernance | `test_search_on_empty_gallery_reports_no_match` | 3 | *(no docstring)* |
| TestEndToEndIdentification | `test_enrol_then_search_finds_the_right_person` | 8 | *(no docstring)* |
| TestEndToEndIdentification | `test_candidates_can_be_adjudicated` | 3 | *(no docstring)* |
| TestEndToEndIdentification | `test_verify_compares_two_images` | 1 | *(no docstring)* |
| TestEngineStatus | `test_status_reports_the_loaded_model` | 6 | *(no docstring)* |
| TestEngineStatus | `test_health_is_public` | 2 | *(no docstring)* |
| TestSecurityHeaders | `test_responses_are_not_cacheable` | 3 | Biometric findings must not sit in a shared or browser cache. |
| TestSecurityHeaders | `test_every_response_carries_a_request_id` | 1 | *(no docstring)* |

### `TestCases.test_create_and_read_a_case`

```python
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
```

### `TestCases.test_duplicate_reference_is_rejected`

```python
    def test_duplicate_reference_is_rejected(self, client: TestClient, headers):
        payload = {"reference": "OP-DUP", "title": "First"}
        assert client.post("/api/cases", headers=headers, json=payload).status_code == 201
        assert client.post("/api/cases", headers=headers, json=payload).status_code == 409
```

### `TestCases.test_case_from_another_tenant_is_not_found`

**Rationale as recorded in the test**

```text
A 403 here would confirm the id exists; 404 reveals nothing.
```

```python
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
```

### `TestCases.test_closing_a_case_stamps_the_time`

```python
    def test_closing_a_case_stamps_the_time(self, client: TestClient, headers):
        case_id = client.post(
            "/api/cases", headers=headers, json={"reference": "OP-CLOSE", "title": "Closing"}
        ).json()["id"]

        response = client.patch(f"/api/cases/{case_id}", headers=headers, json={"status": "closed"})
        assert response.status_code == 200
        assert response.json()["closed_at"] is not None
```

### `TestEnrolment.test_low_quality_enrolment_is_refused`

**Rationale as recorded in the test**

```text
A weak enrolment degrades every future search against that subject.
```

```python
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
```

### `TestEnrolment.test_malformed_base64_is_rejected`

```python
    def test_malformed_base64_is_rejected(self, client: TestClient, headers):
        response = client.post(
            "/api/subjects",
            headers=headers,
            json={"display_name": "Bad", "image_base64": "!!!not base64!!!", "lawful_basis": "test"},
        )
        assert response.status_code == 400
```

### `TestEnrolment.test_non_image_payload_is_rejected`

```python
    def test_non_image_payload_is_rejected(self, client: TestClient, headers):
        import base64

        payload = base64.b64encode(b"this is a text file, not an image").decode()
        response = client.post(
            "/api/subjects",
            headers=headers,
            json={"display_name": "Text", "image_base64": payload, "lawful_basis": "test"},
        )
        assert response.status_code == 400
```

### `TestEnrolment.test_template_vectors_are_never_returned`

**Rationale as recorded in the test**

```text
A template can be inverted into an approximation of the face, so it is
as sensitive as the photograph and must not leave the server.
```

```python
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
```

### `TestSearchGovernance.test_search_without_lawful_basis_is_refused`

**Rationale as recorded in the test**

```text
The system cannot judge lawfulness, but it can require that someone
states a basis and preserve that statement.
```

```python
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
```

### `TestSearchGovernance.test_search_requires_authentication`

```python
    def test_search_requires_authentication(self, client: TestClient, face_b64):
        response = client.post(
            "/api/imatch/search",
            json={"image_base64": face_b64[0], "lawful_basis": "test"},
        )
        assert response.status_code == 401
```

### `TestSearchGovernance.test_server_side_url_import_is_disabled`

**Rationale as recorded in the test**

```text
Fetching a caller-supplied URL server-side is an SSRF primitive.
```

```python
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
```

### `TestSearchGovernance.test_http_source_url_is_rejected_at_validation`

```python
    def test_http_source_url_is_rejected_at_validation(self, client: TestClient, headers):
        response = client.post(
            "/api/imatch/search",
            headers=headers,
            json={"source_url": "http://example.com/face.jpg", "lawful_basis": "test"},
        )
        assert response.status_code == 422
```

### `TestSearchGovernance.test_search_on_empty_gallery_reports_no_match`

```python
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
```

### `TestEndToEndIdentification.test_enrol_then_search_finds_the_right_person`

```python
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
```

### `TestEndToEndIdentification.test_candidates_can_be_adjudicated`

```python
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
```

### `TestEndToEndIdentification.test_verify_compares_two_images`

```python
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
```

### `TestEngineStatus.test_status_reports_the_loaded_model`

```python
    def test_status_reports_the_loaded_model(self, client: TestClient, headers):
        response = client.get("/api/imatch/engine/status", headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert body["recognition_capable"] is True
        assert body["recognizer"]["backend"] == "insightface_arcface"
        assert body["recognizer"]["embedding_dim"] == 512
        assert body["device"]["effective"] in {"cpu", "cuda"}
        assert "match" in body["thresholds"]
```

### `TestEngineStatus.test_health_is_public`

```python
    def test_health_is_public(self, client: TestClient):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] in {"ok", "degraded"}
```

### `TestSecurityHeaders.test_responses_are_not_cacheable`

**Rationale as recorded in the test**

```text
Biometric findings must not sit in a shared or browser cache.
```

```python
    def test_responses_are_not_cacheable(self, client: TestClient):
        """Biometric findings must not sit in a shared or browser cache."""
        response = client.get("/api/health")
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
```

### `TestSecurityHeaders.test_every_response_carries_a_request_id`

```python
    def test_every_response_carries_a_request_id(self, client: TestClient):
        assert client.get("/api/health").headers.get("X-Request-ID")
```


## `backend/tests/test_audit.py`

**14 tests · 23 assertions**

| Class | Test | Asserts | What it checks |
|---|---|---|---|
| TestChainIntegrity | `test_empty_chain_is_valid` | 1 | *(no docstring)* |
| TestChainIntegrity | `test_chain_links_records_in_order` | 3 | *(no docstring)* |
| TestChainIntegrity | `test_editing_a_record_breaks_verification` | 2 | This is the property the whole audit design exists for. |
| TestChainIntegrity | `test_deleting_a_record_breaks_verification` | 1 | *(no docstring)* |
| TestChainIntegrity | `test_records_carry_increasing_chain_positions` | 1 | Chain order comes from an explicit position, never from timestamps. |
| TestChainIntegrity | `test_a_burst_of_records_still_verifies` | 2 | Regression: 12 records written inside one clock tick must verify. |
| TestChainIntegrity | `test_chains_are_independent_per_tenant` | 3 | *(no docstring)* |
| TestChainIntegrity | `test_records_are_mirrored_to_disk` | 2 | *(no docstring)* |
| TestAuditThroughTheApi | `test_login_is_recorded` | 1 | *(no docstring)* |
| TestAuditThroughTheApi | `test_failed_login_is_recorded` | 1 | *(no docstring)* |
| TestAuditThroughTheApi | `test_audit_is_scoped_to_the_tenant` | 1 | *(no docstring)* |
| TestAuditThroughTheApi | `test_only_admins_can_verify_the_chain` | 1 | *(no docstring)* |
| TestAuditThroughTheApi | `test_admin_verification_passes_on_a_clean_chain` | 2 | *(no docstring)* |
| TestAuditThroughTheApi | `test_case_creation_is_audited` | 2 | *(no docstring)* |

### `TestChainIntegrity.test_empty_chain_is_valid`

```python
    def test_empty_chain_is_valid(self, audit, tenant_factory):
        tenant = tenant_factory()
        with Session(get_engine()) as session:
            assert audit.verify_chain(session, tenant.id).valid is True
```

### `TestChainIntegrity.test_chain_links_records_in_order`

```python
    def test_chain_links_records_in_order(self, audit, tenant_factory):
        tenant = tenant_factory()
        with Session(get_engine()) as session:
            first = audit.record(session, tenant_id=tenant.id, action="a.one")
            second = audit.record(session, tenant_id=tenant.id, action="a.two")
            session.commit()

            assert first.previous_hash == ""
            assert second.previous_hash == first.entry_hash
            assert audit.verify_chain(session, tenant.id).valid is True
```

### `TestChainIntegrity.test_editing_a_record_breaks_verification`

**Rationale as recorded in the test**

```text
This is the property the whole audit design exists for.
```

```python
    def test_editing_a_record_breaks_verification(self, audit, tenant_factory):
        """This is the property the whole audit design exists for."""
        tenant = tenant_factory()
        with Session(get_engine()) as session:
            audit.record(session, tenant_id=tenant.id, action="a.one")
            target = audit.record(session, tenant_id=tenant.id, action="a.two")
            audit.record(session, tenant_id=tenant.id, action="a.three")
            session.commit()

            stored = session.get(AuditRecord, target.id)
            stored.outcome = "quietly changed"
            session.add(stored)
            session.commit()

            verification = audit.verify_chain(session, tenant.id)
            assert verification.valid is False
            assert verification.broken_at == target.id
```

### `TestChainIntegrity.test_deleting_a_record_breaks_verification`

```python
    def test_deleting_a_record_breaks_verification(self, audit, tenant_factory):
        tenant = tenant_factory()
        with Session(get_engine()) as session:
            audit.record(session, tenant_id=tenant.id, action="a.one")
            middle = audit.record(session, tenant_id=tenant.id, action="a.two")
            audit.record(session, tenant_id=tenant.id, action="a.three")
            session.commit()

            session.delete(session.get(AuditRecord, middle.id))
            session.commit()

            assert audit.verify_chain(session, tenant.id).valid is False
```

### `TestChainIntegrity.test_records_carry_increasing_chain_positions`

**Rationale as recorded in the test**

```text
Chain order comes from an explicit position, never from timestamps.

created_at has millisecond resolution at best, so a burst of records
shares a tick; ordering by time with a random-UUID tiebreaker made
verification non-deterministic and could report tampering on an
untouched log.
```

```python
    def test_records_carry_increasing_chain_positions(self, audit, tenant_factory):
        """Chain order comes from an explicit position, never from timestamps.

        created_at has millisecond resolution at best, so a burst of records
        shares a tick; ordering by time with a random-UUID tiebreaker made
        verification non-deterministic and could report tampering on an
        untouched log.
        """
        tenant = tenant_factory()
        with Session(get_engine()) as session:
            positions = [
                audit.record(session, tenant_id=tenant.id, action=f"a.{i}").sequence
                for i in range(12)
            ]
            session.commit()

        assert positions == list(range(1, 13))
```

### `TestChainIntegrity.test_a_burst_of_records_still_verifies`

**Rationale as recorded in the test**

```text
Regression: 12 records written inside one clock tick must verify.
```

```python
    def test_a_burst_of_records_still_verifies(self, audit, tenant_factory):
        """Regression: 12 records written inside one clock tick must verify."""
        tenant = tenant_factory()
        with Session(get_engine()) as session:
            for i in range(12):
                audit.record(session, tenant_id=tenant.id, action=f"burst.{i}")
            session.commit()

            verification = audit.verify_chain(session, tenant.id)
            assert verification.valid is True, verification.reason
            assert verification.checked == 12
```

### `TestChainIntegrity.test_chains_are_independent_per_tenant`

```python
    def test_chains_are_independent_per_tenant(self, audit, tenant_factory):
        first = tenant_factory("tenant-one")
        second = tenant_factory("tenant-two")
        with Session(get_engine()) as session:
            audit.record(session, tenant_id=first.id, action="a.one")
            entry = audit.record(session, tenant_id=second.id, action="a.one")
            session.commit()

            # A second tenant's first record starts its own chain rather than
            # continuing another tenant's.
            assert entry.previous_hash == ""
            assert audit.verify_chain(session, first.id).valid is True
            assert audit.verify_chain(session, second.id).valid is True
```

### `TestChainIntegrity.test_records_are_mirrored_to_disk`

```python
    def test_records_are_mirrored_to_disk(self, audit, tenant_factory, tmp_path):
        tenant = tenant_factory()
        with Session(get_engine()) as session:
            record = audit.record(session, tenant_id=tenant.id, action="a.one")
            session.commit()
            # Read the hash before the session closes: commit expires attributes,
            # and a detached instance cannot refresh them.
            entry_hash = record.entry_hash

        lines = (tmp_path / "chain.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["entry_hash"] == entry_hash
```

### `TestAuditThroughTheApi.test_login_is_recorded`

```python
    def test_login_is_recorded(self, client: TestClient, tenant_factory, user_factory, auth_headers):
        tenant = tenant_factory()
        user_factory(tenant, email="admin@example.com", role=Role.ADMIN)
        headers = auth_headers("admin@example.com")

        records = client.get("/api/audit", headers=headers).json()
        assert any(record["action"] == ACTION_LOGIN for record in records)
```

### `TestAuditThroughTheApi.test_failed_login_is_recorded`

```python
    def test_failed_login_is_recorded(self, client: TestClient, tenant_factory, user_factory, auth_headers):
        tenant = tenant_factory()
        user_factory(tenant, email="admin@example.com", role=Role.ADMIN)
        client.post("/api/auth/login", json={"email": "admin@example.com", "password": "wrong-one-1!"})

        headers = auth_headers("admin@example.com")
        records = client.get("/api/audit", headers=headers).json()
        assert any(record["action"] == "auth.login_failed" for record in records)
```

### `TestAuditThroughTheApi.test_audit_is_scoped_to_the_tenant`

```python
    def test_audit_is_scoped_to_the_tenant(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        first = tenant_factory("tenant-one")
        user_factory(first, email="one@example.com", role=Role.ADMIN)
        second = tenant_factory("tenant-two")
        user_factory(second, email="two@example.com", role=Role.ADMIN)

        client.get("/api/audit", headers=auth_headers("one@example.com"))
        records = client.get("/api/audit", headers=auth_headers("two@example.com")).json()

        assert all(record["actor_label"] != "one@example.com" for record in records)
```

### `TestAuditThroughTheApi.test_only_admins_can_verify_the_chain`

```python
    def test_only_admins_can_verify_the_chain(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        tenant = tenant_factory()
        user_factory(tenant)
        assert client.get("/api/audit/verify", headers=auth_headers()).status_code == 403
```

### `TestAuditThroughTheApi.test_admin_verification_passes_on_a_clean_chain`

```python
    def test_admin_verification_passes_on_a_clean_chain(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        tenant = tenant_factory()
        user_factory(tenant, email="admin@example.com", role=Role.ADMIN)
        response = client.get("/api/audit/verify", headers=auth_headers("admin@example.com"))
        assert response.status_code == 200
        assert response.json()["valid"] is True
```

### `TestAuditThroughTheApi.test_case_creation_is_audited`

```python
    def test_case_creation_is_audited(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        tenant = tenant_factory()
        user_factory(tenant, email="admin@example.com", role=Role.ADMIN)
        headers = auth_headers("admin@example.com")

        created = client.post(
            "/api/cases",
            headers=headers,
            json={"reference": "OP-AUDIT", "title": "Audited", "lawful_basis": "Warrant 1"},
        ).json()

        records = client.get("/api/audit", headers=headers, params={"resource_id": created["id"]}).json()
        assert any(record["action"] == "case.create" for record in records)
        assert any(record["lawful_basis"] == "Warrant 1" for record in records)
```


## `backend/tests/test_auth_flows.py`

**29 tests · 61 assertions**

### Purpose of this module, as recorded in it

```text
Registration, verification, lockout and password reset.

Codes and reset tokens are read from the mail OUTBOX, not from the database and
not returned by any endpoint. That is deliberate: a test that reaches into
`user.otp_hash` would still pass if the e-mail were never sent, and "the code
reached the user" is the property that actually matters. Reading the outbox
exercises the same path a real mailbox does.
```

| Class | Test | Asserts | What it checks |
|---|---|---|---|
| TestRegistration | `test_registration_is_disabled_by_default` | 2 | *(no docstring)* |
| TestRegistration | `test_registration_creates_an_unverified_account_and_sends_a_code` | 5 | *(no docstring)* |
| TestRegistration | `test_password_is_never_stored_in_plaintext` | 2 | *(no docstring)* |
| TestRegistration | `test_weak_password_is_rejected` | 1 | *(no docstring)* |
| TestRegistration | `test_mismatched_confirmation_is_rejected` | 1 | *(no docstring)* |
| TestRegistration | `test_invalid_email_is_rejected` | 1 | *(no docstring)* |
| TestRegistration | `test_duplicate_registration_does_not_reveal_the_account` | 2 | Enumeration guard: the second attempt must be indistinguishable. |
| TestVerification | `test_correct_code_verifies_and_clears_the_secret` | 5 | *(no docstring)* |
| TestVerification | `test_wrong_code_is_rejected_and_counted` | 2 | *(no docstring)* |
| TestVerification | `test_expired_code_is_rejected` | 2 | *(no docstring)* |
| TestVerification | `test_code_is_single_use` | 2 | *(no docstring)* |
| TestVerification | `test_attempts_are_capped` | 1 | *(no docstring)* |
| TestVerification | `test_resend_invalidates_the_previous_code` | 4 | *(no docstring)* |
| TestVerification | `test_resend_is_capped_per_account` | 1 | *(no docstring)* |
| TestVerification | `test_resend_for_unknown_address_does_not_reveal_it` | 2 | *(no docstring)* |
| TestLoginGating | `test_unverified_account_cannot_log_in` | 2 | *(no docstring)* |
| TestLoginGating | `test_verified_account_can_log_in` | 3 | *(no docstring)* |
| TestLoginGating | `test_remember_me_extends_only_the_refresh_token` | 2 | *(no docstring)* |
| TestLockout | `test_account_locks_after_repeated_failures` | 3 | *(no docstring)* |
| TestLockout | `test_lock_expires` | 1 | *(no docstring)* |
| TestPasswordReset | `test_forgot_password_never_reveals_whether_the_account_exists` | 2 | *(no docstring)* |
| TestPasswordReset | `test_reset_changes_the_password_and_revokes_sessions` | 4 | *(no docstring)* |
| TestPasswordReset | `test_reset_token_is_single_use` | 2 | *(no docstring)* |
| TestPasswordReset | `test_expired_reset_token_is_rejected` | 1 | *(no docstring)* |
| TestPasswordReset | `test_reset_rejects_a_weak_password` | 1 | *(no docstring)* |
| TestPasswordReset | `test_unknown_reset_token_is_rejected` | 1 | *(no docstring)* |
| TestSessionRevocation | `test_logout_revokes_the_refresh_token` | 2 | *(no docstring)* |
| TestSessionRevocation | `test_refresh_rotates_and_the_old_token_dies` | 2 | *(no docstring)* |
| TestAuditTrail | `test_the_lifecycle_is_audited` | 2 | *(no docstring)* |

### `TestRegistration.test_registration_is_disabled_by_default`

```python
    def test_registration_is_disabled_by_default(self, anon_client, tmp_path, monkeypatch):
        monkeypatch.setenv("NEXGEN_ALLOW_SELF_REGISTRATION", "false")
        get_settings.cache_clear()
        seed_tenant()
        response = register(anon_client, "nobody@example.com")
        assert response.status_code == 403
        assert "disabled" in response.json()["detail"].lower()
```

### `TestRegistration.test_registration_creates_an_unverified_account_and_sends_a_code`

```python
    def test_registration_creates_an_unverified_account_and_sends_a_code(
        self, anon_client, mail_outbox
    ):
        seed_tenant()
        response = register(anon_client, "newuser@example.com")
        assert response.status_code == 201, response.text
        assert response.json()["message"] == "Registration successful. Please verify your email."

        user = load_user("newuser@example.com")
        assert user is not None and user.email_verified is False
        # Only the hash is stored, never the code itself.
        assert user.otp_hash and len(user.otp_hash) == 64
        assert latest_otp(mail_outbox).isdigit()
```

### `TestRegistration.test_password_is_never_stored_in_plaintext`

```python
    def test_password_is_never_stored_in_plaintext(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "hash@example.com")
        user = load_user("hash@example.com")
        assert STRONG not in user.password_hash
        assert user.password_hash.startswith("$argon2")
```

### `TestRegistration.test_weak_password_is_rejected`

```python
    def test_weak_password_is_rejected(self, anon_client, mail_outbox):
        seed_tenant()
        assert register(anon_client, "weak@example.com", password="short").status_code == 422
```

### `TestRegistration.test_mismatched_confirmation_is_rejected`

```python
    def test_mismatched_confirmation_is_rejected(self, anon_client, mail_outbox):
        seed_tenant()
        assert register(anon_client, "mm@example.com", confirm="Different-Pass-99!").status_code == 422
```

### `TestRegistration.test_invalid_email_is_rejected`

```python
    def test_invalid_email_is_rejected(self, anon_client, mail_outbox):
        seed_tenant()
        assert register(anon_client, "not-an-email").status_code == 422
```

### `TestRegistration.test_duplicate_registration_does_not_reveal_the_account`

**Rationale as recorded in the test**

```text
Enumeration guard: the second attempt must be indistinguishable.
```

```python
    def test_duplicate_registration_does_not_reveal_the_account(self, anon_client, mail_outbox):
        """Enumeration guard: the second attempt must be indistinguishable."""
        seed_tenant()
        first = register(anon_client, "dupe@example.com")
        second = register(anon_client, "dupe@example.com")
        assert first.status_code == second.status_code == 201
        assert first.json() == second.json()
```

### `TestVerification.test_correct_code_verifies_and_clears_the_secret`

```python
    def test_correct_code_verifies_and_clears_the_secret(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "verify@example.com")
        code = latest_otp(mail_outbox)

        response = anon_client.post("/api/auth/verify-email",
                                    json={"email": "verify@example.com", "otp": code})
        assert response.status_code == 200, response.text
        assert response.json()["email_verified"] is True

        user = load_user("verify@example.com")
        assert user.email_verified is True
        assert user.otp_hash is None and user.otp_expires_at is None
        assert user.otp_attempts == 0
```

### `TestVerification.test_wrong_code_is_rejected_and_counted`

```python
    def test_wrong_code_is_rejected_and_counted(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "wrong@example.com")
        response = anon_client.post("/api/auth/verify-email",
                                    json={"email": "wrong@example.com", "otp": "000000"})
        assert response.status_code == 400
        assert load_user("wrong@example.com").otp_attempts == 1
```

### `TestVerification.test_expired_code_is_rejected`

```python
    def test_expired_code_is_rejected(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "expired@example.com")
        code = latest_otp(mail_outbox)

        user = load_user("expired@example.com")
        user.otp_expires_at = utcnow() - timedelta(minutes=1)
        save_user(user)

        response = anon_client.post("/api/auth/verify-email",
                                    json={"email": "expired@example.com", "otp": code})
        assert response.status_code == 400
        assert "expired" in response.json()["detail"].lower()
```

### `TestVerification.test_code_is_single_use`

```python
    def test_code_is_single_use(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "once@example.com")
        code = latest_otp(mail_outbox)
        assert anon_client.post("/api/auth/verify-email",
                                json={"email": "once@example.com", "otp": code}).status_code == 200
        again = anon_client.post("/api/auth/verify-email",
                                 json={"email": "once@example.com", "otp": code})
        # Already verified: reports success without re-consuming anything.
        assert again.json()["email_verified"] is True
```

### `TestVerification.test_attempts_are_capped`

```python
    def test_attempts_are_capped(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "brute@example.com")
        settings = get_settings()
        codes = [f"{n:06d}" for n in range(settings.otp_max_attempts)]
        for guess in codes:
            anon_client.post("/api/auth/verify-email",
                             json={"email": "brute@example.com", "otp": guess})
        blocked = anon_client.post("/api/auth/verify-email",
                                   json={"email": "brute@example.com", "otp": "999999"})
        assert blocked.status_code == 429
```

### `TestVerification.test_resend_invalidates_the_previous_code`

```python
    def test_resend_invalidates_the_previous_code(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "resend@example.com")
        first = latest_otp(mail_outbox)
        assert anon_client.post("/api/auth/resend-otp",
                                json={"email": "resend@example.com"}).status_code == 200
        second = latest_otp(mail_outbox)
        assert first != second
        assert anon_client.post("/api/auth/verify-email",
                                json={"email": "resend@example.com", "otp": first}).status_code == 400
        assert anon_client.post("/api/auth/verify-email",
                                json={"email": "resend@example.com", "otp": second}).status_code == 200
```

### `TestVerification.test_resend_is_capped_per_account`

```python
    def test_resend_is_capped_per_account(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "flood@example.com")
        settings = get_settings()
        statuses = [
            anon_client.post("/api/auth/resend-otp", json={"email": "flood@example.com"}).status_code
            for _ in range(settings.otp_resend_max + 2)
        ]
        assert 429 in statuses
```

### `TestVerification.test_resend_for_unknown_address_does_not_reveal_it`

```python
    def test_resend_for_unknown_address_does_not_reveal_it(self, anon_client, mail_outbox):
        seed_tenant()
        response = anon_client.post("/api/auth/resend-otp", json={"email": "ghost@example.com"})
        assert response.status_code == 200
        assert not outbox(mail_outbox)
```

### `TestLoginGating.test_unverified_account_cannot_log_in`

```python
    def test_unverified_account_cannot_log_in(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "gated@example.com")
        response = anon_client.post("/api/auth/login",
                                    json={"email": "gated@example.com", "password": STRONG})
        assert response.status_code == 403
        assert response.json()["detail"] == "Please verify your email before logging in."
```

### `TestLoginGating.test_verified_account_can_log_in`

```python
    def test_verified_account_can_log_in(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "good@example.com")
        anon_client.post("/api/auth/verify-email",
                         json={"email": "good@example.com", "otp": latest_otp(mail_outbox)})
        response = anon_client.post("/api/auth/login",
                                    json={"email": "good@example.com", "password": STRONG})
        assert response.status_code == 200, response.text
        assert response.json()["access_token"]
        # HTTPOnly cookies are set alongside the body tokens.
        assert "nx_access" in response.cookies or "nx_access" in str(response.headers)
```

### `TestLoginGating.test_remember_me_extends_only_the_refresh_token`

```python
    def test_remember_me_extends_only_the_refresh_token(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "remember@example.com")
        anon_client.post("/api/auth/verify-email",
                         json={"email": "remember@example.com", "otp": latest_otp(mail_outbox)})
        short = anon_client.post("/api/auth/login",
                                 json={"email": "remember@example.com", "password": STRONG})
        long = anon_client.post("/api/auth/login",
                                json={"email": "remember@example.com", "password": STRONG,
                                      "remember_me": True})
        # Access-token lifetime must be identical; only the refresh differs.
        assert short.json()["expires_in"] == long.json()["expires_in"]

        import jwt
        settings = get_settings()
        decode = lambda t: jwt.decode(t, settings.resolved_jwt_secret(),
                                      algorithms=[settings.jwt_algorithm])
        assert decode(long.json()["refresh_token"])["exp"] > decode(short.json()["refresh_token"])["exp"]
```

### `TestLockout.test_account_locks_after_repeated_failures`

```python
    def test_account_locks_after_repeated_failures(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "lock@example.com")
        anon_client.post("/api/auth/verify-email",
                         json={"email": "lock@example.com", "otp": latest_otp(mail_outbox)})

        settings = get_settings()
        codes = []
        for _ in range(settings.max_failed_logins):
            codes.append(anon_client.post(
                "/api/auth/login",
                json={"email": "lock@example.com", "password": "Wrong-Password-1!"}).status_code)
        assert codes[-1] == 429, codes

        # Correct password is now refused too: the lock is on the account.
        blocked = anon_client.post("/api/auth/login",
                                   json={"email": "lock@example.com", "password": STRONG})
        assert blocked.status_code == 429
        assert "locked" in blocked.json()["detail"].lower()
```

### `TestLockout.test_lock_expires`

```python
    def test_lock_expires(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "unlock@example.com")
        anon_client.post("/api/auth/verify-email",
                         json={"email": "unlock@example.com", "otp": latest_otp(mail_outbox)})
        user = load_user("unlock@example.com")
        user.locked_until = utcnow() - timedelta(seconds=1)
        save_user(user)
        response = anon_client.post("/api/auth/login",
                                    json={"email": "unlock@example.com", "password": STRONG})
        assert response.status_code == 200
```

### `TestPasswordReset.test_forgot_password_never_reveals_whether_the_account_exists`

```python
    def test_forgot_password_never_reveals_whether_the_account_exists(
        self, anon_client, mail_outbox
    ):
        seed_tenant()
        register(anon_client, "real@example.com")
        known = anon_client.post("/api/auth/forgot-password", json={"email": "real@example.com"})
        unknown = anon_client.post("/api/auth/forgot-password", json={"email": "ghost@example.com"})
        assert known.status_code == unknown.status_code == 200
        assert known.json() == unknown.json()
```

### `TestPasswordReset.test_reset_changes_the_password_and_revokes_sessions`

```python
    def test_reset_changes_the_password_and_revokes_sessions(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "reset@example.com")
        anon_client.post("/api/auth/verify-email",
                         json={"email": "reset@example.com", "otp": latest_otp(mail_outbox)})
        signed_in = anon_client.post("/api/auth/login",
                                     json={"email": "reset@example.com", "password": STRONG})
        old_refresh = signed_in.json()["refresh_token"]

        anon_client.post("/api/auth/forgot-password", json={"email": "reset@example.com"})
        token = latest_reset_token(mail_outbox)

        new_password = "Brand-New-Secret-77!"
        response = anon_client.post("/api/auth/reset-password", json={
            "token": token, "password": new_password, "confirm_password": new_password})
        assert response.status_code == 200, response.text

        assert anon_client.post("/api/auth/login",
                                json={"email": "reset@example.com",
                                      "password": STRONG}).status_code == 401
        assert anon_client.post("/api/auth/login",
                                json={"email": "reset@example.com",
                                      "password": new_password}).status_code == 200
        # The refresh token issued before the reset must no longer work.
        assert anon_client.post("/api/auth/refresh",
                                json={"refresh_token": old_refresh}).status_code == 401
```

### `TestPasswordReset.test_reset_token_is_single_use`

```python
    def test_reset_token_is_single_use(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "single@example.com")
        anon_client.post("/api/auth/forgot-password", json={"email": "single@example.com"})
        token = latest_reset_token(mail_outbox)
        body = {"token": token, "password": "First-Change-88!",
                "confirm_password": "First-Change-88!"}
        assert anon_client.post("/api/auth/reset-password", json=body).status_code == 200
        assert anon_client.post("/api/auth/reset-password", json=body).status_code == 400
```

### `TestPasswordReset.test_expired_reset_token_is_rejected`

```python
    def test_expired_reset_token_is_rejected(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "stale@example.com")
        anon_client.post("/api/auth/forgot-password", json={"email": "stale@example.com"})
        token = latest_reset_token(mail_outbox)
        user = load_user("stale@example.com")
        user.reset_token_expires_at = utcnow() - timedelta(minutes=1)
        save_user(user)
        response = anon_client.post("/api/auth/reset-password", json={
            "token": token, "password": "Another-One-99!", "confirm_password": "Another-One-99!"})
        assert response.status_code == 400
```

### `TestPasswordReset.test_reset_rejects_a_weak_password`

```python
    def test_reset_rejects_a_weak_password(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "weakreset@example.com")
        anon_client.post("/api/auth/forgot-password", json={"email": "weakreset@example.com"})
        token = latest_reset_token(mail_outbox)
        response = anon_client.post("/api/auth/reset-password", json={
            "token": token, "password": "abc", "confirm_password": "abc"})
        assert response.status_code == 422
```

### `TestPasswordReset.test_unknown_reset_token_is_rejected`

```python
    def test_unknown_reset_token_is_rejected(self, anon_client, mail_outbox):
        seed_tenant()
        response = anon_client.post("/api/auth/reset-password", json={
            "token": "x" * 40, "password": "Valid-Password-42!",
            "confirm_password": "Valid-Password-42!"})
        assert response.status_code == 400
```

### `TestSessionRevocation.test_logout_revokes_the_refresh_token`

```python
    def test_logout_revokes_the_refresh_token(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "logout@example.com")
        anon_client.post("/api/auth/verify-email",
                         json={"email": "logout@example.com", "otp": latest_otp(mail_outbox)})
        tokens = anon_client.post("/api/auth/login",
                                  json={"email": "logout@example.com", "password": STRONG}).json()

        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        assert anon_client.post("/api/auth/logout", headers=headers).status_code == 204

        replay = anon_client.post("/api/auth/refresh",
                                  json={"refresh_token": tokens["refresh_token"]})
        assert replay.status_code == 401, "a logged-out refresh token must not work"
```

### `TestSessionRevocation.test_refresh_rotates_and_the_old_token_dies`

```python
    def test_refresh_rotates_and_the_old_token_dies(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "rotate@example.com")
        anon_client.post("/api/auth/verify-email",
                         json={"email": "rotate@example.com", "otp": latest_otp(mail_outbox)})
        first = anon_client.post("/api/auth/login",
                                 json={"email": "rotate@example.com", "password": STRONG}).json()
        second = anon_client.post("/api/auth/refresh",
                                  json={"refresh_token": first["refresh_token"]})
        assert second.status_code == 200
        replay = anon_client.post("/api/auth/refresh",
                                  json={"refresh_token": first["refresh_token"]})
        assert replay.status_code == 401, "refresh tokens must be single-use"
```

### `TestAuditTrail.test_the_lifecycle_is_audited`

```python
    def test_the_lifecycle_is_audited(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "audited@example.com")
        anon_client.post("/api/auth/verify-email",
                         json={"email": "audited@example.com", "otp": latest_otp(mail_outbox)})
        anon_client.post("/api/auth/login",
                         json={"email": "audited@example.com", "password": "Nope-Nope-11!"})
        anon_client.post("/api/auth/login",
                         json={"email": "audited@example.com", "password": STRONG})

        entries = [json.loads(line) for line in
                   Path(get_settings().audit_path).read_text(encoding="utf-8").splitlines() if line]
        actions = {e["action"] for e in entries}
        for expected in ("auth.register", "auth.otp_sent", "auth.otp_verified",
                         "auth.login_failed", "auth.login"):
            assert expected in actions, f"{expected} missing from {sorted(actions)}"

        login = next(e for e in entries if e["action"] == "auth.login")
        assert login["ip_address"] and login["user_agent"] and login["actor_id"]
```


## `backend/tests/test_engine.py`

**35 tests · 48 assertions**

| Class | Test | Asserts | What it checks |
|---|---|---|---|
| TestAlignment | `test_umeyama_recovers_a_known_transform` | 1 | *(no docstring)* |
| TestAlignment | `test_umeyama_rejects_degenerate_points` | 1 | *(no docstring)* |
| TestAlignment | `test_norm_crop_maps_landmarks_onto_the_reference_layout` | 2 | The whole point of alignment: landmarks must land where ArcFace expects. |
| TestAlignment | `test_pose_is_neutral_for_a_frontal_layout` | 2 | *(no docstring)* |
| TestAlignment | `test_pose_detects_a_turned_head` | 1 | *(no docstring)* |
| TestAlignment | `test_pose_detects_roll` | 1 | *(no docstring)* |
| TestQuality | `test_laplacian_variance_separates_sharp_from_blurred` | 1 | *(no docstring)* |
| TestQuality | `test_uniform_image_is_rejected` | 2 | *(no docstring)* |
| TestQuality | `test_dark_image_flags_brightness` | 1 | *(no docstring)* |
| TestQuality | `test_small_face_is_flagged` | 1 | *(no docstring)* |
| TestQuality | `test_quality_measures_the_face_not_the_frame` | 2 | A sharp face on a noisy background must not be scored on the background. |
| TestDecoding | `test_rejects_empty_payload` | 1 | *(no docstring)* |
| TestDecoding | `test_rejects_non_image_bytes` | 1 | *(no docstring)* |
| TestDecoding | `test_decodes_common_formats` | 1 | *(no docstring)* |
| TestTemplateEncryption | `test_round_trip` | 1 | *(no docstring)* |
| TestTemplateEncryption | `test_ciphertext_differs_across_calls` | 1 | A fixed nonce would leak that two subjects share an enrolment image. |
| TestTemplateEncryption | `test_cross_tenant_ciphertext_fails_to_decrypt` | 1 | Moving a row between tenants must break, not silently succeed. |
| TestTemplateEncryption | `test_wrong_key_fails` | 1 | *(no docstring)* |
| TestTemplateEncryption | `test_tampered_ciphertext_is_detected` | 1 | *(no docstring)* |
| TestTemplateEncryption | `test_rejects_wrong_key_length` | 1 | *(no docstring)* |
| TestDecisionEngine | `test_unavailable_engine_never_claims_a_match` | 2 | *(no docstring)* |
| TestDecisionEngine | `test_rejected_probe_is_not_searched` | 1 | *(no docstring)* |
| TestDecisionEngine | `test_empty_gallery_returns_no_match` | 1 | *(no docstring)* |
| TestDecisionEngine | `test_low_score_is_no_match` | 1 | *(no docstring)* |
| TestDecisionEngine | `test_borderline_score_goes_to_review` | 3 | A score inside the review band must reach a human. |
| TestDecisionEngine | `test_clear_score_is_a_candidate_match` | 2 | *(no docstring)* |
| TestDecisionEngine | `test_near_tie_on_a_large_gallery_forces_review` | 2 | A high score that barely beats the runner-up is the shape of a false match. |
| TestDecisionEngine | `test_a_lone_candidate_is_not_treated_as_a_tie` | 3 | With one candidate the margin is 0.0, which means "nothing else came close" -- the opposite of a dead heat. |
| TestDecisionEngine | `test_probe_flags_downgrade_a_passing_score` | 1 | *(no docstring)* |
| TestScoreNormalizer | `test_margin_is_the_top_two_gap` | 2 | *(no docstring)* |
| TestScoreNormalizer | `test_margin_of_single_score_is_zero` | 1 | *(no docstring)* |
| TestScoreNormalizer | `test_z_scores_are_centred` | 2 | *(no docstring)* |
| TestScoreNormalizer | `test_identical_scores_do_not_divide_by_zero` | 1 | *(no docstring)* |
| TestCohortNormalizer | `test_small_cohort_returns_the_raw_score` | 1 | *(no docstring)* |
| TestCohortNormalizer | `test_normalization_does_not_mutate_the_template` | 1 | Regression: an earlier version adjusted the embedding itself, which made a stored identity depend on unrelated search history. |

### `TestAlignment.test_umeyama_recovers_a_known_transform`

```python
    def test_umeyama_recovers_a_known_transform(self):
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        angle = np.pi / 6
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        target = (source @ rotation.T) * 2.5 + np.array([7.0, -3.0])

        transform = umeyama_similarity(source, target)
        recovered = source @ transform[:2, :2].T + transform[:2, 2]

        np.testing.assert_allclose(recovered, target, atol=1e-9)
```

### `TestAlignment.test_umeyama_rejects_degenerate_points`

```python
    def test_umeyama_rejects_degenerate_points(self):
        collinear = np.zeros((5, 2))
        with pytest.raises(ValueError):
            umeyama_similarity(collinear, ARCFACE_REFERENCE_5PT)
```

### `TestAlignment.test_norm_crop_maps_landmarks_onto_the_reference_layout`

**Rationale as recorded in the test**

```text
The whole point of alignment: landmarks must land where ArcFace expects.
```

```python
    def test_norm_crop_maps_landmarks_onto_the_reference_layout(self):
        """The whole point of alignment: landmarks must land where ArcFace expects."""
        image = noise_image(seed=1, size=400)
        # Landmarks scaled and shifted away from the canonical layout.
        landmarks = ARCFACE_REFERENCE_5PT * 2.0 + np.array([60.0, 40.0])

        transform = umeyama_similarity(landmarks, ARCFACE_REFERENCE_5PT)
        mapped = landmarks @ transform[:2, :2].T + transform[:2, 2]
        np.testing.assert_allclose(mapped, ARCFACE_REFERENCE_5PT, atol=1e-6)

        crop = norm_crop(image, landmarks)
        assert crop.size == (112, 112)
```

### `TestAlignment.test_pose_is_neutral_for_a_frontal_layout`

```python
    def test_pose_is_neutral_for_a_frontal_layout(self):
        pose = estimate_pose(ARCFACE_REFERENCE_5PT)
        assert abs(pose.yaw) < 5.0
        assert abs(pose.roll) < 5.0
```

### `TestAlignment.test_pose_detects_a_turned_head`

```python
    def test_pose_detects_a_turned_head(self):
        turned = ARCFACE_REFERENCE_5PT.copy()
        turned[2, 0] += 18.0  # push the nose toward one eye
        assert estimate_pose(turned).yaw > 15.0
```

### `TestAlignment.test_pose_detects_roll`

```python
    def test_pose_detects_roll(self):
        angle = np.pi / 8
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rolled = ARCFACE_REFERENCE_5PT @ rotation.T
        assert abs(estimate_pose(rolled).roll) > 15.0
```

### `TestQuality.test_laplacian_variance_separates_sharp_from_blurred`

```python
    def test_laplacian_variance_separates_sharp_from_blurred(self):
        from PIL import ImageFilter

        original = noise_image(seed=2)
        sharp = np.asarray(original.convert("L"), dtype=np.float64)
        blurred = np.asarray(
            original.filter(ImageFilter.GaussianBlur(6)).convert("L"), dtype=np.float64
        )
        assert laplacian_variance(sharp) > laplacian_variance(blurred) * 2
```

### `TestQuality.test_uniform_image_is_rejected`

```python
    def test_uniform_image_is_rejected(self):
        flat = Image.new("RGB", (300, 300), (128, 128, 128))
        report = ImageQualityFilter().evaluate(flat)
        assert not report.accepted
        assert "low_contrast" in report.reasons or "blur_risk" in report.reasons
```

### `TestQuality.test_dark_image_flags_brightness`

```python
    def test_dark_image_flags_brightness(self):
        dark = Image.new("RGB", (300, 300), (8, 8, 8))
        assert "brightness_out_of_range" in ImageQualityFilter().evaluate(dark).reasons
```

### `TestQuality.test_small_face_is_flagged`

```python
    def test_small_face_is_flagged(self):
        image = noise_image(seed=3, size=400)
        tiny = DetectedFace(box=FaceBox(10, 10, 40, 40, 0.99), confidence=0.99)
        assert "face_too_small" in ImageQualityFilter().evaluate(image, tiny).reasons
```

### `TestQuality.test_quality_measures_the_face_not_the_frame`

**Rationale as recorded in the test**

```text
A sharp face on a noisy background must not be scored on the background.
```

```python
    def test_quality_measures_the_face_not_the_frame(self):
        """A sharp face on a noisy background must not be scored on the background."""
        image = noise_image(seed=4, size=400)
        whole = ImageQualityFilter().evaluate(image)
        face = DetectedFace(box=FaceBox(100, 100, 300, 300, 0.99), confidence=0.99)
        cropped = ImageQualityFilter().evaluate(image, face)
        assert cropped.face_pixels == 200
        assert whole.face_pixels == 400
```

### `TestDecoding.test_rejects_empty_payload`

```python
    def test_rejects_empty_payload(self):
        with pytest.raises(InvalidImageError):
            decode_image(b"")
```

### `TestDecoding.test_rejects_non_image_bytes`

```python
    def test_rejects_non_image_bytes(self):
        with pytest.raises(InvalidImageError):
            decode_image(b"this is definitely not a JPEG")
```

### `TestDecoding.test_decodes_common_formats`

```python
    def test_decodes_common_formats(self):
        for fmt in ("JPEG", "PNG", "BMP"):
            decoded = decode_image(image_bytes(noise_image(seed=5), fmt))
            assert decoded.mode == "RGB"
```

### `TestTemplateEncryption.test_round_trip`

```python
    def test_round_trip(self):
        encryptor = TemplateEncryptor.generate()
        template = np.random.default_rng(0).normal(size=512).astype(np.float32)
        sealed = encryptor.encrypt(template, "tenant-a")
        np.testing.assert_allclose(encryptor.decrypt(sealed), template, rtol=1e-6)
```

### `TestTemplateEncryption.test_ciphertext_differs_across_calls`

**Rationale as recorded in the test**

```text
A fixed nonce would leak that two subjects share an enrolment image.
```

```python
    def test_ciphertext_differs_across_calls(self):
        """A fixed nonce would leak that two subjects share an enrolment image."""
        encryptor = TemplateEncryptor.generate()
        template = np.ones(512, dtype=np.float32)
        first = encryptor.encrypt(template, "tenant-a")
        second = encryptor.encrypt(template, "tenant-a")
        assert first.ciphertext != second.ciphertext
```

### `TestTemplateEncryption.test_cross_tenant_ciphertext_fails_to_decrypt`

**Rationale as recorded in the test**

```text
Moving a row between tenants must break, not silently succeed.
```

```python
    def test_cross_tenant_ciphertext_fails_to_decrypt(self):
        """Moving a row between tenants must break, not silently succeed."""
        encryptor = TemplateEncryptor.generate()
        sealed = encryptor.encrypt(np.ones(512, dtype=np.float32), "tenant-a")
        forged = type(sealed)(
            nonce=sealed.nonce,
            ciphertext=sealed.ciphertext,
            dimensions=sealed.dimensions,
            tenant_id="tenant-b",
        )
        with pytest.raises(TemplateDecryptionError):
            encryptor.decrypt(forged)
```

### `TestTemplateEncryption.test_wrong_key_fails`

```python
    def test_wrong_key_fails(self):
        sealed = TemplateEncryptor.generate().encrypt(np.ones(512, dtype=np.float32), "tenant-a")
        with pytest.raises(TemplateDecryptionError):
            TemplateEncryptor.generate().decrypt(sealed)
```

### `TestTemplateEncryption.test_tampered_ciphertext_is_detected`

```python
    def test_tampered_ciphertext_is_detected(self):
        import base64

        encryptor = TemplateEncryptor.generate()
        sealed = encryptor.encrypt(np.ones(512, dtype=np.float32), "tenant-a")
        raw = bytearray(base64.b64decode(sealed.ciphertext))
        raw[0] ^= 0xFF
        tampered = type(sealed)(
            nonce=sealed.nonce,
            ciphertext=base64.b64encode(bytes(raw)).decode(),
            dimensions=sealed.dimensions,
            tenant_id=sealed.tenant_id,
        )
        with pytest.raises(TemplateDecryptionError):
            encryptor.decrypt(tampered)
```

### `TestTemplateEncryption.test_rejects_wrong_key_length`

```python
    def test_rejects_wrong_key_length(self):
        with pytest.raises(ValueError):
            TemplateEncryptor(b"too-short")
```

### `TestDecisionEngine.test_unavailable_engine_never_claims_a_match`

```python
    def test_unavailable_engine_never_claims_a_match(self, decisions):
        decision = decisions.decide(
            top_score=0.99, recognition_capable=False, probe_accepted=True, gallery_size=100
        )
        assert decision.label == DECISION_UNAVAILABLE
        assert decision.confidence == 0.0
```

### `TestDecisionEngine.test_rejected_probe_is_not_searched`

```python
    def test_rejected_probe_is_not_searched(self, decisions):
        decision = decisions.decide(
            top_score=0.9,
            recognition_capable=True,
            probe_accepted=False,
            probe_reasons=("blur_risk",),
            gallery_size=100,
        )
        assert decision.label == DECISION_REJECTED
```

### `TestDecisionEngine.test_empty_gallery_returns_no_match`

```python
    def test_empty_gallery_returns_no_match(self, decisions):
        decision = decisions.decide(
            top_score=0.0, recognition_capable=True, probe_accepted=True, gallery_size=0
        )
        assert decision.label == DECISION_NO_MATCH
```

### `TestDecisionEngine.test_low_score_is_no_match`

```python
    def test_low_score_is_no_match(self, decisions):
        decision = decisions.decide(
            top_score=0.15, recognition_capable=True, probe_accepted=True, gallery_size=100, margin=0.1
        )
        assert decision.label == DECISION_NO_MATCH
```

### `TestDecisionEngine.test_borderline_score_goes_to_review`

**Rationale as recorded in the test**

```text
A score inside the review band must reach a human.

The probe score is DERIVED from the configured thresholds, not
hardcoded. It previously used a literal 0.36, which sat in the review
band only while the thresholds were 0.32/0.42. When they were
recalibrated to 0.2153/0.2871 for false-match control (BENCHMARKS.md
section 5c), 0.36 became a clear match and this test failed -- it was
asserting a number, not a behaviour.
```

```python
    def test_borderline_score_goes_to_review(self, decisions):
        """A score inside the review band must reach a human.

        The probe score is DERIVED from the configured thresholds, not
        hardcoded. It previously used a literal 0.36, which sat in the review
        band only while the thresholds were 0.32/0.42. When they were
        recalibrated to 0.2153/0.2871 for false-match control (BENCHMARKS.md
        section 5c), 0.36 became a clear match and this test failed -- it was
        asserting a number, not a behaviour.
        """
        t = decisions.config.thresholds
        midband = (t.review + t.match) / 2
        assert t.review < midband < t.match, "fixture assumption: a review band exists"

        decision = decisions.decide(
            top_score=midband,
            recognition_capable=True,
            probe_accepted=True,
            gallery_size=100,
            margin=0.1,
        )
        assert decision.label == DECISION_REVIEW
        assert "score_in_review_band" in decision.reasons
```

### `TestDecisionEngine.test_clear_score_is_a_candidate_match`

```python
    def test_clear_score_is_a_candidate_match(self, decisions):
        decision = decisions.decide(
            top_score=0.72, recognition_capable=True, probe_accepted=True, gallery_size=100, margin=0.2
        )
        assert decision.label == DECISION_MATCH
        assert "not a positive identification" in decision.explanation
```

### `TestDecisionEngine.test_near_tie_on_a_large_gallery_forces_review`

**Rationale as recorded in the test**

```text
A high score that barely beats the runner-up is the shape of a false match.
```

```python
    def test_near_tie_on_a_large_gallery_forces_review(self, decisions):
        """A high score that barely beats the runner-up is the shape of a false match."""
        decision = decisions.decide(
            top_score=0.72,
            recognition_capable=True,
            probe_accepted=True,
            gallery_size=5000,
            margin=0.01,
            candidate_count=4,
        )
        assert decision.label == DECISION_REVIEW
        assert "low_margin_over_runner_up" in decision.reasons
```

### `TestDecisionEngine.test_a_lone_candidate_is_not_treated_as_a_tie`

**Rationale as recorded in the test**

```text
With one candidate the margin is 0.0, which means "nothing else came
close" -- the opposite of a dead heat. Reading it as a tie would send
the strongest possible result to review and word it alarmingly.
```

```python
    def test_a_lone_candidate_is_not_treated_as_a_tie(self, decisions):
        """With one candidate the margin is 0.0, which means "nothing else came
        close" -- the opposite of a dead heat. Reading it as a tie would send
        the strongest possible result to review and word it alarmingly."""
        decision = decisions.decide(
            top_score=0.72,
            recognition_capable=True,
            probe_accepted=True,
            gallery_size=5000,
            margin=0.0,
            candidate_count=1,
        )
        assert decision.label == DECISION_MATCH
        assert "low_margin_over_runner_up" not in decision.reasons
        assert "no other enrolled subject scored above" in decision.explanation
```

### `TestDecisionEngine.test_probe_flags_downgrade_a_passing_score`

```python
    def test_probe_flags_downgrade_a_passing_score(self, decisions):
        decision = decisions.decide(
            top_score=0.80,
            recognition_capable=True,
            probe_accepted=True,
            probe_reasons=("liveness_below_threshold",),
            gallery_size=100,
            margin=0.3,
        )
        assert decision.label == DECISION_REVIEW
```

### `TestScoreNormalizer.test_margin_is_the_top_two_gap`

```python
    def test_margin_is_the_top_two_gap(self):
        assert ScoreNormalizer.margin(np.array([0.9, 0.6, 0.2])) == pytest.approx(0.3)
```

### `TestScoreNormalizer.test_margin_of_single_score_is_zero`

```python
    def test_margin_of_single_score_is_zero(self):
        assert ScoreNormalizer.margin(np.array([0.9])) == 0.0
```

### `TestScoreNormalizer.test_z_scores_are_centred`

```python
    def test_z_scores_are_centred(self):
        scores = ScoreNormalizer.z_scores(np.array([0.1, 0.2, 0.3, 0.4]))
        assert scores.mean() == pytest.approx(0.0, abs=1e-9)
```

### `TestScoreNormalizer.test_identical_scores_do_not_divide_by_zero`

```python
    def test_identical_scores_do_not_divide_by_zero(self):
        np.testing.assert_array_equal(ScoreNormalizer.z_scores(np.ones(5)), np.zeros(5))
```

### `TestCohortNormalizer.test_small_cohort_returns_the_raw_score`

```python
    def test_small_cohort_returns_the_raw_score(self):
        normalizer = CohortNormalizer()
        normalizer.set_cohort(np.random.default_rng(0).normal(size=(3, 512)).astype(np.float32))
        assert normalizer.normalize_score(np.ones(512, dtype=np.float32), 0.7) == 0.7
```

### `TestCohortNormalizer.test_normalization_does_not_mutate_the_template`

**Rationale as recorded in the test**

```text
Regression: an earlier version adjusted the embedding itself, which made
a stored identity depend on unrelated search history.
```

```python
    def test_normalization_does_not_mutate_the_template(self):
        """Regression: an earlier version adjusted the embedding itself, which made
        a stored identity depend on unrelated search history."""
        normalizer = CohortNormalizer()
        normalizer.set_cohort(np.random.default_rng(1).normal(size=(50, 512)).astype(np.float32))
        probe = np.random.default_rng(2).normal(size=512).astype(np.float32)
        before = probe.copy()
        normalizer.normalize_score(probe, 0.5)
        np.testing.assert_array_equal(probe, before)
```


## `backend/tests/test_gallery_index.py`

**20 tests · 36 assertions**

| Class | Test | Asserts | What it checks |
|---|---|---|---|
| TestTenantIsolation | `test_search_never_crosses_tenants` | 3 | *(no docstring)* |
| TestTenantIsolation | `test_unknown_tenant_returns_nothing` | 2 | *(no docstring)* |
| TestTenantIsolation | `test_clearing_one_tenant_leaves_others_intact` | 2 | *(no docstring)* |
| TestSearch | `test_identical_template_scores_one` | 2 | *(no docstring)* |
| TestSearch | `test_results_are_ranked_descending` | 3 | *(no docstring)* |
| TestSearch | `test_subjects_collapse_to_their_best_template` | 1 | One well-enrolled subject must not fill the whole candidate list. |
| TestSearch | `test_collapse_can_be_disabled` | 1 | *(no docstring)* |
| TestSearch | `test_min_score_truncates_results` | 1 | *(no docstring)* |
| TestSearch | `test_top_k_is_respected` | 1 | *(no docstring)* |
| TestSearch | `test_margin_reflects_the_top_two_gap` | 2 | *(no docstring)* |
| TestMutation | `test_remove_drops_the_template` | 2 | *(no docstring)* |
| TestMutation | `test_remove_unknown_template_is_false` | 1 | *(no docstring)* |
| TestMutation | `test_removal_keeps_remaining_rows_aligned` | 3 | Deleting from the middle must not shift ids away from their vectors. |
| TestMutation | `test_remove_subject_drops_every_template` | 2 | *(no docstring)* |
| TestMutation | `test_re_adding_the_same_id_replaces_it` | 3 | *(no docstring)* |
| TestMutation | `test_add_many_matches_repeated_add` | 1 | *(no docstring)* |
| TestMutation | `test_subject_count_is_distinct` | 2 | *(no docstring)* |
| TestValidation | `test_wrong_dimension_is_rejected` | 1 | *(no docstring)* |
| TestValidation | `test_nan_template_is_rejected` | 1 | *(no docstring)* |
| TestValidation | `test_unnormalized_input_is_normalized` | 2 | *(no docstring)* |

### `TestTenantIsolation.test_search_never_crosses_tenants`

```python
    def test_search_never_crosses_tenants(self):
        index = GalleryIndex(512)
        shared = unit(1)
        index.add("tenant-a", "template-a", "subject-a", shared)
        index.add("tenant-b", "template-b", "subject-b", shared)

        outcome = index.search("tenant-a", shared, top_k=10)

        assert len(outcome.matches) == 1
        assert outcome.matches[0].template_id == "template-a"
        assert outcome.gallery_size == 1
```

### `TestTenantIsolation.test_unknown_tenant_returns_nothing`

```python
    def test_unknown_tenant_returns_nothing(self):
        index = GalleryIndex(512)
        index.add("tenant-a", "template-a", "subject-a", unit(2))
        outcome = index.search("tenant-does-not-exist", unit(2), top_k=10)
        assert outcome.matches == ()
        assert outcome.gallery_size == 0
```

### `TestTenantIsolation.test_clearing_one_tenant_leaves_others_intact`

```python
    def test_clearing_one_tenant_leaves_others_intact(self):
        index = GalleryIndex(512)
        index.add("tenant-a", "t1", "s1", unit(3))
        index.add("tenant-b", "t2", "s2", unit(4))
        index.clear("tenant-a")
        assert index.size("tenant-a") == 0
        assert index.size("tenant-b") == 1
```

### `TestSearch.test_identical_template_scores_one`

```python
    def test_identical_template_scores_one(self):
        index = GalleryIndex(512)
        vector = unit(5)
        index.add("t", "template-1", "subject-1", vector)
        assert index.search("t", vector, top_k=1).top_score == pytest.approx(1.0, abs=1e-5)
```

### `TestSearch.test_results_are_ranked_descending`

```python
    def test_results_are_ranked_descending(self):
        index = GalleryIndex(512)
        for i in range(10):
            index.add("t", f"template-{i}", f"subject-{i}", unit(100 + i))
        scores = [match.score for match in index.search("t", unit(105), top_k=10).matches]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(1.0, abs=1e-5)
```

### `TestSearch.test_subjects_collapse_to_their_best_template`

**Rationale as recorded in the test**

```text
One well-enrolled subject must not fill the whole candidate list.
```

```python
    def test_subjects_collapse_to_their_best_template(self):
        """One well-enrolled subject must not fill the whole candidate list."""
        index = GalleryIndex(512)
        base = unit(6)
        for i in range(5):
            index.add("t", f"template-{i}", "subject-shared", base)
        index.add("t", "template-other", "subject-other", unit(7))

        matches = index.search("t", base, top_k=5).matches
        assert [match.subject_id for match in matches] == ["subject-shared", "subject-other"]
```

### `TestSearch.test_collapse_can_be_disabled`

```python
    def test_collapse_can_be_disabled(self):
        index = GalleryIndex(512)
        base = unit(8)
        for i in range(3):
            index.add("t", f"template-{i}", "subject-shared", base)
        matches = index.search("t", base, top_k=5, collapse_subjects=False).matches
        assert len(matches) == 3
```

### `TestSearch.test_min_score_truncates_results`

```python
    def test_min_score_truncates_results(self):
        index = GalleryIndex(512)
        for i in range(20):
            index.add("t", f"template-{i}", f"subject-{i}", unit(200 + i))
        outcome = index.search("t", unit(205), top_k=20, min_score=0.99)
        assert all(match.score >= 0.99 for match in outcome.matches)
```

### `TestSearch.test_top_k_is_respected`

```python
    def test_top_k_is_respected(self):
        index = GalleryIndex(512)
        for i in range(50):
            index.add("t", f"template-{i}", f"subject-{i}", unit(300 + i))
        assert len(index.search("t", unit(305), top_k=7).matches) == 7
```

### `TestSearch.test_margin_reflects_the_top_two_gap`

```python
    def test_margin_reflects_the_top_two_gap(self):
        index = GalleryIndex(512)
        probe = unit(9)
        index.add("t", "near", "subject-near", probe)
        far = unit(10)
        index.add("t", "far", "subject-far", far)
        outcome = index.search("t", probe, top_k=2)
        assert outcome.margin == pytest.approx(outcome.matches[0].score - outcome.matches[1].score)
```

### `TestMutation.test_remove_drops_the_template`

```python
    def test_remove_drops_the_template(self):
        index = GalleryIndex(512)
        index.add("t", "template-1", "subject-1", unit(11))
        assert index.remove("t", "template-1") is True
        assert index.size("t") == 0
```

### `TestMutation.test_remove_unknown_template_is_false`

```python
    def test_remove_unknown_template_is_false(self):
        assert GalleryIndex(512).remove("t", "nope") is False
```

### `TestMutation.test_removal_keeps_remaining_rows_aligned`

**Rationale as recorded in the test**

```text
Deleting from the middle must not shift ids away from their vectors.
```

```python
    def test_removal_keeps_remaining_rows_aligned(self):
        """Deleting from the middle must not shift ids away from their vectors."""
        index = GalleryIndex(512)
        vectors = {f"template-{i}": unit(400 + i) for i in range(5)}
        for template_id, vector in vectors.items():
            index.add("t", template_id, f"subject-{template_id}", vector)

        index.remove("t", "template-2")

        for template_id in ("template-0", "template-1", "template-3", "template-4"):
            outcome = index.search("t", vectors[template_id], top_k=1)
            assert outcome.matches[0].template_id == template_id
            assert outcome.matches[0].score == pytest.approx(1.0, abs=1e-5)
```

### `TestMutation.test_remove_subject_drops_every_template`

```python
    def test_remove_subject_drops_every_template(self):
        index = GalleryIndex(512)
        for i in range(4):
            index.add("t", f"template-{i}", "subject-x", unit(500 + i))
        index.add("t", "keep", "subject-y", unit(600))

        assert index.remove_subject("t", "subject-x") == 4
        assert index.size("t") == 1
```

### `TestMutation.test_re_adding_the_same_id_replaces_it`

```python
    def test_re_adding_the_same_id_replaces_it(self):
        index = GalleryIndex(512)
        index.add("t", "template-1", "subject-1", unit(12))
        index.add("t", "template-1", "subject-1", unit(13))
        assert index.size("t") == 1
        assert index.search("t", unit(13), top_k=1).top_score == pytest.approx(1.0, abs=1e-5)
```

### `TestMutation.test_add_many_matches_repeated_add`

```python
    def test_add_many_matches_repeated_add(self):
        rows = [(f"template-{i}", f"subject-{i}", unit(700 + i), {}) for i in range(6)]
        bulk = GalleryIndex(512)
        bulk.add_many("t", rows)
        single = GalleryIndex(512)
        for template_id, subject_id, vector, meta in rows:
            single.add("t", template_id, subject_id, vector, meta)

        probe = unit(703)
        assert [m.template_id for m in bulk.search("t", probe, top_k=6).matches] == [
            m.template_id for m in single.search("t", probe, top_k=6).matches
        ]
```

### `TestMutation.test_subject_count_is_distinct`

```python
    def test_subject_count_is_distinct(self):
        index = GalleryIndex(512)
        for i in range(3):
            index.add("t", f"template-{i}", "subject-1", unit(800 + i))
        index.add("t", "template-x", "subject-2", unit(900))
        assert index.size("t") == 4
        assert index.subject_count("t") == 2
```

### `TestValidation.test_wrong_dimension_is_rejected`

```python
    def test_wrong_dimension_is_rejected(self):
        with pytest.raises(ValueError, match="512-d"):
            GalleryIndex(512).add("t", "template-1", "subject-1", np.ones(128, dtype=np.float32))
```

### `TestValidation.test_nan_template_is_rejected`

```python
    def test_nan_template_is_rejected(self):
        broken = np.full(512, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            GalleryIndex(512).add("t", "template-1", "subject-1", broken)
```

### `TestValidation.test_unnormalized_input_is_normalized`

```python
    def test_unnormalized_input_is_normalized(self):
        index = GalleryIndex(512)
        index.add("t", "template-1", "subject-1", unit(14) * 42.0)
        assert index.search("t", unit(14), top_k=1).top_score == pytest.approx(1.0, abs=1e-5)
```


## `backend/tests/test_recognition_engine.py`

**22 tests · 33 assertions**

### Purpose of this module, as recorded in it

```text
Tests that prove the recognition engine actually recognizes people.

Everything else in the suite can pass while the system is incapable of its one
job. These run the real model on real photographs and assert on measured
behaviour: same-person pairs must score higher than different-person pairs, and
identification must find the right subject in a gallery.

They skip when the model or the face dataset is unavailable, and they never
assert a fixed accuracy figure -- the numbers reported here are properties of
this model on this dataset, not a claim about the product.
```

| Class | Test | Asserts | What it checks |
|---|---|---|---|
| TestEngineLoads | `test_a_real_model_is_loaded` | 3 | *(no docstring)* |
| TestEngineLoads | `test_detector_produces_landmarks` | 1 | Without landmarks there is no proper alignment, and accuracy collapses. |
| TestEngineLoads | `test_device_is_reported_honestly` | 2 | *(no docstring)* |
| TestEmbeddings | `test_embedding_is_512d_and_unit_length` | 3 | *(no docstring)* |
| TestEmbeddings | `test_encoding_is_deterministic` | 2 | The same bytes must always give the same template, or one person enrolled twice becomes two people. |
| TestEmbeddings | `test_encoding_does_not_drift_with_use` | 2 | Regression: query-dependent state used to leak into stored templates. |
| TestEmbeddings | `test_embeddings_are_not_hashes` | 1 | A hash of pixels would make two images of one person unrelated. |
| TestSeparation | `test_same_person_scores_higher_than_different_people` | 2 | The single claim this product rests on. |
| TestSeparation | `test_impostor_scores_cluster_near_zero` | 1 | *(no docstring)* |
| TestSeparation | `test_false_match_rate_is_low_at_the_default_threshold` | 2 | *(no docstring)* |
| TestSeparation | `test_report_measured_distributions` | 1 | Not an assertion so much as a record of what this build measured. |
| TestGallerySearch | `test_rank1_identification_finds_the_right_person` | 1 | End-to-end through the index, not just pairwise similarity. |
| TestGallerySearch | `test_search_is_tenant_isolated_with_real_templates` | 2 | *(no docstring)* |
| TestGallerySearch | `test_faiss_and_numpy_agree` | 1 | FAISS is a speed optimisation, never a different answer. |
| TestGallerySearch | `test_search_latency_is_reasonable` | 1 | *(no docstring)* |
| TestRefusals | `test_flat_image_has_no_face` | 1 | *(no docstring)* |
| TestRefusals | `test_noise_has_no_face` | 1 | *(no docstring)* |
| TestRefusals | `test_non_image_bytes_are_rejected` | 1 | *(no docstring)* |
| TestRefusals | `test_empty_payload_is_rejected` | 1 | *(no docstring)* |
| TestRefusals | `test_unknown_model_pack_fails_loudly` | 1 | A bad configuration must raise, not silently substitute something. |
| TestPreCroppedFaces | `test_tightly_cropped_faces_are_detected` | 1 | AgeDB images are 112x112 crops with no margin. |
| TestPreCroppedFaces | `test_padding_is_reported` | 2 | The examiner should be able to tell how a detection was obtained. |

### `TestEngineLoads.test_a_real_model_is_loaded`

```python
    def test_a_real_model_is_loaded(self, engine_runtime):
        info = engine_runtime.recognizer.info
        assert info.backend == "insightface_arcface"
        assert info.embedding_dim == 512
        assert info.recognition_network  # the actual network name, not a label
```

### `TestEngineLoads.test_detector_produces_landmarks`

**Rationale as recorded in the test**

```text
Without landmarks there is no proper alignment, and accuracy collapses.
```

```python
    def test_detector_produces_landmarks(self, engine_runtime):
        """Without landmarks there is no proper alignment, and accuracy collapses."""
        assert engine_runtime.detector.produces_landmarks is True
```

### `TestEngineLoads.test_device_is_reported_honestly`

```python
    def test_device_is_reported_honestly(self, engine_runtime):
        status = engine_runtime.status()
        assert status["device"]["effective"] in {"cpu", "cuda"}
        assert status["device"]["providers"]
```

### `TestEmbeddings.test_embedding_is_512d_and_unit_length`

```python
    def test_embedding_is_512d_and_unit_length(self, pipeline, face_paths):
        path = next(iter(face_paths.values()))[0]
        result = pipeline.encode_bytes(path.read_bytes())
        assert result.embedding.shape == (512,)
        assert float(np.linalg.norm(result.embedding)) == pytest.approx(1.0, abs=1e-4)
```

### `TestEmbeddings.test_encoding_is_deterministic`

**Rationale as recorded in the test**

```text
The same bytes must always give the same template, or one person
enrolled twice becomes two people.
```

```python
    def test_encoding_is_deterministic(self, pipeline, face_paths):
        """The same bytes must always give the same template, or one person
        enrolled twice becomes two people."""
        payload = next(iter(face_paths.values()))[0].read_bytes()
        first = pipeline.encode_bytes(payload).embedding
        second = pipeline.encode_bytes(payload).embedding
        assert float(np.dot(first, second)) == pytest.approx(1.0, abs=1e-5)
```

### `TestEmbeddings.test_encoding_does_not_drift_with_use`

**Rationale as recorded in the test**

```text
Regression: query-dependent state used to leak into stored templates.
```

```python
    def test_encoding_does_not_drift_with_use(self, pipeline, face_paths):
        """Regression: query-dependent state used to leak into stored templates."""
        paths = list(face_paths.values())
        payload = paths[0][0].read_bytes()
        baseline = pipeline.encode_bytes(payload).embedding
        for other in paths[1:6]:
            pipeline.encode_bytes(other[0].read_bytes())
        assert float(np.dot(pipeline.encode_bytes(payload).embedding, baseline)) == pytest.approx(
            1.0, abs=1e-5
        )
```

### `TestEmbeddings.test_embeddings_are_not_hashes`

**Rationale as recorded in the test**

```text
A hash of pixels would make two images of one person unrelated.

This is the specific failure this codebase previously shipped, so it is
asserted directly rather than only implied by the separation tests.
```

```python
    def test_embeddings_are_not_hashes(self, pipeline, face_paths):
        """A hash of pixels would make two images of one person unrelated.

        This is the specific failure this codebase previously shipped, so it is
        asserted directly rather than only implied by the separation tests.
        """
        paths = next(iter(face_paths.values()))
        a = pipeline.encode_bytes(paths[0].read_bytes()).embedding
        b = pipeline.encode_bytes(paths[1].read_bytes()).embedding
        similarity = float(np.dot(a, b))
        # Independent 512-d unit vectors sit near 0 with sd ~0.044, so anything
        # above 0.15 is far outside what any hash could produce.
        assert similarity > 0.15, (
            f"Same-person similarity {similarity:.4f} is indistinguishable from random. "
            "The embedding is not encoding identity."
        )
```

### `TestSeparation.test_same_person_scores_higher_than_different_people`

**Rationale as recorded in the test**

```text
The single claim this product rests on.
```

```python
    def test_same_person_scores_higher_than_different_people(self, score_distributions):
        """The single claim this product rests on."""
        genuine, impostor = score_distributions
        assert genuine.size and impostor.size
        assert genuine.mean() > impostor.mean() + 0.15, (
            f"genuine mean {genuine.mean():.4f} vs impostor mean {impostor.mean():.4f}: "
            "the engine is not separating identities."
        )
```

### `TestSeparation.test_impostor_scores_cluster_near_zero`

```python
    def test_impostor_scores_cluster_near_zero(self, score_distributions):
        _, impostor = score_distributions
        assert abs(float(impostor.mean())) < 0.20
```

### `TestSeparation.test_false_match_rate_is_low_at_the_default_threshold`

```python
    def test_false_match_rate_is_low_at_the_default_threshold(self, score_distributions, engine_runtime):
        genuine, impostor = score_distributions
        threshold = engine_runtime.config.thresholds.match
        false_match_rate = float((impostor >= threshold).mean())
        true_match_rate = float((genuine >= threshold).mean())
        assert false_match_rate < 0.02, (
            f"FAR {false_match_rate:.2%} at threshold {threshold}. "
            "Recalibrate with scripts/calibrate_threshold.py."
        )
        assert true_match_rate > 0.40, f"TAR only {true_match_rate:.2%} at threshold {threshold}."
```

### `TestSeparation.test_report_measured_distributions`

**Rationale as recorded in the test**

```text
Not an assertion so much as a record of what this build measured.
```

```python
    def test_report_measured_distributions(self, score_distributions, capsys):
        """Not an assertion so much as a record of what this build measured."""
        genuine, impostor = score_distributions
        with capsys.disabled():
            print(
                f"\n  genuine  n={genuine.size} mean={genuine.mean():.4f} sd={genuine.std():.4f}"
                f"\n  impostor n={impostor.size} mean={impostor.mean():.4f} sd={impostor.std():.4f}"
                f"\n  separation={genuine.mean() - impostor.mean():.4f}"
            )
        assert genuine.mean() > impostor.mean()
```

### `TestGallerySearch.test_rank1_identification_finds_the_right_person`

**Rationale as recorded in the test**

```text
End-to-end through the index, not just pairwise similarity.
```

```python
    def test_rank1_identification_finds_the_right_person(self, templates):
        """End-to-end through the index, not just pairwise similarity."""
        index = GalleryIndex(512)
        probes: list[tuple[str, np.ndarray]] = []

        for name, vectors in templates.items():
            index.add("tenant", f"{name}-enrolled", name, vectors[0])
            probes.append((name, vectors[1]))

        correct = sum(
            1
            for name, probe in probes
            if (matches := index.search("tenant", probe, top_k=1).matches)
            and matches[0].subject_id == name
        )
        accuracy = correct / len(probes)
        assert accuracy > 0.70, (
            f"Rank-1 accuracy {accuracy:.1%} over {len(probes)} probes "
            f"in a {len(templates)}-subject gallery."
        )
```

### `TestGallerySearch.test_search_is_tenant_isolated_with_real_templates`

```python
    def test_search_is_tenant_isolated_with_real_templates(self, templates):
        vectors = next(iter(templates.values()))
        index = GalleryIndex(512)
        index.add("tenant-a", "t1", "s1", vectors[0])
        index.add("tenant-b", "t2", "s2", vectors[0])

        outcome = index.search("tenant-a", vectors[1], top_k=10)
        assert len(outcome.matches) == 1
        assert outcome.matches[0].template_id == "t1"
```

### `TestGallerySearch.test_faiss_and_numpy_agree`

**Rationale as recorded in the test**

```text
FAISS is a speed optimisation, never a different answer.
```

```python
    def test_faiss_and_numpy_agree(self, templates):
        """FAISS is a speed optimisation, never a different answer."""
        if not faiss_available():
            pytest.skip("faiss is not installed.")

        index = GalleryIndex(512)
        for name, vectors in templates.items():
            index.add("tenant", f"{name}-0", name, vectors[0])

        probe = next(iter(templates.values()))[1]
        shard = index._shards["tenant"]  # noqa: SLF001 - comparing the two backends

        faiss_scores = shard.scores(probe / np.linalg.norm(probe))
        numpy_scores = (shard.vectors @ (probe / np.linalg.norm(probe))).astype(np.float32)
        np.testing.assert_allclose(faiss_scores, numpy_scores, atol=1e-5)
```

### `TestGallerySearch.test_search_latency_is_reasonable`

```python
    def test_search_latency_is_reasonable(self, templates):
        index = GalleryIndex(512)
        for name, vectors in templates.items():
            for position, vector in enumerate(vectors):
                index.add("tenant", f"{name}-{position}", name, vector)

        probe = next(iter(templates.values()))[0]
        started = time.perf_counter()
        for _ in range(20):
            index.search("tenant", probe, top_k=10)
        per_search_ms = (time.perf_counter() - started) / 20 * 1000
        assert per_search_ms < 100, f"{per_search_ms:.1f} ms per search on {index.size('tenant')} templates."
```

### `TestRefusals.test_flat_image_has_no_face`

```python
    def test_flat_image_has_no_face(self, pipeline):
        with pytest.raises(NoFaceDetectedError):
            pipeline.encode_bytes(image_bytes(flat_image()))
```

### `TestRefusals.test_noise_has_no_face`

```python
    def test_noise_has_no_face(self, pipeline):
        with pytest.raises(NoFaceDetectedError):
            pipeline.encode_bytes(image_bytes(noise_image(seed=3)))
```

### `TestRefusals.test_non_image_bytes_are_rejected`

```python
    def test_non_image_bytes_are_rejected(self, pipeline):
        with pytest.raises(InvalidImageError):
            pipeline.encode_bytes(b"this is not an image")
```

### `TestRefusals.test_empty_payload_is_rejected`

```python
    def test_empty_payload_is_rejected(self, pipeline):
        with pytest.raises(InvalidImageError):
            pipeline.encode_bytes(b"")
```

### `TestRefusals.test_unknown_model_pack_fails_loudly`

**Rationale as recorded in the test**

```text
A bad configuration must raise, not silently substitute something.
```

```python
    def test_unknown_model_pack_fails_loudly(self, engine_runtime):
        """A bad configuration must raise, not silently substitute something."""
        from nexgen_engine.config import EngineConfig
        from nexgen_engine.runtime import EngineRuntime

        runtime = EngineRuntime(EngineConfig(model_pack="does-not-exist"))
        with pytest.raises(EngineUnavailableError, match="Unknown model pack"):
            runtime.warm_up()
```

### `TestPreCroppedFaces.test_tightly_cropped_faces_are_detected`

**Rationale as recorded in the test**

```text
AgeDB images are 112x112 crops with no margin.

Without pad-and-retry the detector finds nothing in any of them, which
is the common case for mugshots and database thumbnails.
```

```python
    def test_tightly_cropped_faces_are_detected(self, pipeline, face_paths):
        """AgeDB images are 112x112 crops with no margin.

        Without pad-and-retry the detector finds nothing in any of them, which
        is the common case for mugshots and database thumbnails.
        """
        detected = 0
        for paths in list(face_paths.values())[:10]:
            try:
                pipeline.encode_bytes(paths[0].read_bytes())
                detected += 1
            except NoFaceDetectedError:
                pass
        assert detected >= 9, f"only {detected}/10 pre-cropped faces detected."
```

### `TestPreCroppedFaces.test_padding_is_reported`

**Rationale as recorded in the test**

```text
The examiner should be able to tell how a detection was obtained.
```

```python
    def test_padding_is_reported(self, pipeline, face_paths):
        """The examiner should be able to tell how a detection was obtained."""
        result = pipeline.encode_bytes(next(iter(face_paths.values()))[0].read_bytes())
        assert result.padded_detection is True
        assert result.timings.total_ms > 0
```


## `backend/tests/test_security_headers.py`

**19 tests · 33 assertions**

### Purpose of this module, as recorded in it

```text
Security response headers and CSRF enforcement.

These assert the guard REFUSES, not merely that it can be satisfied. A CSRF
layer that every fixture quietly primes would pass a suite while protecting
nothing, so the refusals are tested here directly with an unprimed client.
```

| Class | Test | Asserts | What it checks |
|---|---|---|---|
| TestSecurityHeaders | `test_baseline_headers_are_present` | 8 | *(no docstring)* |
| TestSecurityHeaders | `test_api_responses_get_the_strict_csp` | 3 | *(no docstring)* |
| TestSecurityHeaders | `test_docs_get_a_relaxed_csp_so_swagger_still_renders` | 2 | *(no docstring)* |
| TestSecurityHeaders | `test_errors_also_carry_the_headers` | 2 | *(no docstring)* |
| TestCsrfTokens | `test_a_fresh_token_validates` | 1 | *(no docstring)* |
| TestCsrfTokens | `test_a_token_from_another_secret_is_rejected` | 1 | *(no docstring)* |
| TestCsrfTokens | `test_a_tampered_token_is_rejected` | 2 | *(no docstring)* |
| TestCsrfTokens | `test_an_expired_token_is_rejected` | 1 | *(no docstring)* |
| TestCsrfTokens | `test_a_future_dated_token_is_rejected` | 1 | *(no docstring)* |
| TestCsrfTokens | `test_malformed_tokens_are_rejected` | 1 | *(no docstring)* |
| TestCsrfEnforcement | `test_state_change_without_a_token_is_refused` | 2 | *(no docstring)* |
| TestCsrfEnforcement | `test_header_without_the_matching_cookie_is_refused` | 1 | Double-submit: a valid signature alone is not enough. |
| TestCsrfEnforcement | `test_cookie_without_the_header_is_refused` | 1 | *(no docstring)* |
| TestCsrfEnforcement | `test_mismatched_pair_is_refused` | 1 | *(no docstring)* |
| TestCsrfEnforcement | `test_a_matching_pair_passes_the_guard` | 1 | *(no docstring)* |
| TestCsrfEnforcement | `test_safe_methods_need_no_token` | 1 | *(no docstring)* |
| TestCsrfEnforcement | `test_bearer_authenticated_requests_are_exempt` | 1 | A cross-origin page cannot set Authorization, so those requests are not forgeable and must not be made to carry a token -- otherwise every existing API client b |
| TestCsrfEnforcement | `test_api_key_requests_are_exempt` | 1 | *(no docstring)* |
| TestCsrfEnforcement | `test_csrf_cookie_is_readable_by_script` | 2 | Unlike the session cookies, this one MUST NOT be HTTPOnly -- the page has to read it to echo it back. |

### `TestSecurityHeaders.test_baseline_headers_are_present`

```python
    def test_baseline_headers_are_present(self, raw_client):
        response = raw_client.get("/api/health")
        headers = response.headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert headers["Cache-Control"] == "no-store"
        assert headers["Cross-Origin-Opener-Policy"] == "same-origin"
        assert headers["Cross-Origin-Resource-Policy"] == "same-origin"
        assert headers["X-Permitted-Cross-Domain-Policies"] == "none"
        assert "camera=()" in headers["Permissions-Policy"]
```

### `TestSecurityHeaders.test_api_responses_get_the_strict_csp`

```python
    def test_api_responses_get_the_strict_csp(self, raw_client):
        csp = raw_client.get("/api/health").headers["Content-Security-Policy"]
        # A JSON API should never be a source of executable or embeddable
        # content, so the policy denies everything by default.
        assert "default-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "base-uri 'none'" in csp
```

### `TestSecurityHeaders.test_docs_get_a_relaxed_csp_so_swagger_still_renders`

```python
    def test_docs_get_a_relaxed_csp_so_swagger_still_renders(self, raw_client):
        response = raw_client.get("/docs")
        if response.status_code == 404:
            pytest.skip("docs disabled in this configuration")
        csp = response.headers["Content-Security-Policy"]
        # Swagger UI is CDN-hosted and uses inline styles; the strict policy
        # would leave a blank interactive page rather than an obvious error.
        assert "cdn.jsdelivr.net" in csp
        assert "frame-ancestors 'none'" in csp
```

### `TestSecurityHeaders.test_errors_also_carry_the_headers`

```python
    def test_errors_also_carry_the_headers(self, raw_client):
        response = raw_client.get("/api/imatch/searches")
        assert response.status_code in (401, 403)
        assert response.headers["X-Content-Type-Options"] == "nosniff"
```

### `TestCsrfTokens.test_a_fresh_token_validates`

```python
    def test_a_fresh_token_validates(self):
        assert validate_csrf_token(issue_csrf_token(SECRET), SECRET)
```

### `TestCsrfTokens.test_a_token_from_another_secret_is_rejected`

```python
    def test_a_token_from_another_secret_is_rejected(self):
        assert not validate_csrf_token(issue_csrf_token("different-secret"), SECRET)
```

### `TestCsrfTokens.test_a_tampered_token_is_rejected`

```python
    def test_a_tampered_token_is_rejected(self):
        token = issue_csrf_token(SECRET)
        nonce, issued, signature = token.split(".")
        assert not validate_csrf_token(f"{nonce}x.{issued}.{signature}", SECRET)
        assert not validate_csrf_token(f"{nonce}.{issued}.{signature[:-1]}0", SECRET)
```

### `TestCsrfTokens.test_an_expired_token_is_rejected`

```python
    def test_an_expired_token_is_rejected(self):
        token = issue_csrf_token(SECRET)
        assert not validate_csrf_token(token, SECRET, max_age=-1)
```

### `TestCsrfTokens.test_a_future_dated_token_is_rejected`

```python
    def test_a_future_dated_token_is_rejected(self):
        nonce = "abc"
        issued = str(int(time.time()) + 4000)
        from hashlib import sha256
        import hmac as _hmac
        signature = _hmac.new(SECRET.encode(), f"{nonce}.{issued}".encode(), sha256).hexdigest()
        assert not validate_csrf_token(f"{nonce}.{issued}.{signature}", SECRET)
```

### `TestCsrfTokens.test_malformed_tokens_are_rejected`

```python
    def test_malformed_tokens_are_rejected(self):
        for bad in (None, "", "nodots", "only.two", "a.b.c.d"):
            assert not validate_csrf_token(bad, SECRET)
```

### `TestCsrfEnforcement.test_state_change_without_a_token_is_refused`

```python
    def test_state_change_without_a_token_is_refused(self, raw_client):
        response = raw_client.post("/api/auth/login",
                                   json={"email": "a@example.com", "password": "whatever"})
        assert response.status_code == 403
        assert "CSRF" in response.json()["detail"]
```

### `TestCsrfEnforcement.test_header_without_the_matching_cookie_is_refused`

**Rationale as recorded in the test**

```text
Double-submit: a valid signature alone is not enough.

An attacker can obtain a signed token by visiting the site themselves.
What they cannot do is read the victim's cookie, so the two must match.
```

```python
    def test_header_without_the_matching_cookie_is_refused(self, raw_client):
        """Double-submit: a valid signature alone is not enough.

        An attacker can obtain a signed token by visiting the site themselves.
        What they cannot do is read the victim's cookie, so the two must match.
        """
        from imatch_api.core.config import get_settings

        token = issue_csrf_token(get_settings().resolved_jwt_secret())
        response = raw_client.post("/api/auth/login",
                                   json={"email": "a@example.com", "password": "x"},
                                   headers={CSRF_HEADER: token})
        assert response.status_code == 403
```

### `TestCsrfEnforcement.test_cookie_without_the_header_is_refused`

```python
    def test_cookie_without_the_header_is_refused(self, raw_client):
        raw_client.get("/api/auth/csrf")  # sets the cookie only
        response = raw_client.post("/api/auth/login",
                                   json={"email": "a@example.com", "password": "x"})
        assert response.status_code == 403
```

### `TestCsrfEnforcement.test_mismatched_pair_is_refused`

```python
    def test_mismatched_pair_is_refused(self, raw_client):
        from imatch_api.core.config import get_settings

        raw_client.get("/api/auth/csrf")
        other = issue_csrf_token(get_settings().resolved_jwt_secret())
        response = raw_client.post("/api/auth/login",
                                   json={"email": "a@example.com", "password": "x"},
                                   headers={CSRF_HEADER: other})
        assert response.status_code == 403
```

### `TestCsrfEnforcement.test_a_matching_pair_passes_the_guard`

```python
    def test_a_matching_pair_passes_the_guard(self, raw_client):
        token = raw_client.get("/api/auth/csrf").json()["csrf_token"]
        response = raw_client.post("/api/auth/login",
                                   json={"email": "nobody@example.com", "password": "x"},
                                   headers={CSRF_HEADER: token})
        # Past the guard; the credentials themselves are simply wrong.
        assert response.status_code == 401
```

### `TestCsrfEnforcement.test_safe_methods_need_no_token`

```python
    def test_safe_methods_need_no_token(self, raw_client):
        assert raw_client.get("/api/health").status_code == 200
```

### `TestCsrfEnforcement.test_bearer_authenticated_requests_are_exempt`

**Rationale as recorded in the test**

```text
A cross-origin page cannot set Authorization, so those requests are
not forgeable and must not be made to carry a token -- otherwise every
existing API client breaks for no security gain.
```

```python
    def test_bearer_authenticated_requests_are_exempt(self, raw_client):
        """A cross-origin page cannot set Authorization, so those requests are
        not forgeable and must not be made to carry a token -- otherwise every
        existing API client breaks for no security gain."""
        response = raw_client.post("/api/auth/logout",
                                   headers={"Authorization": "Bearer not-a-real-token"})
        # 401 from authentication, NOT 403 from the CSRF guard.
        assert response.status_code == 401
```

### `TestCsrfEnforcement.test_api_key_requests_are_exempt`

```python
    def test_api_key_requests_are_exempt(self, raw_client):
        response = raw_client.post("/api/auth/logout", headers={"X-API-Key": "nope"})
        assert response.status_code == 401
```

### `TestCsrfEnforcement.test_csrf_cookie_is_readable_by_script`

**Rationale as recorded in the test**

```text
Unlike the session cookies, this one MUST NOT be HTTPOnly -- the page
has to read it to echo it back.
```

```python
    def test_csrf_cookie_is_readable_by_script(self, raw_client):
        """Unlike the session cookies, this one MUST NOT be HTTPOnly -- the page
        has to read it to echo it back."""
        response = raw_client.get("/api/auth/csrf")
        cookie_header = response.headers.get("set-cookie", "")
        assert CSRF_COOKIE in cookie_header
        assert "httponly" not in cookie_header.lower()
```


# Recognition engine, persistence, adversarial input


## `backend/tests_engine/test_adversarial_input.py`

**11 tests · 14 assertions**

### Purpose of this module, as recorded in it

```text
Item 33 — adversarial and malformed input handling.

WHAT THIS IS CHECKING, AND WHY IT MATTERS
------------------------------------------
Every input here is something a real operator could plausibly submit: a
truncated download, a screenshot with no face, a scanned page, a file that was
renamed to .jpg. None of them should produce a 500.

The distinction this suite enforces is between a TYPED failure and an
UNTYPED one:

  * A typed failure (UnsupportedImageError, ImageTooLargeError,
    InvalidImageError, NoFaceDetectedError) is one the API maps to a 4xx with
    a message the operator can act on.
  * Any other exception escaping the pipeline becomes a 500. In this system a
    500 is worse than a rejection: it tells the operator nothing, it may leak
    a stack trace, and in a batch it can abort work that was otherwise fine.

So the assertion is not merely "it raised" — it is "it raised something the
API layer knows how to turn into a clean answer".

A few cases below deliberately assert that a HARMLESS input SUCCEEDS. A
validator that rejects everything would pass a naive version of this suite
while making the product useless.
```

| Class | Test | Asserts | What it checks |
|---|---|---|---|
| — | `test_malformed_base64_raises_handled_error` | 1 | Garbage in the base64 field must not escape as a raw binascii error. |
| — | `test_zero_byte_payload_is_rejected` | 1 | A zero-byte file is the classic truncated-download case. |
| — | `test_oversized_payload_rejected_before_decoding` | 1 | The size gate must fire on the ENCODED length. |
| — | `test_data_url_prefix_is_tolerated` | 1 | Browsers produce data: URLs; stripping the prefix must not be lossy. |
| — | `test_renamed_non_image_is_rejected_by_content_not_name` | 1 | A PDF or ZIP renamed to .jpg must fail on magic bytes. |
| — | `test_real_formats_are_accepted` | 2 | Guard against a validator that rejects everything. |
| — | `test_pipeline_rejects_corrupt_bytes_cleanly` | 1 | Corrupt bytes must raise a HANDLED error, never an unmapped exception. |
| — | `test_pipeline_rejects_degenerate_geometry` | 1 | Extreme aspect ratios must not crash the detector or aligner. |
| — | `test_pipeline_reports_no_face_rather_than_crashing` | 1 | A valid image with no face is the single most common real rejection. |
| — | `test_pipeline_survives_large_but_legal_image` | 1 | A 4000x3000 phone photo is normal input, not an attack. |
| — | `test_real_face_still_succeeds` | 3 | The control. |

### `test_malformed_base64_raises_handled_error`

**Rationale as recorded in the test**

```text
Garbage in the base64 field must not escape as a raw binascii error.
```

```python
@pytest.mark.parametrize(
    "payload,label",
    [
        ("", "empty string"),
        ("!!!!not base64!!!!", "invalid base64 alphabet"),
        ("YWJj", "valid base64, but decodes to 'abc' - not an image"),
        ("=", "lone padding character"),
        ("A", "single char, not a valid base64 quantum"),
    ],
)
def test_malformed_base64_raises_handled_error(payload, label):
    """Garbage in the base64 field must not escape as a raw binascii error."""
    try:
        raw = decode_base64_image(payload, MAX_BYTES)
    except HANDLED:
        return  # rejected at decode, correct
    # Decoded to *something*; it must then be rejected by format sniffing.
    with pytest.raises(HANDLED):
        sniff_content_type(raw)
```

### `test_zero_byte_payload_is_rejected`

**Rationale as recorded in the test**

```text
A zero-byte file is the classic truncated-download case.
```

```python
def test_zero_byte_payload_is_rejected():
    """A zero-byte file is the classic truncated-download case."""
    with pytest.raises(HANDLED):
        sniff_content_type(decode_base64_image(_b64(b""), MAX_BYTES))
```

### `test_oversized_payload_rejected_before_decoding`

**Rationale as recorded in the test**

```text
The size gate must fire on the ENCODED length.

Checking only after decoding would mean allocating the full payload in
memory first, which is the denial-of-service the gate exists to prevent.
```

```python
def test_oversized_payload_rejected_before_decoding():
    """The size gate must fire on the ENCODED length.

    Checking only after decoding would mean allocating the full payload in
    memory first, which is the denial-of-service the gate exists to prevent.
    """
    huge = "A" * (MAX_BYTES * 4 // 3 + 4096)
    with pytest.raises(ImageTooLargeError):
        decode_base64_image(huge, MAX_BYTES)
```

### `test_data_url_prefix_is_tolerated`

**Rationale as recorded in the test**

```text
Browsers produce data: URLs; stripping the prefix must not be lossy.
```

```python
def test_data_url_prefix_is_tolerated():
    """Browsers produce data: URLs; stripping the prefix must not be lossy."""
    raw = _png(64, 64)
    assert decode_base64_image(f"data:image/png;base64,{_b64(raw)}", MAX_BYTES) == raw
```

### `test_renamed_non_image_is_rejected_by_content_not_name`

**Rationale as recorded in the test**

```text
A PDF or ZIP renamed to .jpg must fail on magic bytes.

Trusting a client-supplied filename would be the vulnerability here.
```

```python
def test_renamed_non_image_is_rejected_by_content_not_name():
    """A PDF or ZIP renamed to .jpg must fail on magic bytes.

    Trusting a client-supplied filename would be the vulnerability here.
    """
    for raw, what in [
        (b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", "PDF"),
        (b"PK\x03\x04" + b"\x00" * 32, "ZIP"),
        (b"#!/bin/sh\nrm -rf /\n", "shell script"),
        (b"\x00" * 64, "null bytes"),
    ]:
        with pytest.raises(UnsupportedImageError):
            sniff_content_type(raw)
```

### `test_real_formats_are_accepted`

**Rationale as recorded in the test**

```text
Guard against a validator that rejects everything.
```

```python
def test_real_formats_are_accepted():
    """Guard against a validator that rejects everything."""
    assert sniff_content_type(_png(32, 32)) == "image/png"
    assert sniff_content_type(_jpeg(32, 32)) == "image/jpeg"
```

### `test_pipeline_rejects_corrupt_bytes_cleanly`

**Rationale as recorded in the test**

```text
Corrupt bytes must raise a HANDLED error, never an unmapped exception.
```

```python
@pytest.mark.slow
@pytest.mark.parametrize(
    "raw,label",
    [
        (b"", "zero bytes"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 16, "JPEG magic then garbage"),
        (_jpeg(64, 64)[:40], "truncated JPEG"),
        (b"GIF89a" + b"\x00" * 32, "GIF header, unsupported"),
    ],
)
def test_pipeline_rejects_corrupt_bytes_cleanly(pipeline, raw, label):
    """Corrupt bytes must raise a HANDLED error, never an unmapped exception."""
    with pytest.raises(HANDLED):
        pipeline.encode_bytes(raw)
```

### `test_pipeline_rejects_degenerate_geometry`

**Rationale as recorded in the test**

```text
Extreme aspect ratios must not crash the detector or aligner.

A 1x4000 strip is a realistic accident (a bad crop), and it exercises
resize/padding maths where a divide-by-zero or negative dimension is easy
to introduce.
```

```python
@pytest.mark.slow
@pytest.mark.parametrize(
    "w,h,label",
    [
        (1, 1, "single pixel"),
        (1, 4000, "1px wide, extreme aspect ratio"),
        (4000, 1, "1px tall, extreme aspect ratio"),
        (8, 8, "smaller than any face"),
    ],
)
def test_pipeline_rejects_degenerate_geometry(pipeline, w, h, label):
    """Extreme aspect ratios must not crash the detector or aligner.

    A 1x4000 strip is a realistic accident (a bad crop), and it exercises
    resize/padding maths where a divide-by-zero or negative dimension is easy
    to introduce.
    """
    with pytest.raises(HANDLED):
        pipeline.encode_bytes(_png(w, h))
```

### `test_pipeline_reports_no_face_rather_than_crashing`

**Rationale as recorded in the test**

```text
A valid image with no face is the single most common real rejection.

It must be NoFaceDetectedError specifically -- the operator needs to know
the image was fine but contained no face, not that something broke.
```

```python
@pytest.mark.slow
@pytest.mark.parametrize(
    "raw,label",
    [
        (_png(512, 512, (255, 255, 255)), "blank white image"),
        (_png(512, 512, (0, 0, 0)), "blank black image"),
        (_jpeg(512, 512), "random noise"),
    ],
    ids=["blank-white", "blank-black", "random-noise"],
)
def test_pipeline_reports_no_face_rather_than_crashing(pipeline, raw, label):
    """A valid image with no face is the single most common real rejection.

    It must be NoFaceDetectedError specifically -- the operator needs to know
    the image was fine but contained no face, not that something broke.
    """
    with pytest.raises(NoFaceDetectedError):
        pipeline.encode_bytes(raw)
```

### `test_pipeline_survives_large_but_legal_image`

**Rationale as recorded in the test**

```text
A 4000x3000 phone photo is normal input, not an attack.

It must not be rejected for size alone. It contains no face, so the
expected outcome is a clean NoFaceDetectedError rather than a timeout,
a memory error, or a hang.
```

```python
@pytest.mark.slow
def test_pipeline_survives_large_but_legal_image(pipeline):
    """A 4000x3000 phone photo is normal input, not an attack.

    It must not be rejected for size alone. It contains no face, so the
    expected outcome is a clean NoFaceDetectedError rather than a timeout,
    a memory error, or a hang.
    """
    with pytest.raises(NoFaceDetectedError):
        pipeline.encode_bytes(_jpeg(4000, 3000))
```

### `test_real_face_still_succeeds`

**Rationale as recorded in the test**

```text
The control. Everything above asserts rejection; this proves the
pipeline has not simply been made to reject all input.
```

```python
@pytest.mark.slow
def test_real_face_still_succeeds(pipeline):
    """The control. Everything above asserts rejection; this proves the
    pipeline has not simply been made to reject all input."""
    agedb = _BACKEND.parent / "src_extracted/AgeDB/AgeDB"
    faces = sorted(agedb.glob("*.jpg"))
    if not faces:
        pytest.skip("AgeDB imagery not available")
    result = pipeline.encode_bytes(faces[0].read_bytes())
    assert result.embedding.shape[0] == 512
    assert np.isfinite(result.embedding).all()
    assert abs(float(np.linalg.norm(result.embedding)) - 1.0) < 1e-3
```


## `backend/tests_engine/test_persistence.py`

**7 tests · 21 assertions**

### Purpose of this module, as recorded in it

```text
Durability tests for the biometric template store and audit log.

The central claim under test is the one the product page makes: an enrolled
identity survives a process restart, and an audit hash handed back to a user
can actually be looked up later.
```

| Class | Test | Asserts | What it checks |
|---|---|---|---|
| — | `test_enroll_survives_restart` | 5 | Enroll, drop the process state, reopen -> the template is still there. |
| — | `test_templates_are_encrypted_on_disk` | 1 | The raw float32 bytes must not be readable in the DB file. |
| — | `test_wrong_key_is_detected_not_silently_wrong` | 1 | A mismatched key must raise, never decrypt into a plausible vector. |
| — | `test_tenant_isolation_on_restore` | 3 | Restored templates land in their own tenant shard, not a shared one. |
| — | `test_audit_row_is_queryable_by_hash` | 8 | An audit hash returned to a caller must resolve to a real record. |
| — | `test_audit_survives_restart_and_appends` | 2 | *(no docstring)* |
| — | `test_rejects_newer_schema` | 1 | *(no docstring)* |

### `test_enroll_survives_restart`

**Rationale as recorded in the test**

```text
Enroll, drop the process state, reopen -> the template is still there.
```

```python
def test_enroll_survives_restart(db_path: Path):
    """Enroll, drop the process state, reopen -> the template is still there."""
    cipher = TemplateCipher(base64.b64decode(_key()))
    emb = _vec(1)

    store = BiometricStore(db_path, cipher=cipher)
    store.put_template(
        tenant_id="t1",
        template_id="tpl-1",
        subject_id="alice",
        embedding=emb,
        metadata={"source": "upload"},
        source_sha256="a" * 64,
        model_version="ensemble_v1",
    )
    store.close()

    # --- simulated restart: brand new objects, nothing carried over ---
    store2 = BiometricStore(db_path, cipher=cipher)
    index = GalleryIndex(dimensions=512)
    assert index.size("t1") == 0, "fresh index must start empty"

    loaded = restore_into(store2, index)
    assert loaded == 1
    assert index.size("t1") == 1

    rows = list(store2.iter_templates())
    assert rows[0].subject_id == "alice"
    # round-trip must be bit-exact; a lossy template silently degrades matching
    np.testing.assert_array_equal(rows[0].embedding, emb)
    store2.close()
```

### `test_templates_are_encrypted_on_disk`

**Rationale as recorded in the test**

```text
The raw float32 bytes must not be readable in the DB file.
```

```python
def test_templates_are_encrypted_on_disk(db_path: Path):
    """The raw float32 bytes must not be readable in the DB file."""
    raw_key = base64.b64decode(_key())
    emb = _vec(2)

    store = BiometricStore(db_path, cipher=TemplateCipher(raw_key))
    store.put_template("t1", "tpl-1", "bob", emb, {}, "b" * 64, "m")
    store.close()

    blob = db_path.read_bytes()
    assert emb.tobytes() not in blob, "plaintext embedding found in the database file"
```

### `test_wrong_key_is_detected_not_silently_wrong`

**Rationale as recorded in the test**

```text
A mismatched key must raise, never decrypt into a plausible vector.
```

```python
def test_wrong_key_is_detected_not_silently_wrong(db_path: Path):
    """A mismatched key must raise, never decrypt into a plausible vector."""
    store = BiometricStore(db_path, cipher=TemplateCipher(base64.b64decode(_key())))
    store.put_template("t1", "tpl-1", "carol", _vec(3), {}, "c" * 64, "m")
    store.close()

    other = BiometricStore(db_path, cipher=TemplateCipher(base64.b64decode(_key())))
    with pytest.raises(TemplateDecryptionError):
        list(other.iter_templates())
    other.close()
```

### `test_tenant_isolation_on_restore`

**Rationale as recorded in the test**

```text
Restored templates land in their own tenant shard, not a shared one.
```

```python
def test_tenant_isolation_on_restore(db_path: Path):
    """Restored templates land in their own tenant shard, not a shared one."""
    store = BiometricStore(db_path, cipher=TemplateCipher(None))
    store.put_template("tenant-a", "tpl-a", "alice", _vec(4), {}, "d" * 64, "m")
    store.put_template("tenant-b", "tpl-b", "bob", _vec(5), {}, "e" * 64, "m")
    store.close()

    store2 = BiometricStore(db_path, cipher=TemplateCipher(None))
    index = GalleryIndex(dimensions=512)
    restore_into(store2, index)

    assert index.size("tenant-a") == 1
    assert index.size("tenant-b") == 1
    assert index.size("tenant-c") == 0
    store2.close()
```

### `test_audit_row_is_queryable_by_hash`

**Rationale as recorded in the test**

```text
An audit hash returned to a caller must resolve to a real record.
```

```python
def test_audit_row_is_queryable_by_hash(db_path: Path):
    """An audit hash returned to a caller must resolve to a real record."""
    store = BiometricStore(db_path, cipher=TemplateCipher(None))
    store.write_audit(
        audit_hash="deadbeef",
        operation="verify",
        operator_id="op-7",
        tenant_id="t1",
        decision="match",
        model_version="ensemble_v1",
        score=0.84,
        detail={"threshold": 0.28},
    )
    store.close()

    store2 = BiometricStore(db_path, cipher=TemplateCipher(None))
    rows = store2.get_audit("deadbeef")
    assert len(rows) == 1
    assert rows[0].operation == "verify"
    assert rows[0].operator_id == "op-7"
    assert rows[0].decision == "match"
    assert rows[0].score == pytest.approx(0.84)
    assert rows[0].detail["threshold"] == 0.28
    assert store2.get_audit("nonexistent") == []
    store2.close()
```

### `test_audit_survives_restart_and_appends`

```python
def test_audit_survives_restart_and_appends(db_path: Path):
    store = BiometricStore(db_path, cipher=TemplateCipher(None))
    for i in range(3):
        store.write_audit(f"h{i}", "identify", "op", "t1", "no_match", "m", score=0.1 * i)
    store.close()

    store2 = BiometricStore(db_path, cipher=TemplateCipher(None))
    assert store2.count_audit() == 3
    store2.write_audit("h3", "identify", "op", "t1", "match", "m", score=0.9)
    assert store2.count_audit() == 4
    store2.close()
```

### `test_rejects_newer_schema`

```python
def test_rejects_newer_schema(db_path: Path):
    from nexgen_engine.search.persistence import SchemaVersionError

    store = BiometricStore(db_path, cipher=TemplateCipher(None))
    store._db.execute("UPDATE _meta SET value='999' WHERE key='schema_version'")
    store._db.commit()
    store.close()

    with pytest.raises(SchemaVersionError):
        BiometricStore(db_path, cipher=TemplateCipher(None))
```


## `backend/tests_engine/test_service_durability.py`

**3 tests · 20 assertions**

### Purpose of this module, as recorded in it

```text
End-to-end durability test against the real EngineService.

This is the test the brief asks for in Phase 4 step 4: enroll an identity,
restart the server, confirm /identify still finds it. It exercises the real
recognition pipeline on GPU, so it is slower than the unit tests in
test_persistence.py -- but it is the only test that proves the wiring, not just
the storage layer, actually persists.
```

| Class | Test | Asserts | What it checks |
|---|---|---|---|
| — | `test_enrollment_survives_restart` | 10 | *(no docstring)* |
| — | `test_audit_hash_is_resolvable` | 7 | The hash handed back to a caller must resolve to a stored record. |
| — | `test_identify_writes_audit` | 3 | *(no docstring)* |

### `test_enrollment_survives_restart`

```python
@pytest.mark.slow
def test_enrollment_survives_restart(tmp_path):
    from nexgen_engine.api.service import EngineService

    audit = tmp_path / "audit.jsonl"
    db = tmp_path / "templates.db"
    photo = _face_bytes(0)

    svc = EngineService(audit_path=audit, store_path=db)
    assert svc.restored_count == 0, "a fresh store must start empty"

    res = svc.enroll(photo, "subject-alpha", {"case": "TEST-1"})
    assert res.decision == "enrolled"
    assert svc.store.count_templates() == 1

    # identify against the live index
    before = svc.identify(photo, operator_id="op-1", top_k=5)
    assert before.matches, "probe should match the identity just enrolled"
    assert before.matches[0].identity_id == "subject-alpha"
    top_before = before.matches[0].confidence

    svc.store.close()
    del svc

    # ---- simulated server restart: brand new service, same paths ----
    svc2 = EngineService(audit_path=audit, store_path=db)
    assert svc2.restored_count == 1, "template was not restored from disk"

    after = svc2.identify(photo, operator_id="op-1", top_k=5)
    assert after.matches, "enrollment did NOT survive restart"
    assert after.matches[0].identity_id == "subject-alpha"
    # restored template must be bit-identical, so the score must match exactly
    assert after.matches[0].confidence == pytest.approx(top_before, abs=1e-6)

    svc2.store.close()
```

### `test_audit_hash_is_resolvable`

**Rationale as recorded in the test**

```text
The hash handed back to a caller must resolve to a stored record.
```

```python
@pytest.mark.slow
def test_audit_hash_is_resolvable(tmp_path):
    """The hash handed back to a caller must resolve to a stored record."""
    from nexgen_engine.api.service import EngineService

    svc = EngineService(audit_path=tmp_path / "a.jsonl", store_path=tmp_path / "t.db")
    ref, probe = _face_bytes(1), _face_bytes(2)

    res = svc.verify(ref, probe, operator_id="op-9")
    rows = svc.store.get_audit(res.audit_hash)

    assert len(rows) == 1, "verify() did not write a durable audit row"
    assert rows[0].operation == "verify"
    assert rows[0].operator_id == "op-9"
    assert rows[0].score == pytest.approx(res.score, abs=1e-5)
    assert rows[0].model_version == EngineService.MODEL_VERSION
    assert "ref_sha256" in rows[0].detail
    svc.store.close()
```

### `test_identify_writes_audit`

```python
@pytest.mark.slow
def test_identify_writes_audit(tmp_path):
    from nexgen_engine.api.service import EngineService

    svc = EngineService(audit_path=tmp_path / "a.jsonl", store_path=tmp_path / "t.db")
    svc.enroll(_face_bytes(3), "subject-beta", {})
    res = svc.identify(_face_bytes(3), operator_id="op-5")

    rows = svc.store.get_audit(res.audit_hash)
    assert len(rows) == 1
    assert rows[0].operation == "identify"
    assert rows[0].detail["gallery_size"] >= 1
    svc.store.close()
```

