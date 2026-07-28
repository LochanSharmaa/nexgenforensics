from __future__ import annotations

from fastapi.testclient import TestClient

from imatch_api.db.models import Role

from .conftest import TEST_PASSWORD


class TestLogin:
    def test_login_returns_tokens(self, client: TestClient, tenant_factory, user_factory):
        user_factory(tenant_factory())
        response = client.post(
            "/api/auth/login", json={"email": "investigator@example.com", "password": TEST_PASSWORD}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["access_token"] and body["refresh_token"]
        assert body["user"]["role"] == "investigator"

    def test_wrong_password_is_rejected(self, client: TestClient, tenant_factory, user_factory):
        user_factory(tenant_factory())
        response = client.post(
            "/api/auth/login", json={"email": "investigator@example.com", "password": "wrong-password-1!"}
        )
        assert response.status_code == 401

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


class TestTokens:
    def test_protected_route_requires_a_token(self, client: TestClient):
        assert client.get("/api/cases").status_code == 401

    def test_garbage_token_is_rejected(self, client: TestClient):
        response = client.get("/api/cases", headers={"Authorization": "Bearer not-a-real-token"})
        assert response.status_code == 401

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

    def test_refresh_issues_a_new_access_token(self, client: TestClient, tenant_factory, user_factory):
        user_factory(tenant_factory())
        login = client.post(
            "/api/auth/login", json={"email": "investigator@example.com", "password": TEST_PASSWORD}
        ).json()

        refreshed = client.post("/api/auth/refresh", json={"refresh_token": login["refresh_token"]})
        assert refreshed.status_code == 200
        assert refreshed.json()["access_token"]

    def test_me_returns_the_signed_in_user(self, client: TestClient, tenant_factory, user_factory, auth_headers):
        user_factory(tenant_factory())
        response = client.get("/api/auth/me", headers=auth_headers())
        assert response.status_code == 200
        assert response.json()["email"] == "investigator@example.com"


class TestRoles:
    def test_investigator_cannot_enrol(self, client: TestClient, tenant_factory, user_factory, auth_headers, face_b64):
        """Enrolment decides who the system can find, so it needs supervisor rights."""
        user_factory(tenant_factory())
        response = client.post(
            "/api/subjects",
            headers=auth_headers(),
            json={"display_name": "Test", "image_base64": face_b64[0], "lawful_basis": "test"},
        )
        assert response.status_code == 403

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


class TestApiKeys:
    def test_admin_can_mint_and_use_a_key(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        user_factory(tenant_factory(), email="admin@example.com", role=Role.ADMIN)
        headers = auth_headers("admin@example.com")

        created = client.post("/api/admin/api-keys", headers=headers, json={"name": "integration"})
        assert created.status_code == 201
        plaintext = created.json()["api_key"]

        assert client.get("/api/cases", headers={"X-API-Key": plaintext}).status_code == 200

    def test_revoked_key_stops_working(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        user_factory(tenant_factory(), email="admin@example.com", role=Role.ADMIN)
        headers = auth_headers("admin@example.com")

        created = client.post("/api/admin/api-keys", headers=headers, json={"name": "temp"}).json()
        assert client.get("/api/cases", headers={"X-API-Key": created["api_key"]}).status_code == 200

        client.delete(f"/api/admin/api-keys/{created['id']}", headers=headers)
        assert client.get("/api/cases", headers={"X-API-Key": created["api_key"]}).status_code == 401

    def test_key_plaintext_is_never_returned_again(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        user_factory(tenant_factory(), email="admin@example.com", role=Role.ADMIN)
        headers = auth_headers("admin@example.com")
        client.post("/api/admin/api-keys", headers=headers, json={"name": "listed"})

        listed = client.get("/api/admin/api-keys", headers=headers).json()
        assert listed and all("api_key" not in entry for entry in listed)
