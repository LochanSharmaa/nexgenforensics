"""Single-user mode: credential-less requests act as the local owner.

The rest of the suite runs with NEXGEN_SINGLE_USER=false (conftest.py) and
asserts the multi-user contract. These tests flip the flag on per-case and
assert the individual-use contract: no sign-in required, but presented
credentials are still verified, and everything is still attributed to a real
user row.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from imatch_api.core.config import get_settings
from imatch_api.db.models import Role
from tests.conftest import prime_csrf


@pytest.fixture
def single_user_env(monkeypatch: pytest.MonkeyPatch):
    """Flip the flag on for one test.

    The autouse environment fixture has already bound a fresh database and
    cleared the cached settings and owner principal; only the flag (and a
    settings reload) is needed before an app reads Settings.
    """
    monkeypatch.setenv("NEXGEN_SINGLE_USER", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def single_user_client(single_user_env) -> TestClient:
    from imatch_api.main import create_app

    with TestClient(create_app()) as client:
        yield prime_csrf(client)


class TestSingleUserMode:
    def test_me_works_without_credentials_and_provisions_owner(self, single_user_client):
        response = single_user_client.get("/api/auth/me")
        assert response.status_code == 200, response.text
        profile = response.json()
        # Empty database: the owner is created on first use, as an admin, so
        # every workspace feature (including enrolment) is available.
        assert profile["email"] == "owner@local"
        assert profile["role"] == "admin"

    def test_binds_to_earliest_active_admin_when_accounts_exist(
        self, single_user_env, tenant_factory, user_factory
    ):
        from imatch_api.main import create_app

        tenant = tenant_factory()
        user_factory(tenant, email="older-admin@example.com", role=Role.ADMIN)
        user_factory(tenant, email="newer-admin@example.com", role=Role.ADMIN)

        with TestClient(create_app()) as client:
            response = client.get("/api/auth/me")
            assert response.status_code == 200, response.text
            assert response.json()["email"] == "older-admin@example.com"

    def test_owner_email_pins_the_acting_account(
        self, single_user_env, monkeypatch: pytest.MonkeyPatch, tenant_factory, user_factory
    ):
        from imatch_api.main import create_app

        tenant = tenant_factory()
        user_factory(tenant, email="older-admin@example.com", role=Role.ADMIN)
        user_factory(tenant, email="pinned@example.com", role=Role.ADMIN)

        monkeypatch.setenv("NEXGEN_OWNER_EMAIL", "pinned@example.com")
        get_settings.cache_clear()
        with TestClient(create_app()) as client:
            response = client.get("/api/auth/me")
            assert response.status_code == 200, response.text
            assert response.json()["email"] == "pinned@example.com"

    def test_invalid_bearer_is_still_rejected(self, single_user_client):
        # A presented credential must be verified, not shrugged off into the
        # owner identity: a client sending a dead token needs to hear so.
        response = single_user_client.get(
            "/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"}
        )
        assert response.status_code == 401

    def test_state_changing_request_works_without_credentials(self, single_user_client):
        response = single_user_client.post(
            "/api/cases",
            json={
                "reference": "SU-001",
                "title": "Single-user smoke case",
                "lawful_basis": "Owner's own investigation",
            },
        )
        assert response.status_code in (200, 201), response.text

        listed = single_user_client.get("/api/cases")
        assert listed.status_code == 200
        references = [case["reference"] for case in listed.json()]
        assert "SU-001" in references
