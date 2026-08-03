"""Federated sign-in from the NexGen iMATCH workspace.

An investigator already authenticated by the workspace must not be asked to
sign in a second time. IIE verifies iMATCH's own token instead of minting a
parallel credential — but it verifies every claim, not merely the signature.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from jose import jwt

from api.dependencies import get_clock, get_db_session, get_object_store, get_session_factory
from api.main import create_app
from shared.config import Settings

IMATCH_SECRET = "shared-with-nexgen-jwt-secret-32-chars-min"
ISSUER = "nexgen-imatch"


def imatch_token(
    *,
    subject: str = "user-abc-123",
    tenant: str = "nexgen-demo",
    role: str = "investigator",
    token_type: str = "access",  # noqa: S107 - a claim value, not a credential
    issuer: str = ISSUER,
    secret: str = IMATCH_SECRET,
    expires_in_minutes: int = 60,
) -> str:
    """Mint a token exactly as `imatch_api.core.security.create_token` does."""
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": subject,
            "tid": tenant,
            "role": role,
            "type": token_type,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=expires_in_minutes)).timestamp()),
            "jti": secrets.token_urlsafe(8),
            "iss": issuer,
        },
        secret,
        algorithm="HS256",
    )


@pytest.fixture
def federated_settings() -> Settings:
    return Settings(
        environment="test",
        database_url="sqlite+aiosqlite:///:memory:",
        secret_key="iie-own-secret-key-at-least-32-characters",
        imatch_jwt_secret=IMATCH_SECRET,
        archive_lookup_enabled=False,
        log_format="console",
        log_level="WARNING",
    )


@pytest.fixture
async def federated_client(federated_settings, session_factory, clock, object_store):
    """A client with no IIE account and no IIE login — only an iMATCH token."""
    app = create_app(federated_settings)

    async def _session_override():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    from shared.config import get_settings as real_get_settings

    app.dependency_overrides[get_db_session] = _session_override
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_clock] = lambda: clock
    app.dependency_overrides[get_object_store] = lambda: object_store
    app.dependency_overrides[real_get_settings] = lambda: federated_settings

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        yield http


CASE = {
    "case_id": "FED-2026-1",
    "title": "Federated access",
    "lawful_basis": "Workspace engagement",
}


# ------------------------------------------------------------ the happy path --


async def test_workspace_token_is_accepted_without_a_second_login(federated_client):
    """The whole point: no separate IIE sign-in."""
    response = await federated_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {imatch_token()}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["email"] == "user-abc-123@imatch.federated"


async def test_federated_investigator_can_open_a_case(federated_client):
    response = await federated_client.post(
        "/api/v1/investigations",
        json=CASE,
        headers={"Authorization": f"Bearer {imatch_token()}"},
    )
    assert response.status_code == 201, response.text
    assert response.json()["case_id"] == "FED-2026-1"


async def test_the_same_subject_reuses_one_account(federated_client):
    """Two requests must not provision two accounts, or a case opened on the
    first would be invisible on the second."""
    headers = {"Authorization": f"Bearer {imatch_token()}"}
    await federated_client.post("/api/v1/investigations", json=CASE, headers=headers)

    # A fresh token for the same subject — different jti, different iat.
    second = {"Authorization": f"Bearer {imatch_token()}"}
    listing = await federated_client.get("/api/v1/investigations", headers=second)
    assert listing.status_code == 200
    assert [c["case_id"] for c in listing.json()] == ["FED-2026-1"]


async def test_different_subjects_do_not_share_cases(federated_client):
    """Ownership scoping must survive federation."""
    await federated_client.post(
        "/api/v1/investigations",
        json=CASE,
        headers={"Authorization": f"Bearer {imatch_token(subject='user-one')}"},
    )
    other = await federated_client.get(
        "/api/v1/investigations",
        headers={"Authorization": f"Bearer {imatch_token(subject='user-two')}"},
    )
    assert other.json() == []


async def test_actions_are_audited_against_the_federated_actor(federated_client):
    """A federated session must never produce anonymous evidence."""
    headers = {"Authorization": f"Bearer {imatch_token()}"}
    await federated_client.post("/api/v1/investigations", json=CASE, headers=headers)
    entries = (await federated_client.get("/api/v1/audit", headers=headers)).json()
    created = [e for e in entries if e["action"] == "investigation.create"]
    assert created
    assert "user-abc-123@imatch.federated" in created[-1]["actor_label"]


# ------------------------------------------------------------- claim checking --


async def test_a_refresh_token_is_not_an_access_token(federated_client):
    """Without this check a long-lived refresh credential would silently gain
    access-token privileges."""
    response = await federated_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {imatch_token(token_type='refresh')}"},
    )
    assert response.status_code == 401


async def test_a_token_from_another_issuer_is_rejected(federated_client):
    """Same secret, different service — still not ours to trust."""
    response = await federated_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {imatch_token(issuer='some-other-app')}"},
    )
    assert response.status_code == 401


async def test_a_token_without_a_tenant_is_rejected(federated_client):
    token = jwt.encode(
        {
            "sub": "u1",
            "role": "investigator",
            "type": "access",
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(minutes=30)).timestamp()),
            "iss": ISSUER,
        },
        IMATCH_SECRET,
        algorithm="HS256",
    )
    response = await federated_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401


async def test_a_wrongly_signed_token_is_rejected(federated_client):
    response = await federated_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {imatch_token(secret='not-the-shared-secret-at-all')}"},
    )
    assert response.status_code == 401


async def test_an_expired_token_is_rejected(federated_client):
    response = await federated_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {imatch_token(expires_in_minutes=-5)}"},
    )
    assert response.status_code == 401


# ------------------------------------------------------------- opt-in only ----


async def test_federation_is_off_unless_configured(client):
    """`client` uses the default settings, where no iMATCH secret is set.

    Federation is a trust relationship, so it must be declared rather than
    assumed — an unconfigured deployment must not accept foreign tokens.
    """
    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {imatch_token()}"},
    )
    assert response.status_code == 401


async def test_iie_native_tokens_still_work_when_federation_is_on(
    federated_client, user, federated_settings
):
    """Enabling federation must not break IIE's own login."""
    response = await federated_client.post(
        "/api/v1/auth/token",
        json={"email": user.email, "password": "investigator-pass-123"},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]

    me = await federated_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert me.status_code == 200
    assert me.json()["email"] == user.email
