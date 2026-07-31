"""Registration, verification, lockout and password reset.

Codes and reset tokens are read from the mail OUTBOX, not from the database and
not returned by any endpoint. That is deliberate: a test that reaches into
`user.otp_hash` would still pass if the e-mail were never sent, and "the code
reached the user" is the property that actually matters. Reading the outbox
exercises the same path a real mailbox does.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from imatch_api.core.config import get_settings
from imatch_api.db.models import Tenant, User, utcnow
from imatch_api.db.session import build_engine

STRONG = "Redwood-Harbour-2026!"
REG = "/api/auth/register"


@pytest.fixture
def mail_outbox(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "outbox.jsonl"
    monkeypatch.setenv("NEXGEN_MAIL_OUTBOX_PATH", str(path))
    # No RESEND_API_KEY in tests: MailService writes to the outbox instead of
    # making a network call, so the suite never depends on a third party.
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.setenv("NEXGEN_ALLOW_SELF_REGISTRATION", "true")
    get_settings.cache_clear()
    return path


def outbox(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def latest_otp(path: Path) -> str:
    for entry in reversed(outbox(path)):
        found = re.search(r"\b(\d{6})\b", entry["text"])
        if found:
            return found.group(1)
    raise AssertionError(f"no OTP in outbox: {[e['subject'] for e in outbox(path)]}")


def latest_reset_token(path: Path) -> str:
    for entry in reversed(outbox(path)):
        found = re.search(r"token=([A-Za-z0-9_\-]+)", entry["text"])
        if found:
            return found.group(1)
    raise AssertionError("no reset token in outbox")


def seed_tenant(slug: str = "nexgen-demo") -> str:
    engine = build_engine(get_settings())
    with Session(engine) as session:
        tenant = session.exec(select(Tenant).where(Tenant.slug == slug)).first()
        if tenant is None:
            tenant = Tenant(slug=slug, name="Test Org")
            session.add(tenant)
            session.commit()
            session.refresh(tenant)
        return tenant.id


def load_user(email: str) -> User | None:
    engine = build_engine(get_settings())
    with Session(engine) as session:
        return session.exec(select(User).where(User.email == email)).first()


def save_user(user: User) -> None:
    engine = build_engine(get_settings())
    with Session(engine) as session:
        session.add(user)
        session.commit()


def register(client: TestClient, email: str, password: str = STRONG, **kw):
    body = {"full_name": "Test Person", "email": email,
            "password": password, "confirm_password": kw.pop("confirm", password)}
    body.update(kw)
    return client.post(REG, json=body)


# ------------------------------------------------------------- registration --


class TestRegistration:
    def test_registration_is_disabled_by_default(self, anon_client, tmp_path, monkeypatch):
        monkeypatch.setenv("NEXGEN_ALLOW_SELF_REGISTRATION", "false")
        get_settings.cache_clear()
        seed_tenant()
        response = register(anon_client, "nobody@example.com")
        assert response.status_code == 403
        assert "disabled" in response.json()["detail"].lower()

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

    def test_password_is_never_stored_in_plaintext(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "hash@example.com")
        user = load_user("hash@example.com")
        assert STRONG not in user.password_hash
        assert user.password_hash.startswith("$argon2")

    def test_weak_password_is_rejected(self, anon_client, mail_outbox):
        seed_tenant()
        assert register(anon_client, "weak@example.com", password="short").status_code == 422

    def test_mismatched_confirmation_is_rejected(self, anon_client, mail_outbox):
        seed_tenant()
        assert register(anon_client, "mm@example.com", confirm="Different-Pass-99!").status_code == 422

    def test_invalid_email_is_rejected(self, anon_client, mail_outbox):
        seed_tenant()
        assert register(anon_client, "not-an-email").status_code == 422

    def test_duplicate_registration_does_not_reveal_the_account(self, anon_client, mail_outbox):
        """Enumeration guard: the second attempt must be indistinguishable."""
        seed_tenant()
        first = register(anon_client, "dupe@example.com")
        second = register(anon_client, "dupe@example.com")
        assert first.status_code == second.status_code == 201
        assert first.json() == second.json()


# ------------------------------------------------------------ verification --


class TestVerification:
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

    def test_wrong_code_is_rejected_and_counted(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "wrong@example.com")
        response = anon_client.post("/api/auth/verify-email",
                                    json={"email": "wrong@example.com", "otp": "000000"})
        assert response.status_code == 400
        assert load_user("wrong@example.com").otp_attempts == 1

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

    def test_resend_is_capped_per_account(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "flood@example.com")
        settings = get_settings()
        statuses = [
            anon_client.post("/api/auth/resend-otp", json={"email": "flood@example.com"}).status_code
            for _ in range(settings.otp_resend_max + 2)
        ]
        assert 429 in statuses

    def test_resend_for_unknown_address_does_not_reveal_it(self, anon_client, mail_outbox):
        seed_tenant()
        response = anon_client.post("/api/auth/resend-otp", json={"email": "ghost@example.com"})
        assert response.status_code == 200
        assert not outbox(mail_outbox)


# ------------------------------------------------------------------- login --


class TestLoginGating:
    def test_unverified_account_cannot_log_in(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "gated@example.com")
        response = anon_client.post("/api/auth/login",
                                    json={"email": "gated@example.com", "password": STRONG})
        assert response.status_code == 403
        assert response.json()["detail"] == "Please verify your email before logging in."

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


class TestLockout:
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


# ---------------------------------------------------------- password reset --


class TestPasswordReset:
    def test_forgot_password_never_reveals_whether_the_account_exists(
        self, anon_client, mail_outbox
    ):
        seed_tenant()
        register(anon_client, "real@example.com")
        known = anon_client.post("/api/auth/forgot-password", json={"email": "real@example.com"})
        unknown = anon_client.post("/api/auth/forgot-password", json={"email": "ghost@example.com"})
        assert known.status_code == unknown.status_code == 200
        assert known.json() == unknown.json()

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

    def test_reset_token_is_single_use(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "single@example.com")
        anon_client.post("/api/auth/forgot-password", json={"email": "single@example.com"})
        token = latest_reset_token(mail_outbox)
        body = {"token": token, "password": "First-Change-88!",
                "confirm_password": "First-Change-88!"}
        assert anon_client.post("/api/auth/reset-password", json=body).status_code == 200
        assert anon_client.post("/api/auth/reset-password", json=body).status_code == 400

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

    def test_reset_rejects_a_weak_password(self, anon_client, mail_outbox):
        seed_tenant()
        register(anon_client, "weakreset@example.com")
        anon_client.post("/api/auth/forgot-password", json={"email": "weakreset@example.com"})
        token = latest_reset_token(mail_outbox)
        response = anon_client.post("/api/auth/reset-password", json={
            "token": token, "password": "abc", "confirm_password": "abc"})
        assert response.status_code == 422

    def test_unknown_reset_token_is_rejected(self, anon_client, mail_outbox):
        seed_tenant()
        response = anon_client.post("/api/auth/reset-password", json={
            "token": "x" * 40, "password": "Valid-Password-42!",
            "confirm_password": "Valid-Password-42!"})
        assert response.status_code == 400


# --------------------------------------------------------------- sessions --


class TestSessionRevocation:
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


# ------------------------------------------------------------------ audit --


class TestAuditTrail:
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
