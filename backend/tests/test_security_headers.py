"""Security response headers and CSRF enforcement.

These assert the guard REFUSES, not merely that it can be satisfied. A CSRF
layer that every fixture quietly primes would pass a suite while protecting
nothing, so the refusals are tested here directly with an unprimed client.
"""

from __future__ import annotations

import time

import pytest

from imatch_api.core.csrf import (
    CSRF_COOKIE,
    CSRF_HEADER,
    issue_csrf_token,
    validate_csrf_token,
)

SECRET = "unit-test-secret-value-not-used-anywhere-else"


class TestSecurityHeaders:
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

    def test_api_responses_get_the_strict_csp(self, raw_client):
        csp = raw_client.get("/api/health").headers["Content-Security-Policy"]
        # A JSON API should never be a source of executable or embeddable
        # content, so the policy denies everything by default.
        assert "default-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "base-uri 'none'" in csp

    def test_docs_get_a_relaxed_csp_so_swagger_still_renders(self, raw_client):
        response = raw_client.get("/docs")
        if response.status_code == 404:
            pytest.skip("docs disabled in this configuration")
        csp = response.headers["Content-Security-Policy"]
        # Swagger UI is CDN-hosted and uses inline styles; the strict policy
        # would leave a blank interactive page rather than an obvious error.
        assert "cdn.jsdelivr.net" in csp
        assert "frame-ancestors 'none'" in csp

    def test_errors_also_carry_the_headers(self, raw_client):
        response = raw_client.get("/api/imatch/searches")
        assert response.status_code in (401, 403)
        assert response.headers["X-Content-Type-Options"] == "nosniff"


class TestCsrfTokens:
    def test_a_fresh_token_validates(self):
        assert validate_csrf_token(issue_csrf_token(SECRET), SECRET)

    def test_a_token_from_another_secret_is_rejected(self):
        assert not validate_csrf_token(issue_csrf_token("different-secret"), SECRET)

    def test_a_tampered_token_is_rejected(self):
        token = issue_csrf_token(SECRET)
        nonce, issued, signature = token.split(".")
        assert not validate_csrf_token(f"{nonce}x.{issued}.{signature}", SECRET)
        assert not validate_csrf_token(f"{nonce}.{issued}.{signature[:-1]}0", SECRET)

    def test_an_expired_token_is_rejected(self):
        token = issue_csrf_token(SECRET)
        assert not validate_csrf_token(token, SECRET, max_age=-1)

    def test_a_future_dated_token_is_rejected(self):
        nonce = "abc"
        issued = str(int(time.time()) + 4000)
        from hashlib import sha256
        import hmac as _hmac
        signature = _hmac.new(SECRET.encode(), f"{nonce}.{issued}".encode(), sha256).hexdigest()
        assert not validate_csrf_token(f"{nonce}.{issued}.{signature}", SECRET)

    def test_malformed_tokens_are_rejected(self):
        for bad in (None, "", "nodots", "only.two", "a.b.c.d"):
            assert not validate_csrf_token(bad, SECRET)


class TestCsrfEnforcement:
    def test_state_change_without_a_token_is_refused(self, raw_client):
        response = raw_client.post("/api/auth/login",
                                   json={"email": "a@example.com", "password": "whatever"})
        assert response.status_code == 403
        assert "CSRF" in response.json()["detail"]

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

    def test_cookie_without_the_header_is_refused(self, raw_client):
        raw_client.get("/api/auth/csrf")  # sets the cookie only
        response = raw_client.post("/api/auth/login",
                                   json={"email": "a@example.com", "password": "x"})
        assert response.status_code == 403

    def test_mismatched_pair_is_refused(self, raw_client):
        from imatch_api.core.config import get_settings

        raw_client.get("/api/auth/csrf")
        other = issue_csrf_token(get_settings().resolved_jwt_secret())
        response = raw_client.post("/api/auth/login",
                                   json={"email": "a@example.com", "password": "x"},
                                   headers={CSRF_HEADER: other})
        assert response.status_code == 403

    def test_a_matching_pair_passes_the_guard(self, raw_client):
        token = raw_client.get("/api/auth/csrf").json()["csrf_token"]
        response = raw_client.post("/api/auth/login",
                                   json={"email": "nobody@example.com", "password": "x"},
                                   headers={CSRF_HEADER: token})
        # Past the guard; the credentials themselves are simply wrong.
        assert response.status_code == 401

    def test_safe_methods_need_no_token(self, raw_client):
        assert raw_client.get("/api/health").status_code == 200

    def test_bearer_authenticated_requests_are_exempt(self, raw_client):
        """A cross-origin page cannot set Authorization, so those requests are
        not forgeable and must not be made to carry a token -- otherwise every
        existing API client breaks for no security gain."""
        response = raw_client.post("/api/auth/logout",
                                   headers={"Authorization": "Bearer not-a-real-token"})
        # 401 from authentication, NOT 403 from the CSRF guard.
        assert response.status_code == 401

    def test_api_key_requests_are_exempt(self, raw_client):
        response = raw_client.post("/api/auth/logout", headers={"X-API-Key": "nope"})
        assert response.status_code == 401

    def test_csrf_cookie_is_readable_by_script(self, raw_client):
        """Unlike the session cookies, this one MUST NOT be HTTPOnly -- the page
        has to read it to echo it back."""
        response = raw_client.get("/api/auth/csrf")
        cookie_header = response.headers.get("set-cookie", "")
        assert CSRF_COOKIE in cookie_header
        assert "httponly" not in cookie_header.lower()
