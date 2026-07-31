"""CSRF protection for cookie-borne requests.

WHAT THIS DOES AND DOES NOT PROTECT
-----------------------------------
CSRF exists because a browser attaches cookies to cross-site requests
automatically. It does NOT attach `Authorization` or `X-API-Key`, because a
cross-origin page cannot set custom headers on a form post or an <img> and the
CORS preflight blocks it on fetch. So a request carrying a header credential is
already immune, and requiring a token there would be ceremony that protects
nothing while breaking every existing API client.

Enforcement is therefore scoped precisely:

  skipped   safe methods (GET/HEAD/OPTIONS/TRACE) -- no state change
  skipped   requests presenting Authorization or X-API-Key
  enforced  every other state-changing request

Today that last group is the unauthenticated auth endpoints -- login, register,
verify-email, resend-otp, forgot-password, reset-password. Login CSRF is a real
attack and it matters here specifically: an attacker who can force a victim's
browser to log into an ATTACKER-controlled account causes the victim's
subsequent searches to be recorded, in the audit chain, against the wrong
person. On a system whose audit trail is meant to be evidence, that is worse
than the usual nuisance.

It also means the protection is already correct for the HTTPOnly session
cookies, should cookie-based authentication ever be switched on.

TOKENS ARE SIGNED, NOT STORED
-----------------------------
A random nonce plus an HMAC over (nonce, issued-at) keyed on the JWT secret.
Nothing is persisted, so this works unchanged across multiple workers and
survives a restart -- a server-side token store would need Redis to do the same.
The double-submit check (cookie value must equal header value) is what ties the
token to this browser; the signature is what stops an attacker minting one.
"""

from __future__ import annotations

import hmac
import secrets
import time
from hashlib import sha256

CSRF_COOKIE = "nx_csrf"
CSRF_HEADER = "X-CSRF-Token"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
DEFAULT_MAX_AGE = 60 * 60 * 12


def _sign(nonce: str, issued: str, secret: str) -> str:
    return hmac.new(secret.encode(), f"{nonce}.{issued}".encode(), sha256).hexdigest()


def issue_csrf_token(secret: str) -> str:
    nonce = secrets.token_urlsafe(24)
    issued = str(int(time.time()))
    return f"{nonce}.{issued}.{_sign(nonce, issued, secret)}"


def validate_csrf_token(token: str | None, secret: str, max_age: int = DEFAULT_MAX_AGE) -> bool:
    if not token:
        return False
    parts = token.split(".")
    if len(parts) != 3:
        return False
    nonce, issued, signature = parts

    # compare_digest, not ==, so a forged token cannot be refined byte by byte
    # from response timing.
    if not hmac.compare_digest(signature, _sign(nonce, issued, secret)):
        return False

    try:
        age = int(time.time()) - int(issued)
    except ValueError:
        return False
    # Reject future-dated tokens as well as expired ones: a clock skew large
    # enough to matter is itself a sign the value was not minted here.
    return -300 <= age <= max_age


def request_is_exempt(method: str, has_auth_header: bool) -> bool:
    return method.upper() in SAFE_METHODS or has_auth_header


def tokens_match(cookie_value: str | None, header_value: str | None) -> bool:
    if not cookie_value or not header_value:
        return False
    return hmac.compare_digest(cookie_value, header_value)
