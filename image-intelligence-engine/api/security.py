"""Password hashing and JWT issuance.

Single-user local deployments still authenticate. That is deliberate: an auth
path exercised from day one is one that works when multi-user arrives, whereas
auth bolted on later is auth that was never tested against real handlers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
from jose import JWTError, jwt

from shared.config import Settings
from shared.errors import AuthenticationError

# `bcrypt` is used directly rather than through passlib: passlib 1.7.4 (its last
# release, 2020) reads `bcrypt.__about__`, which bcrypt 4.1 removed, so the pair
# raises on every hash. Calling bcrypt directly removes a dependency and the
# version-pin fragility along with it.
BCRYPT_ROUNDS = 12
MAX_PASSWORD_BYTES = 72

MIN_PASSWORD_LENGTH = 12


def hash_password(plain: str) -> str:
    if len(plain) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    encoded = plain.encode("utf-8")
    # bcrypt silently ignores everything past 72 bytes. Rejecting is better than
    # accepting a password whose tail does not participate in authentication.
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes.")
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time comparison via bcrypt. Never raises for bad input.

    A malformed hash in the database is a failed login, not a 500 — the caller
    is on an unauthenticated path and must learn nothing from the difference.
    """
    try:
        encoded = plain.encode("utf-8")[:MAX_PASSWORD_BYTES]
        return bcrypt.checkpw(encoded, hashed.encode("ascii"))
    except (ValueError, TypeError, UnicodeDecodeError):
        return False


def create_access_token(subject: uuid.UUID, settings: Settings, **claims: Any) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.access_token_ttl_minutes)).timestamp()),
        "iss": "iie",
        **claims,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    try:
        return jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
            issuer="iie",
        )
    except JWTError as exc:
        raise AuthenticationError(f"Invalid or expired token: {exc}") from exc


__all__ = [
    "ImatchIdentity",
    "create_access_token",
    "decode_imatch_token",
    "decode_access_token",
    "hash_password",
    "verify_password",
]


# --------------------------------------------------- federated iMATCH tokens --


@dataclass(frozen=True)
class ImatchIdentity:
    """An investigator authenticated by the NexGen iMATCH workspace."""

    subject_id: str
    tenant_id: str
    role: str


def decode_imatch_token(token: str, settings: Settings) -> ImatchIdentity:
    """Verify a token issued by NexGen iMATCH.

    The workspace already authenticates its investigators. Requiring a second
    sign-in to reach provenance would add friction without adding security, so
    IIE verifies iMATCH's own token instead of minting a parallel credential.

    Every claim iMATCH sets is checked, not just the signature:

    * ``iss`` must be the configured issuer, so a token signed with the same
      secret by some other service is not accepted here.
    * ``type`` must be ``access`` — without this a long-lived *refresh* token
      would be usable as an access token, silently extending its privileges.
    * ``tid`` must be present. iMATCH is multi-tenant and a token with no
      tenant cannot be attributed to anyone.
    """
    if not settings.imatch_federation_enabled:
        raise AuthenticationError("iMATCH federation is not configured.")

    try:
        claims = jwt.decode(
            token,
            settings.imatch_jwt_secret,
            algorithms=[settings.imatch_jwt_algorithm],
            issuer=settings.imatch_issuer,
            options={"require": ["exp", "iat", "sub", "iss"]},
        )
    except JWTError as exc:
        raise AuthenticationError(f"Not a valid iMATCH token: {exc}") from exc

    if claims.get("type") != "access":
        raise AuthenticationError(
            f"Expected an iMATCH access token, got {claims.get('type')!r}."
        )

    tenant_id = str(claims.get("tid") or "")
    if not tenant_id:
        raise AuthenticationError("iMATCH token carries no tenant claim.")

    return ImatchIdentity(
        subject_id=str(claims["sub"]),
        tenant_id=tenant_id,
        role=str(claims.get("role", "investigator")),
    )
