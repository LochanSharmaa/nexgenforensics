from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

logger = logging.getLogger(__name__)

API_KEY_PREFIX = "imk"
_PREFIX_LENGTH = 12

# Argon2id is the current password-hashing recommendation. Defaults here follow
# the OWASP guidance for interactive logins: enough work to make offline
# cracking expensive, fast enough not to be its own denial-of-service vector.
_hasher = PasswordHasher(time_cost=3, memory_cost=64 * 1024, parallelism=4)


class TokenError(Exception):
    """Raised when a token is missing, malformed, expired, or not trustworthy."""


@dataclass(frozen=True)
class TokenPayload:
    subject_id: str
    tenant_id: str
    role: str
    token_type: str
    expires_at: datetime


# ---------------------------------------------------------------- passwords --


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    try:
        return _hasher.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


def validate_password_strength(password: str) -> None:
    """Reject passwords that would not survive an offline attack.

    Deliberately a length-and-variety floor rather than a composition maze: long
    passphrases beat short complex strings, and rules that force symbol
    substitution mostly produce predictable patterns.
    """
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters.")
    if len(password) > 256:
        raise ValueError("Password must be at most 256 characters.")
    classes = (
        any(char.islower() for char in password),
        any(char.isupper() for char in password),
        any(char.isdigit() for char in password),
        any(not char.isalnum() for char in password),
    )
    if sum(classes) < 3:
        raise ValueError(
            "Password must combine at least three of: lowercase, uppercase, digits, symbols."
        )


# ------------------------------------------------------------------- tokens --


def create_token(
    *,
    subject_id: str,
    tenant_id: str,
    role: str,
    secret: str,
    algorithm: str = "HS256",
    token_type: str = "access",
    expires_in: timedelta = timedelta(minutes=60),
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject_id,
        "tid": tenant_id,
        "role": role,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_in).timestamp()),
        "jti": secrets.token_urlsafe(16),
        "iss": "nexgen-imatch",
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_token(token: str, secret: str, algorithm: str = "HS256", expected_type: str = "access") -> TokenPayload:
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[algorithm],
            issuer="nexgen-imatch",
            options={"require": ["exp", "iat", "sub", "iss"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Token is invalid.") from exc

    if claims.get("type") != expected_type:
        # Without this check a refresh token would be accepted as an access
        # token, silently extending a long-lived credential's privileges.
        raise TokenError(f"Expected a {expected_type} token, got {claims.get('type')!r}.")

    tenant_id = claims.get("tid")
    if not tenant_id:
        raise TokenError("Token carries no tenant claim.")

    return TokenPayload(
        subject_id=str(claims["sub"]),
        tenant_id=str(tenant_id),
        role=str(claims.get("role", "investigator")),
        token_type=str(claims.get("type")),
        expires_at=datetime.fromtimestamp(int(claims["exp"]), tz=timezone.utc),
    )


# ----------------------------------------------------------------- api keys --


def generate_api_key() -> tuple[str, str, str]:
    """Create an API key. Returns ``(plaintext, prefix, hash)``.

    The plaintext is returned once, to be shown to the operator and never
    persisted. Lookup happens by the non-secret prefix; the remainder is
    verified against a SHA-256 hash in constant time.

    SHA-256 rather than Argon2 here is deliberate: an API key is 32 bytes of
    CSPRNG output, so there is no low-entropy secret to slow-hash, and key
    verification runs on every machine request.
    """
    secret = secrets.token_urlsafe(32)
    plaintext = f"{API_KEY_PREFIX}_{secret}"
    return plaintext, plaintext[:_PREFIX_LENGTH], hash_api_key(plaintext)


def hash_api_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def verify_api_key(plaintext: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_api_key(plaintext), stored_hash)


def api_key_prefix(plaintext: str) -> str:
    return plaintext[:_PREFIX_LENGTH]


__all__ = [
    "API_KEY_PREFIX",
    "TokenError",
    "TokenPayload",
    "api_key_prefix",
    "create_token",
    "decode_token",
    "generate_api_key",
    "hash_api_key",
    "hash_password",
    "needs_rehash",
    "validate_password_strength",
    "verify_api_key",
    "verify_password",
]
