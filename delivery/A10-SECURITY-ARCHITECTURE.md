# A10 — Security Architecture and Controls

**Generated:** 2026-07-31 20:32 UTC · **Repository state:** `da66fad0d7f1`

Implemented controls, the reasoning behind each, and the threats this system does not defend against.

---

## Implemented controls

| Control | Implementation | Notes |
|---|---|---|
| Password hashing | Argon2id | Rehashed transparently when cost parameters change |
| Password strength | 12+ characters, 3 of 4 character classes | Length-and-variety floor, not a composition maze |
| Session tokens | JWT access + refresh, typed claims | Refresh rotated on every use; replay of a used token fails |
| Session revocation | Refresh hash stored; cleared on logout and on password reset | `session_epoch` bumped on reset so already-issued access tokens can be rejected |
| E-mail verification | 6-digit OTP, SHA-256 hashed, 10-minute expiry, 5-attempt cap | SHA-256 rather than Argon2 deliberately — see below |
| Account lockout | 5 failed logins, 15-minute lock | Checked BEFORE the credential verdict |
| Rate limiting | Sliding window, per-principal and per-IP per-flow | Single-process; see limitations |
| Template encryption | AES-256-GCM at rest | Authenticated encryption; tampering is detected, not just undetected |
| Transport | HTTPS required in production, HSTS | `Strict-Transport-Security: max-age=63072000` |
| CSRF | Signed double-submit | Scoped to cookie-borne state changes; header-credentialed requests exempt |
| Security headers | CSP, COOP, CORP, Permissions-Policy, X-Frame-Options, nosniff | CSP is `default-src 'none'` for API responses |
| Access control | Role hierarchy, enforced server-side on every request | UI gating is usability, not security |
| Audit trail | Hash-chained, append-only | Tampering with an earlier record invalidates the chain |
| Model integrity | SHA-256 published for every weight file | A4 Part I |

---

## Reasoning behind the non-obvious choices

### OTP is hashed with SHA-256, not Argon2

Unlike a password, a 6-digit code has only 10^6 of entropy. No key-derivation
function saves it from an offline attack against a stolen database — a million
guesses succeeds regardless of cost parameters. What protects it is the
**10-minute expiry** and the **5-attempt cap**, both enforced online. Spending
Argon2 cost on the hash would slow a hot path while changing nothing about the
actual threat.

### CSRF exempts header-credentialed requests

A browser attaches cookies to cross-site requests automatically; it does not
attach `Authorization` or `X-API-Key`, and CORS preflight blocks a cross-origin
page from setting them. A request carrying a header credential is therefore
already immune, and demanding a token would break every API client while
protecting nothing.

What the guard does protect is **login CSRF** — an attacker forcing a victim's
browser into an account the attacker controls, so the victim's subsequent
searches are recorded in the audit chain against the wrong person. On a system
whose audit trail is intended as evidence, that is more serious than the usual
framing of CSRF suggests.

### Enumeration is closed on every unauthenticated endpoint

`register`, `resend-otp` and `forgot-password` return identical responses
whether or not an address exists, and the unverified-email refusal on login is
raised only **after** the password check. An attacker must not be able to use a
forensic system's auth endpoints to discover who holds an account on it.

### Login timing is flattened

A login for a non-existent account still performs an Argon2 verification against
a dummy hash, so response time does not distinguish "no such user" from "wrong
password".

---

## Threats NOT defended against

Stated so that no reader assumes coverage that does not exist.

| Threat | Status |
|---|---|
| **Presentation attack / spoofing** | **Not defended.** The liveness figure is a heuristic reported with `certified: false`. It is not a trained PAD classifier and the `presentation_attack` module is not wired into the service. |
| Distributed rate-limit evasion | Rate limiting is per-process. Behind multiple workers the effective limit multiplies by worker count, and it resets on restart. A shared store or edge limiter is required for a real defence. |
| Adversarial perturbation of input images | Not evaluated. No robustness measurement exists. |
| Model extraction via query volume | Not defended beyond ordinary rate limiting. |
| Insider misuse by an authorised user | Detectable after the fact through the audit chain; not prevented. |
| Compromise of the host | Out of scope. Template encryption protects data at rest, not a running process holding keys. |

---

## Audit chain

Every biometric operation, authentication event and administrative action is
recorded with timestamp, actor, IP address, user agent, outcome and the stated
lawful basis. Records are hash-chained: each entry incorporates the digest of
its predecessor, so altering or removing an earlier record invalidates every
record after it. Verification is available to administrators through
`/api/audit/verify`.

---

## Implementation

### `backend/imatch_api/core/security.py`

```python
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


# --------------------------------------------------------- one-time codes --


def generate_otp(length: int = 6) -> str:
    """A numeric one-time code from a cryptographically secure source.

    `secrets.randbelow` rather than `random`: the stdlib Mersenne Twister is
    predictable from prior outputs, which for an account-recovery code means a
    remote attacker can compute the next one.

    Leading zeros are preserved by zero-padding, so every code is exactly
    `length` digits and the whole space is reachable -- trimming them would
    quietly shrink a 6-digit space and bias it away from low values.
    """
    if length < 4:
        raise ValueError("OTP length must be at least 4 digits.")
    upper = 10 ** length
    return str(secrets.randbelow(upper)).zfill(length)


def hash_otp(otp: str) -> str:
    """Hash a one-time code for storage.

    SHA-256 rather than Argon2, unlike passwords. That is a deliberate
    difference: this value is high-entropy only in combination with a short
    expiry and a hard attempt cap, and it is verified on a hot path. The thing
    protecting a 6-digit code is `otp_max_attempts` and `otp_ttl_minutes`, not
    the cost of the hash -- a million-guess offline attack on 10^6 codes
    succeeds regardless of the KDF, so the defence has to be online rate
    limiting.
    """
    return hashlib.sha256(otp.strip().encode("utf-8")).hexdigest()


def verify_otp(otp: str, stored_hash: str | None) -> bool:
    """Constant-time comparison, so a wrong code leaks nothing by timing."""
    if not stored_hash:
        return False
    return hmac.compare_digest(hash_otp(otp), stored_hash)


def generate_reset_token() -> tuple[str, str]:
    """Return (plaintext, hash) for a password-reset token.

    256 bits from `secrets.token_urlsafe`, so unlike the 6-digit OTP this one
    genuinely cannot be guessed and can safely travel in a URL. Only the hash
    is stored; the plaintext exists in the e-mail and nowhere else.
    """
    plaintext = secrets.token_urlsafe(32)
    return plaintext, hash_reset_token(plaintext)


def hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.strip().encode("utf-8")).hexdigest()


def verify_reset_token(token: str, stored_hash: str | None) -> bool:
    if not stored_hash:
        return False
    return hmac.compare_digest(hash_reset_token(token), stored_hash)


def hash_refresh_token(token: str) -> str:
    """Refresh tokens are stored hashed so a database read cannot resume a
    session. The token itself is already a signed JWT with high entropy, so a
    plain digest is sufficient and keeps rotation cheap."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


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

```

### `backend/imatch_api/core/csrf.py`

```python
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

```

### `backend/nexgen_engine/security/template_encryption.py`

```python
from __future__ import annotations

import base64
import os
from dataclasses import dataclass

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

TEMPLATE_KEY_BYTES = 32
_NONCE_BYTES = 12


class TemplateDecryptionError(RuntimeError):
    """Raised when a stored template cannot be decrypted or has been tampered with."""


@dataclass(frozen=True)
class EncryptedTemplate:
    """An AES-256-GCM sealed biometric template.

    ``tenant_id`` is bound in as additional authenticated data, so a ciphertext
    moved between tenant rows in the database fails to decrypt instead of
    silently matching under the wrong tenant.
    """

    nonce: str
    ciphertext: str
    dimensions: int
    tenant_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "nonce": self.nonce,
            "ciphertext": self.ciphertext,
            "dimensions": self.dimensions,
            "tenant_id": self.tenant_id,
        }


class TemplateEncryptor:
    """Encrypts biometric templates at rest with a single master key.

    Biometrics are irrevocable: a leaked template cannot be reissued like a
    password. Templates are therefore never stored in the clear.

    The master key is supplied once (``NEXGEN_TEMPLATE_KEY``, 32 random bytes,
    base64) and a per-tenant subkey is derived from it with HKDF. Deriving with
    HKDF rather than running PBKDF2 per record matters: the previous version ran
    600k PBKDF2 iterations on every encrypt AND every decrypt, which would add
    roughly a second of CPU to each template touched and make a gallery load of
    any size unusable. PBKDF2 is still used, but only on the ``from_passphrase``
    path where the input is genuinely low-entropy.
    """

    def __init__(self, master_key: bytes) -> None:
        if len(master_key) != TEMPLATE_KEY_BYTES:
            raise ValueError(f"Master key must be exactly {TEMPLATE_KEY_BYTES} bytes, got {len(master_key)}.")
        self._master_key = master_key
        self._subkeys: dict[str, bytes] = {}

    # ------------------------------------------------------------ factories --

    @classmethod
    def from_base64(cls, encoded: str) -> "TemplateEncryptor":
        try:
            return cls(base64.b64decode(encoded, validate=True))
        except Exception as exc:
            raise ValueError(
                "NEXGEN_TEMPLATE_KEY must be 32 random bytes, base64-encoded. Generate with: "
                'python -c "import base64,os; print(base64.b64encode(os.urandom(32)).decode())"'
            ) from exc

    @classmethod
    def from_passphrase(cls, passphrase: str, salt: bytes, iterations: int = 600_000) -> "TemplateEncryptor":
        """Derive a master key from a human-chosen passphrase.

        Only for operators who cannot manage a random key. The salt must be
        stored and reused, or every previously sealed template becomes
        unreadable.
        """
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=TEMPLATE_KEY_BYTES, salt=salt, iterations=iterations)
        return cls(kdf.derive(passphrase.encode("utf-8")))

    @classmethod
    def generate(cls) -> "TemplateEncryptor":
        """Ephemeral key, for tests only. Templates do not survive the process."""
        return cls(os.urandom(TEMPLATE_KEY_BYTES))

    # -------------------------------------------------------------- crypto ---

    def encrypt(self, embedding: np.ndarray, tenant_id: str) -> EncryptedTemplate:
        vector = np.ascontiguousarray(np.asarray(embedding, dtype=np.float32))
        if vector.ndim != 1:
            raise ValueError(f"Expected a 1-D template, got shape {vector.shape}.")

        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = AESGCM(self._subkey(tenant_id)).encrypt(nonce, vector.tobytes(), tenant_id.encode("utf-8"))
        return EncryptedTemplate(
            nonce=base64.b64encode(nonce).decode("ascii"),
            ciphertext=base64.b64encode(ciphertext).decode("ascii"),
            dimensions=int(vector.shape[0]),
            tenant_id=tenant_id,
        )

    def decrypt(self, encrypted: EncryptedTemplate) -> np.ndarray:
        try:
            payload = AESGCM(self._subkey(encrypted.tenant_id)).decrypt(
                base64.b64decode(encrypted.nonce),
                base64.b64decode(encrypted.ciphertext),
                encrypted.tenant_id.encode("utf-8"),
            )
        except (InvalidTag, ValueError) as exc:
            raise TemplateDecryptionError(
                "Template failed authenticated decryption. The master key is wrong, or the "
                "stored record was altered or moved between tenants."
            ) from exc

        vector = np.frombuffer(payload, dtype=np.float32).copy()
        if vector.shape[0] != encrypted.dimensions:
            raise TemplateDecryptionError(
                f"Template length mismatch: expected {encrypted.dimensions}, decoded {vector.shape[0]}."
            )
        return vector

    def _subkey(self, tenant_id: str) -> bytes:
        """Per-tenant key derived from the master key.

        Compromising one tenant's derived key does not expose another's, and the
        HKDF info binding makes cross-tenant ciphertext reuse fail loudly.
        """
        cached = self._subkeys.get(tenant_id)
        if cached is not None:
            return cached
        subkey = HKDF(
            algorithm=hashes.SHA256(),
            length=TEMPLATE_KEY_BYTES,
            salt=None,
            info=b"nexgen-imatch-template:" + tenant_id.encode("utf-8"),
        ).derive(self._master_key)
        self._subkeys[tenant_id] = subkey
        return subkey


__all__ = [
    "TEMPLATE_KEY_BYTES",
    "EncryptedTemplate",
    "TemplateDecryptionError",
    "TemplateEncryptor",
]

```

