"""Account lifecycle: verification codes, lockout, password reset.

Kept out of the route module so the rules live in one testable place and are
not restated per endpoint. The routes stay thin: parse, call, audit, respond.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..core.config import Settings
from ..core.security import (
    generate_otp,
    generate_reset_token,
    hash_otp,
    verify_otp,
    verify_reset_token,
)
from ..db.models import User, utcnow


def _aware(value: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes even when an aware one was stored.
    Comparing naive to aware raises TypeError, which on an expiry check would
    surface as a 500 on a completely ordinary request, so every value read back
    from the database is normalised to UTC before it is compared.
    """
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# ------------------------------------------------------------------- lockout --


def is_locked(user: User) -> bool:
    until = _aware(user.locked_until)
    return until is not None and until > utcnow()


def lock_remaining_seconds(user: User) -> int:
    until = _aware(user.locked_until)
    if until is None:
        return 0
    return max(0, int((until - utcnow()).total_seconds()))


def register_failed_login(user: User, settings: Settings) -> bool:
    """Count a failed attempt. Returns True if this one locked the account."""
    user.failed_login_attempts += 1
    user.updated_at = utcnow()
    if user.failed_login_attempts >= settings.max_failed_logins:
        user.locked_until = utcnow() + timedelta(minutes=settings.lockout_minutes)
        # Reset the counter with the lock. Without this the account re-locks on
        # the very next failure after expiry, and a user who simply mistyped
        # their password once more is locked out permanently in practice.
        user.failed_login_attempts = 0
        return True
    return False


def clear_login_failures(user: User) -> None:
    user.failed_login_attempts = 0
    user.locked_until = None
    user.updated_at = utcnow()


# ----------------------------------------------------------------------- otp --


def issue_otp(user: User, settings: Settings) -> str:
    """Generate a code, store only its hash, and return the plaintext once.

    Issuing invalidates any previous code by overwriting the hash, so two codes
    are never live at the same time.
    """
    code = generate_otp(settings.otp_length)
    user.otp_hash = hash_otp(code)
    user.otp_expires_at = utcnow() + timedelta(minutes=settings.otp_ttl_minutes)
    user.otp_attempts = 0
    user.updated_at = utcnow()

    now = utcnow()
    window_started = _aware(user.otp_window_started_at)
    if window_started is None or (now - window_started) > timedelta(
        minutes=settings.otp_resend_window_minutes
    ):
        user.otp_window_started_at = now
        user.otp_sent_count = 1
    else:
        user.otp_sent_count += 1
    return code


def resend_allowed(user: User, settings: Settings) -> tuple[bool, int]:
    """Per-account resend quota. Returns (allowed, seconds_until_reset).

    This sits alongside the IP rate limiter rather than replacing it: the IP
    limit stops one host hammering the endpoint, this stops a distributed
    attempt to flood ONE mailbox, which is the abuse that actually harms a user
    and burns sending reputation.
    """
    started = _aware(user.otp_window_started_at)
    if started is None:
        return True, 0
    window = timedelta(minutes=settings.otp_resend_window_minutes)
    elapsed = utcnow() - started
    if elapsed > window:
        return True, 0
    if user.otp_sent_count < settings.otp_resend_max:
        return True, 0
    return False, max(0, int((window - elapsed).total_seconds()))


class OtpResult:
    __slots__ = ("ok", "reason", "attempts_left")

    def __init__(self, ok: bool, reason: str = "", attempts_left: int = 0):
        self.ok, self.reason, self.attempts_left = ok, reason, attempts_left


def check_otp(user: User, code: str, settings: Settings) -> OtpResult:
    if not user.otp_hash:
        return OtpResult(False, "no_code")

    expires = _aware(user.otp_expires_at)
    if expires is None or expires <= utcnow():
        return OtpResult(False, "expired")

    if user.otp_attempts >= settings.otp_max_attempts:
        return OtpResult(False, "too_many_attempts")

    if not verify_otp(code, user.otp_hash):
        # Count the failure BEFORE returning, otherwise the attempt cap never
        # advances and the code can be brute-forced within its lifetime.
        user.otp_attempts += 1
        user.updated_at = utcnow()
        left = max(0, settings.otp_max_attempts - user.otp_attempts)
        return OtpResult(False, "mismatch", left)

    return OtpResult(True)


def mark_verified(user: User) -> None:
    user.email_verified = True
    user.otp_hash = None
    user.otp_expires_at = None
    user.otp_attempts = 0
    user.otp_sent_count = 0
    user.otp_window_started_at = None
    user.updated_at = utcnow()


# --------------------------------------------------------------------- reset --


def issue_reset_token(user: User, settings: Settings) -> str:
    plaintext, digest = generate_reset_token()
    user.reset_token_hash = digest
    user.reset_token_expires_at = utcnow() + timedelta(minutes=settings.reset_token_ttl_minutes)
    user.updated_at = utcnow()
    return plaintext


def check_reset_token(user: User, token: str) -> bool:
    expires = _aware(user.reset_token_expires_at)
    if expires is None or expires <= utcnow():
        return False
    return verify_reset_token(token, user.reset_token_hash)


def consume_reset_token(user: User) -> None:
    user.reset_token_hash = None
    user.reset_token_expires_at = None
    user.updated_at = utcnow()


def invalidate_sessions(user: User) -> None:
    """Sign the user out everywhere.

    Refresh tokens are revoked by dropping the stored hash. Access tokens are
    stateless and cannot be withdrawn individually, so `session_epoch` is bumped
    and checked at authentication time -- that is what makes "invalidate
    previous sessions" true for tokens already in the wild rather than only for
    future ones.
    """
    user.refresh_token_hash = None
    user.refresh_token_expires_at = None
    user.session_epoch += 1
    user.updated_at = utcnow()
