from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import Depends, Header, HTTPException, Request, status
from sqlmodel import Session, select

from ..db.models import ApiKey, Role, Tenant, User, utcnow
from ..db.session import get_session
from .config import Settings, get_settings
from .rate_limit import SlidingWindowRateLimiter
from .security import TokenError, api_key_prefix, decode_token, verify_api_key

logger = logging.getLogger(__name__)

_ROLE_RANK = {Role.INVESTIGATOR: 1, Role.SUPERVISOR: 2, Role.ADMIN: 3}

_general_limiter: SlidingWindowRateLimiter | None = None
_search_limiter: SlidingWindowRateLimiter | None = None


@dataclass(frozen=True)
class Principal:
    """The authenticated caller.

    ``tenant_id`` comes from the verified credential, never from a request body
    or header the client controls. Every repository query is scoped by it, so a
    caller cannot reach another tenant's biometric data even by guessing ids.
    """

    id: str
    tenant_id: str
    role: Role
    label: str
    credential: str  # "session" or "api_key"

    def has_role(self, minimum: Role) -> bool:
        return _ROLE_RANK[self.role] >= _ROLE_RANK[minimum]

    @property
    def rate_limit_key(self) -> str:
        return f"{self.credential}:{self.id}"


def get_general_limiter(settings: Settings = Depends(get_settings)) -> SlidingWindowRateLimiter:
    global _general_limiter
    if _general_limiter is None:
        _general_limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)
    return _general_limiter


def get_search_limiter(settings: Settings = Depends(get_settings)) -> SlidingWindowRateLimiter:
    global _search_limiter
    if _search_limiter is None:
        _search_limiter = SlidingWindowRateLimiter(settings.search_rate_limit_per_minute)
    return _search_limiter


def reset_limiters() -> None:
    """Used by tests, which would otherwise trip the limiter across cases."""
    global _general_limiter, _search_limiter
    _general_limiter = None
    _search_limiter = None


def get_current_principal(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Principal:
    """Authenticate via bearer token or API key.

    Failures return a single generic message. Distinguishing "no such user" from
    "wrong password" here would let an unauthenticated caller enumerate accounts.
    """
    principal = None
    if x_api_key:
        principal = _principal_from_api_key(x_api_key, session)
    elif authorization:
        principal = _principal_from_bearer(authorization, session, settings)

    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    request.state.principal = principal
    return principal


def _principal_from_bearer(authorization: str, session: Session, settings: Settings) -> Principal | None:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None

    try:
        payload = decode_token(token, settings.resolved_jwt_secret(), settings.jwt_algorithm, "access")
    except TokenError:
        return None

    user = session.get(User, payload.subject_id)
    if user is None or not user.active:
        return None
    # The token carries a tenant claim, but the database is authoritative: a user
    # moved between tenants must not keep acting under the old one until expiry.
    if user.tenant_id != payload.tenant_id:
        logger.warning("Token tenant claim does not match stored user %s; rejecting.", user.id)
        return None
    if not _tenant_active(session, user.tenant_id):
        return None

    return Principal(
        id=user.id,
        tenant_id=user.tenant_id,
        role=user.role,
        label=user.email,
        credential="session",
    )


def _principal_from_api_key(raw_key: str, session: Session) -> Principal | None:
    record = session.exec(select(ApiKey).where(ApiKey.prefix == api_key_prefix(raw_key))).first()
    if record is None or not record.active:
        return None
    if not verify_api_key(raw_key, record.key_hash):
        return None
    if record.expires_at is not None and _as_utc(record.expires_at) < datetime.now(timezone.utc):
        return None
    if not _tenant_active(session, record.tenant_id):
        return None

    record.last_used_at = utcnow()
    session.add(record)

    return Principal(
        id=record.id,
        tenant_id=record.tenant_id,
        role=record.role,
        label=f"api-key:{record.name}",
        credential="api_key",
    )


def _tenant_active(session: Session, tenant_id: str) -> bool:
    tenant = session.get(Tenant, tenant_id)
    return tenant is not None and tenant.active


def _as_utc(value: datetime) -> datetime:
    """SQLite hands back naive datetimes; compare them as UTC, not local time."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def require_role(minimum: Role) -> Callable[[Principal], Principal]:
    def dependency(principal: Principal = Depends(get_current_principal)) -> Principal:
        if not principal.has_role(minimum):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires the {minimum.value} role or higher.",
            )
        return principal

    return dependency


require_investigator = require_role(Role.INVESTIGATOR)
require_supervisor = require_role(Role.SUPERVISOR)
require_admin = require_role(Role.ADMIN)


def enforce_rate_limit(
    principal: Principal = Depends(get_current_principal),
    limiter: SlidingWindowRateLimiter = Depends(get_general_limiter),
) -> Principal:
    result = limiter.check(principal.rate_limit_key)
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(int(result.retry_after) + 1)},
        )
    return principal


def enforce_search_rate_limit(
    principal: Principal = Depends(get_current_principal),
    limiter: SlidingWindowRateLimiter = Depends(get_search_limiter),
) -> Principal:
    """Tighter limit on biometric search specifically.

    Search is the expensive and the privacy-sensitive path: a bulk scrape of a
    gallery looks exactly like a burst of ordinary searches, so it gets its own
    lower ceiling rather than sharing the general budget.
    """
    result = limiter.check(principal.rate_limit_key)
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Biometric search rate limit exceeded.",
            headers={"Retry-After": str(int(result.retry_after) + 1)},
        )
    return principal


def client_context(request: Request) -> tuple[str, str]:
    """Best-effort caller identification for the audit record.

    X-Forwarded-For is client-controllable unless a trusted proxy overwrites it,
    so it is recorded as a claim, not as fact.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "")
    return ip, request.headers.get("user-agent", "")[:512]


__all__ = [
    "Principal",
    "client_context",
    "enforce_rate_limit",
    "enforce_search_rate_limit",
    "get_current_principal",
    "get_general_limiter",
    "get_search_limiter",
    "require_admin",
    "require_investigator",
    "require_role",
    "require_supervisor",
    "reset_limiters",
]
