from __future__ import annotations

import hmac
import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlmodel import Session, select

from ...core.config import Settings, get_settings
from ...core.dependencies import (
    Principal,
    client_context,
    get_current_principal,
    require_admin,
)
from ...core.rate_limit import SlidingWindowRateLimiter, register_auth_limiter
from ...core.security import (
    TokenError,
    create_token,
    decode_token,
    hash_password,
    hash_refresh_token,
    needs_rehash,
    validate_password_strength,
    verify_password,
)
from ...db.models import Tenant, User, utcnow
from ...db.session import get_session
from ...services import accounts, mail_templates
from ...services.audit_service import (
    ACTION_ACCOUNT_LOCKED,
    ACTION_LOGIN,
    ACTION_LOGIN_FAILED,
    ACTION_LOGOUT,
    ACTION_OTP_FAILED,
    ACTION_OTP_SENT,
    ACTION_OTP_VERIFIED,
    ACTION_PASSWORD_RESET,
    ACTION_PASSWORD_RESET_REQUEST,
    ACTION_REGISTER,
    ACTION_USER_CREATE,
    AuditService,
)
from ...services.mail import MailService
from ..schemas import (
    CreateUserRequest,
    ForgotPasswordRequest,
    LoginRequest,
    MessageResponse,
    RefreshRequest,
    RegisterRequest,
    ResendOtpRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserResponse,
    VerifyEmailRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

_INVALID_CREDENTIALS = "Incorrect email or password."

# Unauthenticated endpoints cannot key a limiter on a principal, so these are
# keyed by client IP. Separate buckets per flow: a burst of password-reset
# requests should not consume the login budget of everyone behind the same NAT.
# Registration, verification and reset limiters live in routes/account.py
# alongside the endpoints they guard.
_LOGIN_LIMITER = register_auth_limiter(
    SlidingWindowRateLimiter(max_events=10, window_seconds=60.0))

# Returned for every forgot-password call, whether or not the address exists.
# Any difference here -- wording, status code or response time -- turns the
# endpoint into an account-enumeration oracle.
_RESET_SENT = "If an account exists, a password reset email has been sent."


def get_mail_service(settings: Settings = Depends(get_settings)) -> MailService:
    return MailService(settings)


def _limit(limiter: SlidingWindowRateLimiter, request: Request, what: str) -> None:
    ip_address, _ = client_context(request)
    result = limiter.check(f"{what}:{ip_address or 'unknown'}")
    if not result.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many requests. Please wait and try again.",
            headers={"Retry-After": str(int(result.retry_after) + 1)},
        )


def _set_auth_cookies(response: Response, tokens: TokenResponse, settings: Settings,
                      remember: bool) -> None:
    """Mirror the tokens into HTTPOnly cookies.

    The body still carries them, because the existing SPA reads Bearer tokens
    and changing that in the same step would break every authenticated screen.
    The cookies are the hardening: JavaScript cannot read an HTTPOnly cookie, so
    an XSS bug cannot exfiltrate the session from them.
    """
    if not settings.auth_cookies_enabled:
        return
    common = {
        "httponly": True,
        "secure": settings.cookie_secure,
        "samesite": settings.cookie_samesite,
        "path": "/",
    }
    if settings.cookie_domain:
        common["domain"] = settings.cookie_domain
    refresh_days = settings.remember_me_refresh_days if remember else settings.refresh_token_days
    response.set_cookie("nx_access", tokens.access_token,
                        max_age=settings.access_token_minutes * 60, **common)
    response.set_cookie("nx_refresh", tokens.refresh_token,
                        max_age=refresh_days * 86400, **common)


def _clear_auth_cookies(response: Response, settings: Settings) -> None:
    for name in ("nx_access", "nx_refresh"):
        response.delete_cookie(name, path="/",
                               domain=settings.cookie_domain or None)


def get_audit_service(settings: Settings = Depends(get_settings)) -> AuditService:
    return AuditService(settings.audit_path)


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    audit: AuditService = Depends(get_audit_service),
) -> TokenResponse:
    _limit(_LOGIN_LIMITER, request, "login")
    ip_address, user_agent = client_context(request)
    email = payload.email.lower().strip()

    statement = select(User).where(User.email == email)
    if payload.tenant.strip():
        tenant = session.exec(select(Tenant).where(Tenant.slug == payload.tenant.strip())).first()
        if tenant is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, _INVALID_CREDENTIALS)
        statement = statement.where(User.tenant_id == tenant.id)

    users = session.exec(statement).all()

    if len(users) > 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "This email exists in more than one tenant. Supply the tenant slug.",
        )

    user = users[0] if users else None

    # Always run a verification, even with no user, so a missing account and a
    # wrong password take the same time. Otherwise response timing reveals which
    # emails are registered.
    stored_hash = user.password_hash if user else _DUMMY_HASH
    password_ok = verify_password(payload.password, stored_hash)

    # Lockout is checked BEFORE the credential verdict. Checking afterwards
    # would let an attacker keep testing passwords against a locked account and
    # learn from the response which guess was right.
    if user is not None and accounts.is_locked(user):
        audit.record(
            session, tenant_id=user.tenant_id, action=ACTION_LOGIN_FAILED,
            actor_id=user.id, actor_label=email, outcome="failure",
            detail={"reason": "locked"}, ip_address=ip_address, user_agent=user_agent,
        )
        session.commit()
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many failed attempts. This account is temporarily locked.",
            headers={"Retry-After": str(accounts.lock_remaining_seconds(user))},
        )

    if user is None or not password_ok or not user.active:
        if user is not None:
            locked_now = False
            if user.active and not password_ok:
                locked_now = accounts.register_failed_login(user, settings)
            session.add(user)
            audit.record(
                session,
                tenant_id=user.tenant_id,
                action=ACTION_ACCOUNT_LOCKED if locked_now else ACTION_LOGIN_FAILED,
                actor_id=user.id,
                actor_label=email,
                outcome="failure",
                detail={"reason": "inactive" if not user.active else "bad_password",
                        "locked": locked_now},
                ip_address=ip_address,
                user_agent=user_agent,
            )
            session.commit()
            if locked_now:
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    "Too many failed attempts. This account is temporarily locked.",
                    headers={"Retry-After": str(settings.lockout_minutes * 60)},
                )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _INVALID_CREDENTIALS)

    # Credentials are correct but the address was never confirmed. Reported
    # only AFTER the password check, so it cannot be used to enumerate accounts.
    if not user.email_verified:
        audit.record(
            session, tenant_id=user.tenant_id, action=ACTION_LOGIN_FAILED,
            actor_id=user.id, actor_label=email, outcome="failure",
            detail={"reason": "email_unverified"},
            ip_address=ip_address, user_agent=user_agent,
        )
        session.commit()
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "Please verify your email before logging in."
        )

    tenant = session.get(Tenant, user.tenant_id)
    if tenant is None or not tenant.active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This tenant is not active.")

    # Transparently upgrade hashes when the cost parameters change.
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)

    accounts.clear_login_failures(user)
    user.last_login_at = utcnow()
    user.last_login_ip = ip_address or ""
    session.add(user)

    tokens = _issue_tokens(user, settings, remember=payload.remember_me)
    _remember_refresh(user, tokens.refresh_token, settings, payload.remember_me)
    session.add(user)

    audit.record(
        session,
        tenant_id=user.tenant_id,
        action=ACTION_LOGIN,
        actor_id=user.id,
        actor_label=user.email,
        detail={"remember_me": payload.remember_me},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()

    _set_auth_cookies(response, tokens, settings, payload.remember_me)
    return tokens


@router.post("/refresh", response_model=TokenResponse)
def refresh(
    payload: RefreshRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    try:
        claims = decode_token(
            payload.refresh_token,
            settings.resolved_jwt_secret(),
            settings.jwt_algorithm,
            expected_type="refresh",
        )
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    user = session.get(User, claims.subject_id)
    if user is None or not user.active or user.tenant_id != claims.tenant_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token is no longer valid.")

    # A signature-valid token is not enough. It must also be the one currently
    # on record, so that logout, a password reset, or a newer sign-in genuinely
    # ends this session instead of leaving the old token usable until expiry.
    presented = hash_refresh_token(payload.refresh_token)
    if not user.refresh_token_hash or not hmac.compare_digest(presented, user.refresh_token_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token is no longer valid.")

    if not user.email_verified:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Please verify your email before logging in.")

    # Rotate on every use: a refresh token is single-use, so replaying a stolen
    # one after the legitimate client has refreshed will fail.
    tokens = _issue_tokens(user, settings)
    _remember_refresh(user, tokens.refresh_token, settings, remember=False)
    session.add(user)
    session.commit()
    return tokens


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    request: Request,
    response: Response,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    audit: AuditService = Depends(get_audit_service),
) -> None:
    """End the session.

    The stored refresh-token hash is cleared, so the refresh token presented
    later is rejected and the session cannot be resumed. Access tokens remain
    stateless and expire on their own within NEXGEN_ACCESS_TOKEN_MINUTES; the
    cookies carrying them are cleared here too.
    """
    ip_address, user_agent = client_context(request)
    user = session.get(User, principal.id)
    if user is not None:
        user.refresh_token_hash = None
        user.refresh_token_expires_at = None
        user.updated_at = utcnow()
        session.add(user)
    _clear_auth_cookies(response, settings)
    audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_LOGOUT,
        actor_id=principal.id,
        actor_label=principal.label,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()


@router.get("/me", response_model=UserResponse)
def me(
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> UserResponse:
    if principal.credential == "api_key":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "API keys have no user profile.")
    user = session.get(User, principal.id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    return UserResponse.model_validate(user)


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: CreateUserRequest,
    request: Request,
    principal: Principal = Depends(require_admin),
    session: Session = Depends(get_session),
    audit: AuditService = Depends(get_audit_service),
) -> UserResponse:
    """Create a user inside the caller's tenant.

    The tenant is taken from the authenticated principal, never from the
    request, so an admin cannot create accounts in another tenant.
    """
    try:
        validate_password_strength(payload.password)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    email = payload.email.lower().strip()
    existing = session.exec(
        select(User).where(User.tenant_id == principal.tenant_id, User.email == email)
    ).first()
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "A user with that email already exists in this tenant.")

    user = User(
        tenant_id=principal.tenant_id,
        email=email,
        full_name=payload.full_name.strip(),
        password_hash=hash_password(payload.password),
        role=payload.role,
        badge_number=payload.badge_number.strip(),
        # Created by a supervisor, who has already vouched for this address, so
        # it does not need an e-mail round-trip before it can be used.
        email_verified=True,
    )
    session.add(user)
    session.flush()

    ip_address, user_agent = client_context(request)
    audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_USER_CREATE,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="user",
        resource_id=user.id,
        detail={"email": email, "role": payload.role.value},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()
    session.refresh(user)
    return UserResponse.model_validate(user)


def _issue_tokens(user: User, settings: Settings, *, remember: bool = False) -> TokenResponse:
    secret = settings.resolved_jwt_secret()
    access_ttl = timedelta(minutes=settings.access_token_minutes)
    access = create_token(
        subject_id=user.id,
        tenant_id=user.tenant_id,
        role=user.role.value,
        secret=secret,
        algorithm=settings.jwt_algorithm,
        token_type="access",
        expires_in=access_ttl,
    )
    refresh_token = create_token(
        subject_id=user.id,
        tenant_id=user.tenant_id,
        role=user.role.value,
        secret=secret,
        algorithm=settings.jwt_algorithm,
        token_type="refresh",
        # "Remember me" extends ONLY the refresh token. The access token
        # lifetime is untouched, so a stolen access token is never long-lived
        # and the longer session still re-checks the account on every refresh.
        expires_in=timedelta(
            days=settings.remember_me_refresh_days if remember else settings.refresh_token_days
        ),
    )
    return TokenResponse(
        access_token=access,
        refresh_token=refresh_token,
        expires_in=int(access_ttl.total_seconds()),
        user=UserResponse.model_validate(user),
    )


def _remember_refresh(user: User, refresh_token: str, settings: Settings, remember: bool) -> None:
    """Persist the hash of the refresh token that is now current.

    Storing it is what makes logout and reset actually revoke. Only one refresh
    token is valid per user at a time, so signing in elsewhere ends the previous
    session -- a deliberate trade of multi-device convenience for the ability to
    say, truthfully, that logout ends the session.
    """
    days = settings.remember_me_refresh_days if remember else settings.refresh_token_days
    user.refresh_token_hash = hash_refresh_token(refresh_token)
    user.refresh_token_expires_at = utcnow() + timedelta(days=days)
    user.updated_at = utcnow()


# Argon2 hash of a value nobody can supply, used to keep login timing flat.
_DUMMY_HASH = hash_password("nexgen-imatch-timing-equalizer")


__all__ = ["get_audit_service", "router"]
