"""Account lifecycle endpoints: registration, e-mail verification, password reset.

Mounted under the same /api/auth prefix as auth.py. Split by file rather than
by URL because auth.py owns "prove who you are and get a token", while this
owns "become a user, confirm your address, recover access" -- different
lifecycles, shared primitives. Both use services/accounts.py for the rules and
services/mail.py for delivery; nothing is reimplemented here.

Two properties are load-bearing throughout and are easy to break by accident:

  NO ENUMERATION. Registration, resend and forgot-password return the same
  response whether or not the address exists. An attacker must not be able to
  use these endpoints to discover who holds an account on a forensic system.

  NOTHING IS TRUSTED FROM THE CLIENT except the code or token itself, which is
  compared against a stored hash in constant time.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session, select

from ...core.config import Settings, get_settings
from ...core.dependencies import client_context
from ...core.rate_limit import SlidingWindowRateLimiter, register_auth_limiter
from ...core.security import hash_password, hash_reset_token, validate_password_strength
from ...db.models import Role, Tenant, User
from ...db.session import get_session
from ...services import accounts, mail_templates
from ...services.audit_service import (
    ACTION_OTP_FAILED,
    ACTION_OTP_SENT,
    ACTION_OTP_VERIFIED,
    ACTION_PASSWORD_RESET,
    ACTION_PASSWORD_RESET_REQUEST,
    ACTION_REGISTER,
    AuditService,
)
from ...services.mail import MailService
from ..schemas import (
    ForgotPasswordRequest,
    MessageResponse,
    RegisterRequest,
    ResendOtpRequest,
    ResetPasswordRequest,
    VerifyEmailRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

# Keyed by client IP: these endpoints are unauthenticated, so there is no
# principal to key on. Separate buckets per flow so a burst of reset requests
# cannot consume the registration budget of everyone behind the same NAT.
_REGISTER_LIMITER = register_auth_limiter(SlidingWindowRateLimiter(max_events=5, window_seconds=300.0))
_VERIFY_LIMITER = register_auth_limiter(SlidingWindowRateLimiter(max_events=10, window_seconds=300.0))
_RESEND_LIMITER = register_auth_limiter(SlidingWindowRateLimiter(max_events=3, window_seconds=1800.0))
_FORGOT_LIMITER = register_auth_limiter(SlidingWindowRateLimiter(max_events=5, window_seconds=900.0))
_RESET_LIMITER = register_auth_limiter(SlidingWindowRateLimiter(max_events=5, window_seconds=900.0))

_RESET_SENT = "If an account exists, a password reset email has been sent."
_REGISTERED = "Registration successful. Please verify your email."


def get_audit_service(settings: Settings = Depends(get_settings)) -> AuditService:
    return AuditService(settings.audit_path)


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


def _domain_allowed(email: str, settings: Settings) -> bool:
    allowed = [d.strip().lower().lstrip("@")
               for d in settings.registration_allowed_domains.split(",") if d.strip()]
    return not allowed or email.rsplit("@", 1)[-1].lower() in allowed


def _send_verification(mail: MailService, user: User, code: str, settings: Settings,
                       *, resend: bool = False) -> bool:
    builder = mail_templates.resend_otp if resend else mail_templates.verify_email
    subject, html, text = builder(
        name=user.full_name, otp=code, ttl_minutes=settings.otp_ttl_minutes)
    return mail.send(to=user.email, subject=subject, html=html, text=text)


# ------------------------------------------------------------- registration --


@router.post("/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    request: Request,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    audit: AuditService = Depends(get_audit_service),
    mail: MailService = Depends(get_mail_service),
) -> MessageResponse:
    """Self-service registration, creating an UNVERIFIED account.

    Disabled unless NEXGEN_ALLOW_SELF_REGISTRATION is set. This is a biometric
    investigation system: who is able to run a search is a controlled decision,
    so an open signup form is something an operator switches on deliberately
    rather than inherits as a default.
    """
    if not settings.allow_self_registration:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Self-service registration is disabled. Contact your administrator for access.",
        )
    _limit(_REGISTER_LIMITER, request, "register")
    ip_address, user_agent = client_context(request)
    email = payload.email

    if not _domain_allowed(email, settings):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "That email domain is not permitted to register.")
    try:
        validate_password_strength(payload.password)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    slug = (payload.tenant or settings.seed_tenant).strip()
    tenant = session.exec(select(Tenant).where(Tenant.slug == slug)).first()
    if tenant is None or not tenant.active:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown organisation.")

    existing = session.exec(
        select(User).where(User.tenant_id == tenant.id, User.email == email)
    ).first()

    if existing is not None:
        # Deliberately identical to the success response. Reporting "already
        # registered" would make this an account-enumeration oracle, which on a
        # forensic system matters more than the small loss of clarity. Whoever
        # actually owns the mailbox still learns what they need: an unverified
        # account receives a fresh code, a verified one receives nothing.
        if not existing.email_verified:
            allowed, _ = accounts.resend_allowed(existing, settings)
            if allowed:
                code = accounts.issue_otp(existing, settings)
                session.add(existing)
                delivered = _send_verification(mail, existing, code, settings)
                audit.record(session, tenant_id=tenant.id, action=ACTION_OTP_SENT,
                             actor_id=existing.id, actor_label=email,
                             detail={"trigger": "register_existing",
                                     "mail_delivered": delivered},
                             ip_address=ip_address, user_agent=user_agent)
                session.commit()
        return MessageResponse(message=_REGISTERED)

    try:
        role = Role(settings.registration_default_role)
    except ValueError:
        role = Role.INVESTIGATOR

    user = User(
        tenant_id=tenant.id,
        email=email,
        full_name=payload.full_name.strip(),
        password_hash=hash_password(payload.password),
        role=role,
        email_verified=False,
    )
    code = accounts.issue_otp(user, settings)
    session.add(user)
    session.flush()

    delivered = _send_verification(mail, user, code, settings)
    audit.record(session, tenant_id=tenant.id, action=ACTION_REGISTER,
                 actor_id=user.id, actor_label=email,
                 detail={"role": role.value, "mail_delivered": delivered},
                 ip_address=ip_address, user_agent=user_agent)
    audit.record(session, tenant_id=tenant.id, action=ACTION_OTP_SENT,
                 actor_id=user.id, actor_label=email, detail={"trigger": "register"},
                 ip_address=ip_address, user_agent=user_agent)
    session.commit()
    return MessageResponse(message=_REGISTERED)


@router.post("/verify-email", response_model=MessageResponse)
def verify_email(
    payload: VerifyEmailRequest,
    request: Request,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    audit: AuditService = Depends(get_audit_service),
    mail: MailService = Depends(get_mail_service),
) -> MessageResponse:
    _limit(_VERIFY_LIMITER, request, "verify")
    ip_address, user_agent = client_context(request)

    user = session.exec(select(User).where(User.email == payload.email)).first()
    if user is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired code.")
    if user.email_verified:
        return MessageResponse(message="Email already verified. You can sign in.",
                               email_verified=True)

    result = accounts.check_otp(user, payload.otp, settings)
    if not result.ok:
        session.add(user)  # persists the incremented attempt counter
        audit.record(session, tenant_id=user.tenant_id, action=ACTION_OTP_FAILED,
                     actor_id=user.id, actor_label=user.email, outcome="failure",
                     detail={"reason": result.reason},
                     ip_address=ip_address, user_agent=user_agent)
        session.commit()
        if result.reason == "expired":
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "That code has expired. Request a new one.")
        if result.reason == "too_many_attempts":
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                                "Too many incorrect attempts. Request a new code.")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired code.")

    accounts.mark_verified(user)
    session.add(user)
    audit.record(session, tenant_id=user.tenant_id, action=ACTION_OTP_VERIFIED,
                 actor_id=user.id, actor_label=user.email,
                 ip_address=ip_address, user_agent=user_agent)
    session.commit()

    subject, html, text = mail_templates.welcome(
        name=user.full_name, login_url=f"{settings.app_public_url.rstrip('/')}/login")
    mail.send(to=user.email, subject=subject, html=html, text=text)
    return MessageResponse(message="Email verified. You can now sign in.", email_verified=True)


@router.post("/resend-otp", response_model=MessageResponse)
def resend_otp(
    payload: ResendOtpRequest,
    request: Request,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    audit: AuditService = Depends(get_audit_service),
    mail: MailService = Depends(get_mail_service),
) -> MessageResponse:
    _limit(_RESEND_LIMITER, request, "resend")
    ip_address, user_agent = client_context(request)
    generic = MessageResponse(
        message="If that account needs verification, a new code has been sent.")

    user = session.exec(select(User).where(User.email == payload.email)).first()
    if user is None or user.email_verified:
        return generic

    # Per-account quota, alongside the per-IP limiter above. The IP limit stops
    # one host hammering the endpoint; this stops a distributed attempt to flood
    # a single mailbox, which is the abuse that actually harms a person and
    # burns sending reputation.
    allowed, retry_after = accounts.resend_allowed(user, settings)
    if not allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many codes requested. Please wait before asking for another.",
            headers={"Retry-After": str(retry_after)},
        )

    code = accounts.issue_otp(user, settings)
    session.add(user)
    delivered = _send_verification(mail, user, code, settings, resend=True)
    audit.record(session, tenant_id=user.tenant_id, action=ACTION_OTP_SENT,
                 actor_id=user.id, actor_label=user.email,
                 detail={"trigger": "resend", "mail_delivered": delivered},
                 ip_address=ip_address, user_agent=user_agent)
    session.commit()
    return generic


# ----------------------------------------------------------- password reset --


@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(
    payload: ForgotPasswordRequest,
    request: Request,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    audit: AuditService = Depends(get_audit_service),
    mail: MailService = Depends(get_mail_service),
) -> MessageResponse:
    _limit(_FORGOT_LIMITER, request, "forgot")
    ip_address, user_agent = client_context(request)

    user = session.exec(select(User).where(User.email == payload.email)).first()
    if user is not None and user.active:
        token = accounts.issue_reset_token(user, settings)
        session.add(user)
        reset_url = f"{settings.app_public_url.rstrip('/')}/reset-password?token={token}"
        subject, html, text = mail_templates.forgot_password(
            name=user.full_name, reset_url=reset_url,
            ttl_minutes=settings.reset_token_ttl_minutes)
        mail.send(to=user.email, subject=subject, html=html, text=text)
        audit.record(session, tenant_id=user.tenant_id,
                     action=ACTION_PASSWORD_RESET_REQUEST,
                     actor_id=user.id, actor_label=user.email,
                     ip_address=ip_address, user_agent=user_agent)
        session.commit()

    # Identical message and status either way.
    return MessageResponse(message=_RESET_SENT)


@router.post("/reset-password", response_model=MessageResponse)
def reset_password(
    payload: ResetPasswordRequest,
    request: Request,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    audit: AuditService = Depends(get_audit_service),
    mail: MailService = Depends(get_mail_service),
) -> MessageResponse:
    _limit(_RESET_LIMITER, request, "reset")
    ip_address, user_agent = client_context(request)

    try:
        validate_password_strength(payload.password)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    # Looked up BY HASH, so the plaintext token never reaches a query and a
    # stolen database cannot be used to reset anyone's password.
    digest = hash_reset_token(payload.token)
    user = session.exec(select(User).where(User.reset_token_hash == digest)).first()
    if user is None or not accounts.check_reset_token(user, payload.token):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "This reset link is invalid or has expired.")

    user.password_hash = hash_password(payload.password)
    accounts.consume_reset_token(user)
    accounts.invalidate_sessions(user)
    accounts.clear_login_failures(user)
    # Controlling the mailbox proves exactly what the OTP asks for, so a reset
    # also verifies an address that had not been confirmed yet.
    user.email_verified = True
    session.add(user)

    audit.record(session, tenant_id=user.tenant_id, action=ACTION_PASSWORD_RESET,
                 actor_id=user.id, actor_label=user.email,
                 ip_address=ip_address, user_agent=user_agent)
    session.commit()

    subject, html, text = mail_templates.password_changed(
        name=user.full_name, ip_address=ip_address or "")
    mail.send(to=user.email, subject=subject, html=html, text=text)
    return MessageResponse(message="Password updated. You can now sign in.")
