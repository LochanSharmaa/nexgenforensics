from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session, select

from ...core.config import Settings, get_settings
from ...core.dependencies import (
    Principal,
    client_context,
    get_current_principal,
    require_admin,
)
from ...core.security import (
    TokenError,
    create_token,
    decode_token,
    hash_password,
    needs_rehash,
    validate_password_strength,
    verify_password,
)
from ...db.models import Tenant, User, utcnow
from ...db.session import get_session
from ...services.audit_service import (
    ACTION_LOGIN,
    ACTION_LOGIN_FAILED,
    ACTION_LOGOUT,
    ACTION_USER_CREATE,
    AuditService,
)
from ..schemas import CreateUserRequest, LoginRequest, RefreshRequest, TokenResponse, UserResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

_INVALID_CREDENTIALS = "Incorrect email or password."


def get_audit_service(settings: Settings = Depends(get_settings)) -> AuditService:
    return AuditService(settings.audit_path)


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    request: Request,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    audit: AuditService = Depends(get_audit_service),
) -> TokenResponse:
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

    if user is None or not password_ok or not user.active:
        if user is not None:
            audit.record(
                session,
                tenant_id=user.tenant_id,
                action=ACTION_LOGIN_FAILED,
                actor_label=email,
                outcome="failure",
                detail={"reason": "inactive" if not user.active else "bad_password"},
                ip_address=ip_address,
                user_agent=user_agent,
            )
            session.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _INVALID_CREDENTIALS)

    tenant = session.get(Tenant, user.tenant_id)
    if tenant is None or not tenant.active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This tenant is not active.")

    # Transparently upgrade hashes when the cost parameters change.
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)

    user.last_login_at = utcnow()
    session.add(user)

    audit.record(
        session,
        tenant_id=user.tenant_id,
        action=ACTION_LOGIN,
        actor_id=user.id,
        actor_label=user.email,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()
    session.refresh(user)

    return _issue_tokens(user, settings)


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

    return _issue_tokens(user, settings)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    request: Request,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
    audit: AuditService = Depends(get_audit_service),
) -> None:
    """Record the logout.

    Access tokens are stateless and stay valid until they expire; the client is
    responsible for discarding them. Keep the access-token lifetime short
    (NEXGEN_ACCESS_TOKEN_MINUTES) so that window is small, or add a revocation
    store if immediate invalidation is a requirement.
    """
    ip_address, user_agent = client_context(request)
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


def _issue_tokens(user: User, settings: Settings) -> TokenResponse:
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
        expires_in=timedelta(days=settings.refresh_token_days),
    )
    return TokenResponse(
        access_token=access,
        refresh_token=refresh_token,
        expires_in=int(access_ttl.total_seconds()),
        user=UserResponse.model_validate(user),
    )


# Argon2 hash of a value nobody can supply, used to keep login timing flat.
_DUMMY_HASH = hash_password("nexgen-imatch-timing-equalizer")


__all__ = ["get_audit_service", "router"]
