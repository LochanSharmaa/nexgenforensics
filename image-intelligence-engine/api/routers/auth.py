"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from shared.errors import AuthenticationError
from shared.logging import get_logger

from ..dependencies import (
    AuditRepoDep,
    ClockDep,
    CurrentUser,
    SettingsDep,
    UserRepoDep,
    client_label,
)
from ..schemas import LoginRequest, TokenResponse, UserResponse
from ..security import create_access_token, verify_password

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    settings: SettingsDep,
    users: UserRepoDep,
    audit: AuditRepoDep,
    clock: ClockDep,
) -> TokenResponse:
    user = await users.get_by_email(payload.email)

    # Same message and roughly the same work whether the account exists or the
    # password is wrong, so the response cannot be used to enumerate accounts.
    if user is None or not verify_password(payload.password, user.password_hash):
        await audit.record(
            action="auth.login",
            outcome="failed",
            actor_label=client_label(request),
            detail={"email": payload.email, "reason": "invalid credentials"},
        )
        raise AuthenticationError("Incorrect email or password.")

    if not user.is_active:
        await audit.record(
            action="auth.login",
            outcome="failed",
            actor_id=user.id,
            actor_label=client_label(request, user),
            detail={"reason": "account disabled"},
        )
        raise AuthenticationError("Incorrect email or password.")

    user.last_login_at = clock.now()
    token = create_access_token(user.id, settings, role=user.role)

    await audit.record(
        action="auth.login",
        outcome="succeeded",
        actor_id=user.id,
        actor_label=client_label(request, user),
        detail={},
    )
    logger.info("auth.login", user_id=str(user.id))
    return TokenResponse(
        access_token=token, expires_in_minutes=settings.access_token_ttl_minutes
    )


@router.get("/me", response_model=UserResponse)
async def me(user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(user)


__all__ = ["router"]
