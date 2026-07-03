from __future__ import annotations

import uuid

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from ..auth import AuthService, Principal, require_role
from ..storage import Database, UserRecord


class RegisterRequest(BaseModel):
    tenant_id: str
    tenant_name: str
    email: str
    password: str
    role: str = "tenant_admin"


class LoginRequest(BaseModel):
    email: str
    password: str


class UserCreateRequest(BaseModel):
    email: str
    password: str
    role: str = "operator"


def build_auth_router(database: Database, auth: AuthService) -> APIRouter:
    router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
    database.migrate()

    @router.post("/register")
    def register(payload: RegisterRequest) -> dict:
        database.upsert_tenant(payload.tenant_id, payload.tenant_name)
        user = UserRecord(
            user_id=f"user_{uuid.uuid4().hex[:12]}",
            tenant_id=payload.tenant_id,
            email=str(payload.email),
            role=payload.role,
            password_hash=auth.hash_password(payload.password, payload.email),
        )
        try:
            database.create_user(user)
        except Exception as exc:
            raise HTTPException(status_code=409, detail="User already exists or tenant is invalid.") from exc
        token = auth.issue_token(Principal(user.user_id, user.tenant_id, user.role, user.email))
        return {"access_token": token, "token_type": "bearer", "user": _user_payload(user)}

    @router.post("/login")
    def login(payload: LoginRequest) -> dict:
        user = database.get_user_by_email(str(payload.email))
        if user is None or not auth.verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials.")
        token = auth.issue_token(Principal(user.user_id, user.tenant_id, user.role, user.email))
        return {"access_token": token, "token_type": "bearer", "user": _user_payload(user)}

    @router.get("/me")
    def me(authorization: str | None = Header(default=None)) -> dict:
        principal = _principal(auth, authorization)
        return principal.__dict__

    @router.post("/users")
    def create_user(payload: UserCreateRequest, authorization: str | None = Header(default=None)) -> dict:
        principal = _principal(auth, authorization)
        require_role(principal.role, "tenant_admin")
        user = UserRecord(
            user_id=f"user_{uuid.uuid4().hex[:12]}",
            tenant_id=principal.tenant_id,
            email=str(payload.email),
            role=payload.role,
            password_hash=auth.hash_password(payload.password, payload.email),
        )
        database.create_user(user)
        return _user_payload(user)

    return router


def _principal(auth: AuthService, authorization: str | None) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required.")
    try:
        return auth.verify_token(authorization.split(" ", 1)[1])
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid token.") from exc


def _user_payload(user: UserRecord) -> dict:
    return {"user_id": user.user_id, "tenant_id": user.tenant_id, "email": user.email, "role": user.role}
