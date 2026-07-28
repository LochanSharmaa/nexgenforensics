from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session, select

from ...core.dependencies import Principal, client_context, require_admin
from ...db.models import ApiKey, utcnow
from ...db.session import get_session
from ...services.audit_service import ACTION_KEY_CREATE, ACTION_KEY_REVOKE, AuditService
from ..schemas import ApiKeyResponse, CreateApiKeyRequest, CreatedApiKeyResponse
from .auth import get_audit_service

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/api-keys", response_model=list[ApiKeyResponse])
def list_api_keys(
    principal: Principal = Depends(require_admin),
    session: Session = Depends(get_session),
) -> list[ApiKeyResponse]:
    keys = session.exec(
        select(ApiKey).where(ApiKey.tenant_id == principal.tenant_id).order_by(ApiKey.created_at.desc())
    ).all()
    return [ApiKeyResponse.model_validate(key, from_attributes=True) for key in keys]


@router.post("/api-keys", response_model=CreatedApiKeyResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(
    payload: CreateApiKeyRequest,
    request: Request,
    principal: Principal = Depends(require_admin),
    session: Session = Depends(get_session),
    audit: AuditService = Depends(get_audit_service),
) -> CreatedApiKeyResponse:
    """Issue a machine credential.

    The plaintext key appears in this response and nowhere else -- only its hash
    is stored, so it cannot be recovered or re-displayed. A key can never exceed
    the role of the admin who created it, and it is bound to their tenant.
    """
    if payload.role.value == "admin" and principal.credential == "api_key":
        # An API key minting another admin key would let a leaked machine
        # credential escalate itself indefinitely.
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Admin-role API keys must be created by an interactive administrator session.",
        )

    from ...core.security import generate_api_key

    plaintext, prefix, key_hash = generate_api_key()
    expires_at = utcnow() + timedelta(days=payload.expires_in_days) if payload.expires_in_days else None

    record = ApiKey(
        tenant_id=principal.tenant_id,
        name=payload.name.strip(),
        prefix=prefix,
        key_hash=key_hash,
        role=payload.role,
        created_by=principal.id,
        expires_at=expires_at,
    )
    session.add(record)
    session.flush()

    ip_address, user_agent = client_context(request)
    audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_KEY_CREATE,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="api_key",
        resource_id=record.id,
        detail={"name": record.name, "role": record.role.value, "expires_at": str(expires_at)},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()
    session.refresh(record)

    return CreatedApiKeyResponse(
        id=record.id,
        name=record.name,
        prefix=record.prefix,
        role=record.role,
        active=record.active,
        expires_at=record.expires_at,
        created_at=record.created_at,
        api_key=plaintext,
    )


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_api_key(
    key_id: str,
    request: Request,
    principal: Principal = Depends(require_admin),
    session: Session = Depends(get_session),
    audit: AuditService = Depends(get_audit_service),
) -> None:
    record = session.get(ApiKey, key_id)
    if record is None or record.tenant_id != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "API key not found.")

    # Deactivate rather than delete: the audit trail references this key id, and
    # a dangling reference would make past actions unattributable.
    record.active = False
    session.add(record)

    ip_address, user_agent = client_context(request)
    audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_KEY_REVOKE,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="api_key",
        resource_id=record.id,
        detail={"name": record.name},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()


__all__ = ["router"]
