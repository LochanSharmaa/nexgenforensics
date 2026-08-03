"""FastAPI dependency wiring.

Everything a router needs arrives through these. Routers hold no business logic
and construct no infrastructure themselves — an architecture test enforces it,
because a router that reaches for a session directly is the first step toward
logic that only the HTTP layer can execute.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database.models import User
from database.repositories import (
    AuditRepository,
    CustodyRepository,
    FactRepository,
    ImageRepository,
    InvestigationRepository,
    LifecycleRepository,
    ObservationRepository,
    PipelineRepository,
    RetentionRepository,
    ReviewRepository,
    UserRepository,
)
from database.session import get_sessionmaker
from shared.clock import Clock, SystemClock
from shared.config import Settings, get_settings
from shared.errors import AuthenticationError
from shared.storage import FilesystemObjectStore, ObjectStore

_bearer = HTTPBearer(auto_error=False)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """One session per request, committed on success.

    Commit happens here rather than in each router so that a handler which
    raises after a partial write cannot leave half a transaction behind — the
    audit entry and the thing it describes commit together or not at all.
    """
    factory = get_sessionmaker()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_clock() -> Clock:
    return SystemClock()


ClockDep = Annotated[Clock, Depends(get_clock)]


# -- repositories ----------------------------------------------------------


def get_user_repository(session: SessionDep, clock: ClockDep) -> UserRepository:
    return UserRepository(session, clock)


def get_audit_repository(session: SessionDep, clock: ClockDep) -> AuditRepository:
    return AuditRepository(session, clock)


def get_custody_repository(session: SessionDep, clock: ClockDep) -> CustodyRepository:
    return CustodyRepository(session, clock)


def get_investigation_repository(session: SessionDep, clock: ClockDep) -> InvestigationRepository:
    return InvestigationRepository(session, clock)


def get_lifecycle_repository(session: SessionDep, clock: ClockDep) -> LifecycleRepository:
    return LifecycleRepository(session, clock)


def get_pipeline_repository(session: SessionDep, clock: ClockDep) -> PipelineRepository:
    return PipelineRepository(session, clock)


def get_retention_repository(session: SessionDep, clock: ClockDep) -> RetentionRepository:
    return RetentionRepository(session, clock)


def get_review_repository(session: SessionDep, clock: ClockDep) -> ReviewRepository:
    return ReviewRepository(session, clock)


def get_fact_repository(session: SessionDep, clock: ClockDep) -> FactRepository:
    return FactRepository(session, clock)


def get_image_repository(session: SessionDep, clock: ClockDep) -> ImageRepository:
    return ImageRepository(session, clock)


@lru_cache(maxsize=4)
def _object_store_for(root: str) -> FilesystemObjectStore:
    """One store instance per root. Cached because constructing it creates the
    directory, and doing that on every request is pointless syscall traffic."""
    return FilesystemObjectStore(root)


def get_object_store(settings: SettingsDep) -> ObjectStore:
    return _object_store_for(str(settings.data_dir / "objects"))


def get_observation_repository(session: SessionDep, clock: ClockDep) -> ObservationRepository:
    return ObservationRepository(session, clock)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """The factory itself, for handlers that outlive their request scope.

    The SSE stream is the only current caller: its generator keeps producing
    after the handler returns, by which point the request-scoped session is
    closed. Overridable in tests alongside `get_db_session`.
    """
    return get_sessionmaker()


UserRepoDep = Annotated[UserRepository, Depends(get_user_repository)]
PipelineRepoDep = Annotated[PipelineRepository, Depends(get_pipeline_repository)]
RetentionRepoDep = Annotated[RetentionRepository, Depends(get_retention_repository)]
ReviewRepoDep = Annotated[ReviewRepository, Depends(get_review_repository)]
FactRepoDep = Annotated[FactRepository, Depends(get_fact_repository)]
ImageRepoDep = Annotated[ImageRepository, Depends(get_image_repository)]
ObservationRepoDep = Annotated[
    ObservationRepository, Depends(get_observation_repository)
]
ObjectStoreDep = Annotated[ObjectStore, Depends(get_object_store)]
SessionFactoryDep = Annotated[
    "async_sessionmaker[AsyncSession]", Depends(get_session_factory)
]
AuditRepoDep = Annotated[AuditRepository, Depends(get_audit_repository)]
CustodyRepoDep = Annotated[CustodyRepository, Depends(get_custody_repository)]
InvestigationRepoDep = Annotated[
    InvestigationRepository, Depends(get_investigation_repository)
]
LifecycleRepoDep = Annotated[LifecycleRepository, Depends(get_lifecycle_repository)]


# -- authentication --------------------------------------------------------


async def get_current_user(
    settings: SettingsDep,
    users: UserRepoDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> User:
    from .security import decode_access_token, decode_imatch_token

    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Bearer token required.")

    token = credentials.credentials

    # Two issuers are accepted. IIE's own token is tried first because it is the
    # common case for direct API use; an iMATCH workspace token is tried second
    # so investigators already signed into the workspace are not asked to sign
    # in again. Federation is off unless IIE_IMATCH_JWT_SECRET is configured.
    try:
        payload = decode_access_token(token, settings)
    except AuthenticationError:
        if not settings.imatch_federation_enabled:
            raise
        identity = decode_imatch_token(token, settings)
        user = await users.get_or_create_federated(
            external_subject=identity.subject_id,
            external_tenant=identity.tenant_id,
            role=identity.role,
        )
        if not user.is_active:
            raise AuthenticationError("This account is disabled.") from None
        return user

    subject = payload.get("sub")
    if not subject:
        raise AuthenticationError("Token is missing a subject claim.")

    try:
        user_id = uuid.UUID(str(subject))
    except ValueError as exc:
        raise AuthenticationError("Token subject is not a valid identifier.") from exc

    user = await users.get(user_id)
    if not user.is_active:
        raise AuthenticationError("This account is disabled.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def client_label(request: Request, user: User | None = None) -> str:
    """Actor label for the audit trail.

    Falls back to the client address for unauthenticated actions — a refused
    login still needs an actor recorded, or the audit trail has a hole exactly
    where an intrusion attempt would sit.
    """
    if user is not None:
        return f"{user.email} ({user.id})"
    host = request.client.host if request.client else "unknown"
    return f"anonymous@{host}"


__all__ = [
    "AuditRepoDep",
    "ClockDep",
    "CurrentUser",
    "CustodyRepoDep",
    "FactRepoDep",
    "ImageRepoDep",
    "InvestigationRepoDep",
    "LifecycleRepoDep",
    "ObjectStoreDep",
    "ObservationRepoDep",
    "PipelineRepoDep",
    "RetentionRepoDep",
    "ReviewRepoDep",
    "SessionDep",
    "SessionFactoryDep",
    "SettingsDep",
    "UserRepoDep",
    "client_label",
    "get_current_user",
    "get_db_session",
    "get_object_store",
]
