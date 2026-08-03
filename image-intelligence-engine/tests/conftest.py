"""Shared test fixtures.

Tests run against SQLite so the suite needs no running services. The schema is
portable by design (database/base.py), and PostgreSQL-specific guarantees —
notably the `REVOKE UPDATE, DELETE` on the append-only tables — are covered by
the migration integration test instead.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from api.dependencies import (
    get_clock,
    get_db_session,
    get_object_store,
    get_session_factory,
)
from api.main import create_app
from api.security import hash_password
from database.base import Base
from database.models import User
from database.repositories import UserRepository
from shared.clock import FrozenClock
from shared.config import Settings, reset_settings_cache
from shared.storage import FilesystemObjectStore

TEST_PASSWORD = "investigator-pass-123"


@pytest.fixture(autouse=True, scope="session")
def _isolate_from_developer_configuration():
    """Keep the developer's own `.env` out of the test run.

    `Settings` reads `.env` by default, which is right for the application and
    wrong for a test suite: it makes the result depend on the machine. The
    concrete damage was that once a real `IIE_GEMINI_API_KEY` was configured
    locally, three tests asserting the *unconfigured* behaviour began to fail,
    and — worse — the API-level ones started making live, billable calls to
    Gemini on every run. A suite that spends money and fails depending on who
    runs it is not testing the code.

    The key is cleared from the ambient environment for the same reason. Other
    `IIE_` variables are left alone: CI sets `IIE_ENVIRONMENT=test` deliberately,
    and this fixture's job is to remove accidental influence, not deliberate
    configuration.
    """
    original_env_file = Settings.model_config.get("env_file")
    Settings.model_config["env_file"] = None
    original_key = os.environ.pop("IIE_GEMINI_API_KEY", None)
    reset_settings_cache()
    try:
        yield
    finally:
        Settings.model_config["env_file"] = original_env_file
        if original_key is not None:
            os.environ["IIE_GEMINI_API_KEY"] = original_key
        reset_settings_cache()


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="test",
        database_url="sqlite+aiosqlite:///:memory:",
        secret_key="test-secret-key-at-least-32-characters-long",
        require_lawful_basis=True,
        archive_lookup_enabled=False,
        log_format="console",
        log_level="WARNING",
        metrics_enabled=True,
    )


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(datetime(2026, 8, 3, 12, 0, tzinfo=UTC))


@pytest_asyncio.fixture
async def engine():
    # StaticPool keeps every session on one connection. Without it SQLite gives
    # each connection its own private in-memory database, so the schema created
    # here would be invisible to the next statement.
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def session(session_factory) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session
        await session.commit()


@pytest_asyncio.fixture
async def user(session_factory, clock) -> User:
    """Seeded investigator.

    Uses its own short-lived session and detaches the result. Under StaticPool
    every session shares one connection, so leaving this one open would collide
    with the sessions the app opens during a request.
    """
    async with session_factory() as setup_session:
        users = UserRepository(setup_session, clock)
        created = await users.create(
            email="investigator@example.com",
            password_hash=hash_password(TEST_PASSWORD),
            display_name="Test Investigator",
        )
        await setup_session.commit()
        setup_session.expunge(created)
    return created


@pytest.fixture
def object_store(tmp_path) -> FilesystemObjectStore:
    return FilesystemObjectStore(tmp_path / "objects")


@pytest_asyncio.fixture
async def client(
    settings, session_factory, clock, user, object_store
) -> AsyncIterator[AsyncClient]:
    """An HTTP client wired to the test database and a frozen clock."""
    app = create_app(settings)

    async def _session_override() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    from shared.config import get_settings as real_get_settings

    app.dependency_overrides[get_db_session] = _session_override
    # The SSE stream outlives its request scope and opens its own sessions.
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_clock] = lambda: clock
    app.dependency_overrides[get_object_store] = lambda: object_store
    app.dependency_overrides[real_get_settings] = lambda: settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


@pytest_asyncio.fixture
async def auth_client(client, user) -> AsyncClient:
    """Client carrying a valid bearer token."""
    response = await client.post(
        "/api/v1/auth/token",
        json={"email": user.email, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client
