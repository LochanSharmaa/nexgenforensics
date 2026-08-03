from __future__ import annotations

import base64
import os
import secrets
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

# Configure the environment before anything imports the settings singleton.
os.environ.setdefault("NEXGEN_ENV", "test")
os.environ.setdefault("NEXGEN_JWT_SECRET", secrets.token_urlsafe(48))
os.environ.setdefault("NEXGEN_TEMPLATE_KEY", base64.b64encode(secrets.token_bytes(32)).decode())
os.environ.setdefault("NEXGEN_REQUIRE_LAWFUL_BASIS", "true")

from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import Session, SQLModel  # noqa: E402

from imatch_api.core.config import get_settings  # noqa: E402
from imatch_api.core.dependencies import reset_limiters  # noqa: E402
from imatch_api.core.security import hash_password  # noqa: E402
from imatch_api.db.models import Role, Tenant, User  # noqa: E402
from imatch_api.db.session import build_engine, set_engine  # noqa: E402
from imatch_api.services.engine_service import EngineService, set_engine_service  # noqa: E402
from imatch_api.services.enhancement_service import set_enhancement_service  # noqa: E402
from nexgen_engine.models.arcface import EngineUnavailableError  # noqa: E402
from nexgen_engine.runtime import EngineRuntime  # noqa: E402

TEST_PASSWORD = "Correct-Horse-Battery-9!"

# Real faces. There is no synthetic substitute: a drawn oval is not a face, the
# detector correctly refuses it, and a test that passed on one would be testing
# nothing. Tests needing faces skip when this is absent.
AGEDB = REPO_ROOT / "src_extracted" / "AgeDB" / "AgeDB"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------- engine -----


@pytest.fixture(scope="session")
def engine_runtime() -> EngineRuntime:
    """One set of loaded models, shared by every test in the session.

    Loading the pack costs several seconds and hundreds of megabytes; doing it
    per test would make the suite unusable.
    """
    runtime = EngineRuntime(get_settings().engine_config())
    try:
        runtime.warm_up()
    except EngineUnavailableError as exc:
        pytest.skip(f"Recognition model unavailable: {exc}")
    return runtime


@pytest.fixture(scope="session")
def face_paths(engine_runtime) -> dict[str, list[Path]]:
    """AgeDB identities whose images the detector can actually use.

    Grouped from ``<index>_<Name>_<age>_<sex>.jpg``, then *verified*: an
    identity is only offered to tests if at least two of its photographs
    genuinely produce a template. Some images in any real dataset are
    unusable -- profile shots, heavy occlusion, motion blur -- and a test that
    happened to draw one would fail for reasons that have nothing to do with
    the behaviour it is checking.
    """
    if not AGEDB.is_dir():
        pytest.skip(f"Real face images required. Expected AgeDB at {AGEDB}.")

    from nexgen_engine.inference.pipeline import FacialRecognitionPipeline

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(AGEDB.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = path.stem.split("_")
        if len(parts) >= 3:
            grouped[parts[1]].append(path)

    pipeline = FacialRecognitionPipeline(engine_runtime.config, engine_runtime)
    verified: dict[str, list[Path]] = {}

    for name, paths in grouped.items():
        if len(paths) < 2:
            continue
        usable = []
        for path in paths[:4]:
            try:
                pipeline.encode_bytes(path.read_bytes())
                usable.append(path)
            except Exception:
                continue
            if len(usable) == 3:
                break
        if len(usable) >= 2:
            verified[name] = usable
        if len(verified) >= 25:
            break

    if len(verified) < 5:
        pytest.skip(f"Only {len(verified)} identities produced usable templates.")
    return verified


@pytest.fixture(scope="session")
def face_bytes(face_paths) -> list[bytes]:
    """A handful of real face images as raw bytes."""
    return [paths[0].read_bytes() for paths in list(face_paths.values())[:5]]


@pytest.fixture(scope="session")
def face_b64(face_bytes) -> list[str]:
    return [base64.b64encode(payload).decode("ascii") for payload in face_bytes]


# ------------------------------------------------------------ environment ---


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request):
    """Give every test its own database, storage root, and audit log."""
    monkeypatch.setenv("NEXGEN_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("NEXGEN_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("NEXGEN_AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("NEXGEN_ENHANCEMENT_CACHE_ROOT", str(tmp_path / "enhancement_cache"))

    get_settings.cache_clear()
    reset_limiters()

    engine = build_engine(get_settings())
    SQLModel.metadata.create_all(engine)
    set_engine(engine)

    # Reuse the session-scoped models when the test asked for them, so the
    # gallery is fresh but the weights are not reloaded.
    runtime = request.getfixturevalue("engine_runtime") if "engine_runtime" in request.fixturenames else None
    if runtime is not None:
        set_engine_service(EngineService(get_settings(), runtime=runtime))

    # The enhancement service caches Settings at construction; a stale instance
    # would write cache entries into a previous test's tmp_path.
    set_enhancement_service(None)

    yield

    set_enhancement_service(None)
    set_engine_service(None)
    set_engine(None)
    engine.dispose()
    get_settings.cache_clear()
    reset_limiters()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def engine_service(engine_runtime) -> EngineService:
    from imatch_api.services.engine_service import get_engine_service

    return get_engine_service()


def prime_csrf(test_client: TestClient) -> TestClient:
    """Give a client a CSRF token, exactly as the browser app does.

    TestClient keeps a cookie jar, so the GET sets the cookie; the header is
    pinned so every later request double-submits it. Without this, every
    unauthenticated POST is correctly refused with 403 -- which is the point of
    the guard, and is asserted directly in test_security_headers.py rather than
    implied by unrelated tests failing.
    """
    from imatch_api.core.csrf import CSRF_HEADER

    token = test_client.get("/api/auth/csrf").json()["csrf_token"]
    test_client.headers.update({CSRF_HEADER: token})
    return test_client


@pytest.fixture
def client(engine_runtime) -> TestClient:
    """API client backed by the real recognition engine."""
    from imatch_api.main import create_app

    with TestClient(create_app()) as test_client:
        yield prime_csrf(test_client)


@pytest.fixture
def anon_client() -> TestClient:
    """Client for tests that never touch recognition (auth, cases, audit)."""
    from imatch_api.main import create_app

    with TestClient(create_app()) as test_client:
        yield prime_csrf(test_client)


@pytest.fixture
def raw_client() -> TestClient:
    """Client with NO CSRF token, for asserting the guard actually refuses."""
    from imatch_api.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


# --------------------------------------------------------------- factories --


@pytest.fixture
def tenant_factory():
    from imatch_api.db.session import get_engine

    def _create(slug: str = "test-tenant") -> Tenant:
        with Session(get_engine()) as session:
            tenant = Tenant(slug=slug, name=slug.title())
            session.add(tenant)
            session.commit()
            session.refresh(tenant)
            return tenant

    return _create


@pytest.fixture
def user_factory():
    from imatch_api.db.session import get_engine

    def _create(
        tenant: Tenant,
        email: str = "investigator@example.com",
        role: Role = Role.INVESTIGATOR,
        password: str = TEST_PASSWORD,
    ) -> User:
        with Session(get_engine()) as session:
            user = User(
                tenant_id=tenant.id,
                email=email,
                full_name=email.split("@")[0].title(),
                password_hash=hash_password(password),
                role=role,
                # Fixture users stand in for admin-created accounts, which are
                # verified on creation.
                email_verified=True,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user

    return _create


@pytest.fixture
def auth_headers(request):
    def _login(email: str = "investigator@example.com", password: str = TEST_PASSWORD) -> dict[str, str]:
        active = (
            request.getfixturevalue("client")
            if "client" in request.fixturenames
            else request.getfixturevalue("anon_client")
        )
        response = active.post("/api/auth/login", json={"email": email, "password": password})
        assert response.status_code == 200, response.text
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    return _login


# ------------------------------------------------------------ image helpers --


def flat_image(colour: tuple[int, int, int] = (128, 128, 128), size: int = 300) -> Image.Image:
    """A featureless image. Contains no face, and the detector must say so."""
    return Image.new("RGB", (size, size), colour)


def noise_image(seed: int = 0, size: int = 300) -> Image.Image:
    """Random noise: not a face, used to confirm detection rejects it."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8), mode="RGB")


def image_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=fmt, quality=95)
    return buffer.getvalue()


def image_b64(image: Image.Image, fmt: str = "JPEG") -> str:
    return base64.b64encode(image_bytes(image, fmt)).decode("ascii")
