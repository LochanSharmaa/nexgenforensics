"""Architecture boundary tests.

A boundary that is not tested is a boundary that will be crossed. These parse
the import graph with `ast` — no imports are executed, so a violation is caught
even in a module that would fail at runtime.

Enforced here:

1. `shared` (domain) imports nothing from outer layers.
2. `database` never imports `api` or `worker`.
3. Routers hold no business logic and construct no infrastructure.
4. No module anywhere imports a face-recognition library.
5. No module imports a concrete provider package directly (plugin isolation).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PACKAGES = (
    "shared", "database", "api", "worker", "crawler", "providers",
    "image_discovery", "page_processing", "entity_extraction", "correlation",
    "confidence", "graph", "timeline", "monitoring", "copilot", "search",
    "reports", "workspace", "retention", "export",
)

FORBIDDEN_FACE_MODULES = frozenset(
    {
        "insightface", "face_recognition", "facenet", "facenet_pytorch",
        "deepface", "arcface", "dlib", "mtcnn", "retinaface", "keras_facenet",
    }
)


def _python_files() -> list[Path]:
    files: list[Path] = []
    for package in SOURCE_PACKAGES:
        directory = PROJECT_ROOT / package
        if directory.is_dir():
            files.extend(
                path
                for path in directory.rglob("*.py")
                if "__pycache__" not in path.parts and "migrations" not in path.parts
            )
    return files


def _imports_of(path: Path) -> set[str]:
    """Top-level module names imported by a file, resolved for relative imports."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # pragma: no cover
        pytest.fail(f"{path} does not parse: {exc}")

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import: resolve against the file's own package.
                parts = path.relative_to(PROJECT_ROOT).parts
                package_parts = list(parts[:-1])
                for _ in range(node.level - 1):
                    if package_parts:
                        package_parts.pop()
                if package_parts:
                    modules.add(package_parts[0])
            elif node.module:
                modules.add(node.module.split(".")[0])
    return modules


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


# -- 1. domain purity ------------------------------------------------------


def test_shared_does_not_import_outer_layers():
    """`shared` holds the domain rules. They must be testable with zero
    infrastructure, which means they cannot reach into it."""
    outer = {"api", "worker", "database", "crawler", "providers", "reports"}
    violations = []

    for path in (PROJECT_ROOT / "shared").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        leaked = _imports_of(path) & outer
        if leaked:
            violations.append(f"{_relative(path)} imports {sorted(leaked)}")

    assert not violations, "Domain layer reaches outward:\n  " + "\n  ".join(violations)


def test_shared_avoids_infrastructure_libraries():
    """No ORM, no HTTP client, no cloud SDK in the domain layer."""
    infrastructure = {"sqlalchemy", "alembic", "httpx", "boto3", "redis", "arq", "fastapi"}
    violations = []

    for path in (PROJECT_ROOT / "shared").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        leaked = _imports_of(path) & infrastructure
        if leaked:
            violations.append(f"{_relative(path)} imports {sorted(leaked)}")

    assert not violations, (
        "Domain layer depends on infrastructure:\n  " + "\n  ".join(violations)
    )


# -- 2. persistence independence -------------------------------------------


def test_database_does_not_import_interface_layer():
    outer = {"api", "worker"}
    violations = []

    for path in (PROJECT_ROOT / "database").rglob("*.py"):
        if "__pycache__" in path.parts or "migrations" in path.parts:
            continue
        leaked = _imports_of(path) & outer
        if leaked:
            violations.append(f"{_relative(path)} imports {sorted(leaked)}")

    assert not violations, "Persistence depends on interface:\n  " + "\n  ".join(violations)


# -- 3. thin routers -------------------------------------------------------


def test_routers_do_not_construct_infrastructure():
    """Routers receive collaborators through dependencies.

    A router that builds a session or an engine itself is the first step toward
    logic only the HTTP layer can run, which then cannot be reused by the worker
    or tested without a request.
    """
    forbidden = {"sqlalchemy", "asyncpg", "redis", "boto3", "httpx", "arq"}
    routers = PROJECT_ROOT / "api" / "routers"
    violations = []

    for path in routers.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        leaked = _imports_of(path) & forbidden
        if leaked:
            violations.append(f"{_relative(path)} imports {sorted(leaked)}")

    assert not violations, (
        "Routers construct infrastructure directly:\n  " + "\n  ".join(violations)
    )


# -- 4. the prohibition ----------------------------------------------------


def test_no_module_imports_a_face_recognition_library():
    """ARCHITECTURE §14. The lockfile guard catches installation; this catches
    an import added before the dependency is declared."""
    violations = []

    for path in _python_files():
        leaked = _imports_of(path) & FORBIDDEN_FACE_MODULES
        if leaked:
            violations.append(f"{_relative(path)} imports {sorted(leaked)}")

    assert not violations, (
        "Facial-recognition library imported. This platform must never identify "
        "people from facial features (ARCHITECTURE §1.1, §14):\n  "
        + "\n  ".join(violations)
    )


def test_no_vector_index_dependency():
    """No ANN index anywhere: a face gallery must have nowhere to live."""
    ann = {"faiss", "annoy", "hnswlib", "pgvector", "chromadb", "qdrant_client"}
    violations = []

    for path in _python_files():
        leaked = _imports_of(path) & ann
        if leaked:
            violations.append(f"{_relative(path)} imports {sorted(leaked)}")

    assert not violations, (
        "Vector index dependency found. The schema declares no embeddings and no "
        "face gallery may be constructible:\n  " + "\n  ".join(violations)
    )


# -- 5. plugin isolation ---------------------------------------------------


def test_no_module_imports_a_concrete_provider():
    """Adding a provider must not require changes outside its own directory
    (ARCHITECTURE §15). Everything else talks to the registry."""
    providers_dir = PROJECT_ROOT / "providers"
    if not providers_dir.is_dir():
        pytest.skip("providers package arrives in Phase 7")

    concrete = {
        entry.name
        for entry in providers_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(("_", "."))
    }
    if not concrete:
        pytest.skip("no concrete providers yet")

    violations = []
    for path in _python_files():
        if path.is_relative_to(providers_dir):
            continue
        source = path.read_text(encoding="utf-8")
        for name in concrete:
            if f"providers.{name}" in source:
                violations.append(f"{_relative(path)} references providers.{name}")

    assert not violations, (
        "Concrete provider imported outside the plugin package:\n  " + "\n  ".join(violations)
    )


# -- meta ------------------------------------------------------------------


def test_source_tree_is_discoverable():
    """Guards that silently scan nothing are worse than no guards."""
    files = _python_files()
    assert len(files) >= 10, f"Only {len(files)} source files found; the scan is misconfigured."
