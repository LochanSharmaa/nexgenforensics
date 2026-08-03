"""The migrations and the models must build the same schema.

Every other test in this suite builds its tables from `Base.metadata`. No
deployment ever does — deployments run `alembic upgrade head`. Wherever the two
disagree, the suite is green and production is broken, and the failure surfaces
as a request that dies on a constraint nobody remembered was there.

That is not hypothetical. `ObservationMethod.VISION` was added to the model when
the vision layer was built and never reached the CHECK constraint on
`observations.method`, so the entire image-reading stage failed on its first
call against a migrated database while the tests stayed green. Revision
`b8f0d1c94a27` fixed that instance; these tests are here so the next one costs a
failing test rather than a broken feature.

Run against SQLite, like the rest of the suite. What is being compared is the
vocabulary and the table set — both dialect-independent — so a gap found here is
a gap on PostgreSQL too.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from database.base import Base

ROOT = Path(__file__).resolve().parents[2]

# `CONSTRAINT <name> CHECK (<column> IN ('A', 'B'))` — the form SQLAlchemy emits
# for an `Enum(native_enum=False, create_constraint=True)` column.
_CHECK = re.compile(r"CONSTRAINT (\w+) CHECK \(([\w.]+) IN \(([^)]*)\)\)", re.I)


def _build_with_migrations(path: Path) -> None:
    """A deployment's schema: alembic upgrade head.

    Run as a subprocess because `database/migrations/env.py` reads the database
    URL from `get_settings()` at import time, and that is cached for the life of
    the process. A separate process with its own environment is both simpler
    than defeating the cache and closer to how migrations actually run.
    """
    env = {**os.environ, "IIE_DATABASE_URL": f"sqlite+aiosqlite:///{path.as_posix()}"}
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        pytest.fail(f"alembic upgrade head failed:\n{result.stdout}\n{result.stderr}")


def _build_with_models(path: Path) -> None:
    """The test suite's schema: create_all from the declarative metadata."""
    Base.metadata.create_all(create_engine(f"sqlite:///{path.as_posix()}"))


def _tables(path: Path) -> dict[str, str]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
        ).fetchall()
    finally:
        connection.close()
    return {name: re.sub(r"\s+", " ", sql) for name, sql in rows}


def _vocabularies(path: Path) -> dict[str, set[str]]:
    """Every controlled vocabulary in the schema, keyed by `table.column`."""
    found: dict[str, set[str]] = {}
    for table, sql in _tables(path).items():
        for _name, column, values in _CHECK.findall(sql):
            found[f"{table}.{column}"] = {
                value.strip().strip("'") for value in values.split(",") if value.strip()
            }
    return found


@pytest.fixture(scope="module")
def schemas(tmp_path_factory) -> tuple[Path, Path]:  # noqa: ANN001
    """One database built each way. Module-scoped — running the full migration
    chain per test would dominate the suite's runtime for no extra coverage."""
    directory = tmp_path_factory.mktemp("schemas")
    migrated, declared = directory / "migrated.db", directory / "declared.db"
    _build_with_migrations(migrated)
    _build_with_models(declared)
    return migrated, declared


def test_migrations_and_models_declare_the_same_tables(schemas) -> None:  # noqa: ANN001
    migrated, declared = schemas
    assert set(_tables(migrated)) == set(_tables(declared)), (
        "A table exists in one build path and not the other — most likely a model "
        "added without a migration. Run `alembic revision --autogenerate`."
    )


def test_no_vocabulary_is_narrower_in_the_migrated_schema(schemas) -> None:  # noqa: ANN001
    """The failure mode that shipped: code produces a value the database refuses.

    Asserted in one direction deliberately. A migrated database allowing a value
    the models no longer use is stale but harmless — the code cannot write it. A
    migrated database *rejecting* a value the models allow is an outage in
    whichever feature produces that value.
    """
    migrated, declared = schemas
    from_migrations, from_models = _vocabularies(migrated), _vocabularies(declared)

    rejected = {
        column: sorted(allowed - from_migrations.get(column, set()))
        for column, allowed in from_models.items()
        if allowed - from_migrations.get(column, set())
    }
    assert not rejected, (
        "The migrated schema rejects values the models can produce: "
        f"{rejected}. Every write of one of these fails with an IntegrityError in "
        "deployment while passing every test that builds its tables from metadata. "
        "Add a migration that widens the CHECK constraint."
    )


def test_vision_is_a_writable_observation_method(schemas) -> None:
    """The specific regression, named so a failure points straight at the cause."""
    migrated, _ = schemas
    assert "VISION" in _vocabularies(migrated)["observations.method"], (
        "observations.method rejects VISION, so the image-reading stage cannot "
        "store a single observation. See revision b8f0d1c94a27."
    )
