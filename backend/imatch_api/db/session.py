from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from ..core.config import REPO_ROOT, Settings, get_settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None


def _sqlite_path(url: str) -> Path | None:
    if not url.startswith("sqlite"):
        return None
    _, _, tail = url.partition("///")
    if not tail or tail == ":memory:":
        return None
    path = Path(tail)
    # Anchor relative paths to the repository root. A cwd-relative path means
    # `uvicorn` started from the repo root and from backend/ silently open TWO
    # different databases, each with its own users -- and a login that works in
    # one launch style fails with "Incorrect email or password" in the other.
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def normalise_database_url(url: str) -> str:
    """Make a hosted-Postgres URL usable by SQLAlchemy with psycopg 3.

    Managed providers (Render, Heroku, Fly) hand out URLs beginning `postgres://`
    or `postgresql://`. SQLAlchemy maps both to the psycopg **2** driver, which
    this project does not install -- the declared dependency is `psycopg[binary]`,
    which is psycopg **3**. Without this the service fails at startup with
    "No module named 'psycopg2'", and the URL you pasted looks perfectly correct
    while being wrong.

    Rewriting the scheme here rather than asking an operator to hand-edit the
    connection string keeps copy-paste from the provider dashboard working.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def build_engine(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    url = normalise_database_url(settings.database_url)

    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        # FastAPI serves requests on a threadpool, so a connection created on one
        # thread is reused on another. SQLite forbids that unless told otherwise.
        connect_args["check_same_thread"] = False
        target = _sqlite_path(url)
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Re-issue the URL with the anchored absolute path, otherwise the
            # engine still opens the file relative to whatever cwd happens to be.
            url = f"sqlite:///{target.as_posix()}"

    engine = create_engine(
        url,
        echo=False,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _configure_sqlite(dbapi_connection, _record):  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            # SQLite ignores foreign keys unless this is set per connection, which
            # would let subjects and templates outlive a deleted tenant.
            cursor.execute("PRAGMA foreign_keys=ON")
            # WAL lets reads proceed during a write; without it, concurrent
            # searches block behind any enrolment.
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return engine


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine


def set_engine(engine: Engine | None) -> None:
    """Override the process engine. Used by tests to bind a temporary database."""
    global _engine
    _engine = engine


def _quote_default(value: object) -> str | None:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return None


def _add_missing_columns(engine: Engine) -> None:
    """Additive migration for SQLite databases created by an older model set.

    ``create_all`` only creates missing *tables*; a column added to an existing
    model (``search_runs.source_kind`` was the first) silently never reaches a
    database that predates it, and the first SELECT of that model then fails at
    runtime. Only ADD COLUMN is ever issued — nothing is dropped, renamed or
    rewritten — so running this on every startup is safe. Scalar defaults are
    carried into the DDL so existing rows read the model's default instead of
    NULL.
    """
    if engine.url.get_backend_name() != "sqlite":
        return

    with engine.begin() as connection:
        for table in SQLModel.metadata.sorted_tables:
            rows = connection.exec_driver_sql(f"PRAGMA table_info('{table.name}')").fetchall()
            if not rows:
                continue  # new table: create_all handles it
            existing = {row[1] for row in rows}
            for column in table.columns:
                if column.name in existing:
                    continue
                ddl = (
                    f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" '
                    f"{column.type.compile(engine.dialect)}"
                )
                default = None
                if column.default is not None and getattr(column.default, "is_scalar", False):
                    default = _quote_default(column.default.arg)
                if default is not None:
                    ddl += f" DEFAULT {default}"
                connection.exec_driver_sql(ddl)
                logger.info("Migrated %s: added column %s.", table.name, column.name)


def init_database(engine: Engine | None = None) -> None:
    target = engine or get_engine()
    SQLModel.metadata.create_all(target)
    _add_missing_columns(target)
    logger.info("Database schema ready at %s.", target.url.render_as_string(hide_password=True))


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Transactional session for background jobs and scripts."""
    with Session(engine or get_engine()) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def get_session() -> Iterator[Session]:
    """FastAPI dependency.

    Commit is left to the route: an endpoint that raises must not persist a
    partial write just because the dependency tore down cleanly.
    """
    with Session(get_engine()) as session:
        yield session


__all__ = [
    "build_engine",
    "get_engine",
    "get_session",
    "init_database",
    "session_scope",
    "set_engine",
]
