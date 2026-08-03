"""Alembic environment.

Migrations run synchronously even though the application is async — Alembic's
migration context is sync, and there is no benefit to driving DDL through an
event loop. `Settings.alembic_url` strips the async driver for exactly this.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# The project root, so `shared` and `database` import when Alembic is invoked
# from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from database import models  # noqa: E402,F401  (import registers the tables)
from database.base import Base  # noqa: E402
from shared.config import get_settings  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.alembic_url)

target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to) -> bool:  # noqa: ANN001
    """Keep autogenerate focused on our own tables."""
    return not (type_ == "table" and name in {"alembic_version", "spatial_ref_sys"})


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        # SQLite cannot ALTER most things in place; batch mode rewrites the
        # table instead. Harmless on PostgreSQL, essential for local test runs.
        render_as_batch=settings.is_sqlite,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
            render_as_batch=settings.is_sqlite,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
