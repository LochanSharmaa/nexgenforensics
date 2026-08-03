"""Declarative base, shared column types and mixins.

Portability note: models avoid PostgreSQL-only types wherever a portable
equivalent exists, so the same schema runs on SQLite for fast unit tests and on
PostgreSQL in every real deployment. Where a PG-specific type is genuinely
required later (``tsvector``, ``bit(64)``, ``uuid[]``), it is introduced behind a
dialect check rather than by forking the model.

Enum-like columns are ``String`` + ``CheckConstraint`` rather than native
enums, per DATA_MODEL §9: adding a value to a PostgreSQL enum locks the table,
and these vocabularies will grow.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, MetaData, Uuid, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Explicit naming so Alembic autogenerate produces stable, reviewable migration
# names instead of database-assigned ones that differ between environments.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    # `JSON` rather than `JSONB` on PostgreSQL is deliberate: the invariant on
    # `graph_edges` uses `json_array_length()`, which both PostgreSQL and SQLite
    # implement for the `json` type. A JSONB column would need `jsonb_array_length`
    # and the guarantee would stop being testable on SQLite.
    type_annotation_map = {
        uuid.UUID: Uuid(as_uuid=True),
        datetime: DateTime(timezone=True),
        dict[str, Any]: JSON,
        list[str]: JSON,
        list[Any]: JSON,
    }

    def __repr__(self) -> str:
        identifier = getattr(self, "id", None)
        return f"<{self.__class__.__name__} id={identifier}>"


def enum_column(
    values: Iterable[str],
    *,
    constraint_name: str,
    default: str | None = None,
    nullable: bool = False,
    length: int = 48,   # noqa: ARG001 - retained for call-site compatibility
    index: bool = False,
) -> Mapped[str]:
    """A text column constrained to a controlled vocabulary.

    ``native_enum=False`` renders VARCHAR + CHECK on every dialect rather than a
    PostgreSQL ``ENUM`` type. That is the behaviour DATA_MODEL §9 asks for:
    adding a value to a PG enum locks the table, and these vocabularies will
    grow. A plain CHECK is altered without a table rewrite.

    Writing the CHECK by hand is not an option here — a column factory does not
    know its own column name, and SQLAlchemy only substitutes ``%(column_0_name)s``
    into constraint *names*, not into constraint SQL. ``Enum`` resolves the name
    at table-construction time and emits the correct predicate.

    The constraint is what makes the vocabulary a database guarantee rather than
    an application convention: a direct SQL write with a typo is rejected too.
    """
    allowed = tuple(sorted({str(v) for v in values}))
    return mapped_column(
        SAEnum(
            *allowed,
            name=constraint_name,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            length=max(len(value) for value in allowed),
        ),
        nullable=nullable,
        default=default,
        server_default=default,
        index=index,
    )


class TimestampMixin:
    """`created_at` / `updated_at` maintained by the database.

    Server-side defaults rather than Python ones: a row inserted by a migration,
    a maintenance script, or psql still gets a correct timestamp. For a platform
    whose timestamps are evidence, "correct only when written by the app" is not
    good enough.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


__all__ = ["Base", "NAMING_CONVENTION", "TimestampMixin", "enum_column"]
