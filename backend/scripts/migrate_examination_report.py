#!/usr/bin/env python
"""
Add the columns and table the photograph examination report needs.

    python backend/scripts/migrate_examination_report.py

SQLModel's `create_all` builds tables that do not exist but never alters one
that does, so the two new filename columns have to be added by hand. The
`examination_notes` table is new and `create_all` will build it; this script
creates it too so a single command leaves the database complete either way.

Idempotent: present columns are skipped and the table creation is
IF NOT EXISTS, so it is safe to run repeatedly and safe in a deploy step.

The filename columns backfill as empty, not as the stored hash. An exhibit
whose original name was never captured must be described in the report as
having no recorded filename -- writing the hash into the filename column would
make the report claim a name the submitter never used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from imatch_api.core.config import get_settings  # noqa: E402
from imatch_api.db.session import build_engine, normalise_database_url  # noqa: E402

# (table, column, DDL). Plain DDL rather than an Alembic revision because this
# project has no migration chain; see migrate_auth_columns.py for the precedent.
COLUMNS: list[tuple[str, str, str]] = [
    ("search_runs", "probe_filename", "VARCHAR NOT NULL DEFAULT ''"),
    ("templates", "image_filename", "VARCHAR NOT NULL DEFAULT ''"),
]

EXAMINATION_NOTES = """
CREATE TABLE IF NOT EXISTS examination_notes (
    id VARCHAR NOT NULL PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    case_id VARCHAR NOT NULL,
    section VARCHAR NOT NULL,
    ordinal INTEGER NOT NULL DEFAULT 0,
    body TEXT NOT NULL DEFAULT '',
    marks TEXT NOT NULL DEFAULT '[]',
    author_id VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
)
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_exam_notes_case_section "
    "ON examination_notes (case_id, section)",
    "CREATE INDEX IF NOT EXISTS ix_examination_notes_tenant_id "
    "ON examination_notes (tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_examination_notes_case_id "
    "ON examination_notes (case_id)",
]


def _columns(conn, table: str) -> set[str]:
    if conn.dialect.name == "sqlite":
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        return {row[1] for row in rows}
    rows = conn.execute(
        text("SELECT column_name FROM information_schema.columns WHERE table_name = :t"),
        {"t": table},
    ).fetchall()
    return {row[0] for row in rows}


def _tables(conn) -> set[str]:
    if conn.dialect.name == "sqlite":
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type = 'table'")
        ).fetchall()
        return {row[0] for row in rows}
    rows = conn.execute(
        text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    ).fetchall()
    return {row[0] for row in rows}


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    settings = get_settings()
    url = normalise_database_url(settings.database_url)
    engine = build_engine(settings)
    print("=" * 72)
    print("  photograph examination report migration")
    print(f"  database: {url.split('@')[-1]}")
    print("=" * 72)

    added: list[str] = []
    skipped: list[str] = []

    with engine.begin() as conn:
        present_tables = _tables(conn)

        for table, name, ddl in COLUMNS:
            if table not in present_tables:
                skipped.append(f"{table}.{name} (no table yet)")
                continue
            if name in _columns(conn, table):
                skipped.append(f"{table}.{name}")
                continue
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
            added.append(f"{table}.{name}")

        if "examination_notes" in present_tables:
            skipped.append("examination_notes (table)")
        else:
            conn.execute(text(EXAMINATION_NOTES))
            added.append("examination_notes (table)")
        for statement in INDEXES:
            conn.execute(text(statement))

    print(f"  added   ({len(added)}): {', '.join(added) or 'none'}")
    print(f"  skipped ({len(skipped)}): {', '.join(skipped) or 'none'}")
    print("\n  done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
