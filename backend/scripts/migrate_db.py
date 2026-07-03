from __future__ import annotations

import json
import sys
from pathlib import Path

from nexgen_engine.settings import Settings
from nexgen_engine.storage import Database


REQUIRED_TABLES = {
    "tenants",
    "users",
    "enrolled_identities",
    "biometric_templates",
    "audit_logs",
    "jobs",
    "model_versions",
    "index_versions",
}


def main() -> int:
    settings = Settings.from_env()
    if settings.database_url.startswith("sqlite:///"):
        db_path = settings.database_url.replace("sqlite:///", "", 1)
    elif settings.database_url.startswith("postgresql://"):
        print(
            json.dumps(
                {
                    "status": "not_run",
                    "reason": "Postgres URL configured. Use production migration tooling/container with psycopg installed.",
                    "database_url": settings.database_url,
                },
                indent=2,
            )
        )
        return 0
    else:
        db_path = str(settings.runtime_dir / "nexgen.db")
    db = Database(Path(db_path))
    db.migrate()
    with db.connect() as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = {row["name"] for row in rows}
    missing = sorted(REQUIRED_TABLES - tables)
    print(json.dumps({"status": "ok" if not missing else "failed", "missing": missing, "tables": sorted(tables)}, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
