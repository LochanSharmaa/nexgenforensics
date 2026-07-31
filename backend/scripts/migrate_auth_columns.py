#!/usr/bin/env python
"""
Add the authentication columns to an existing `users` table.

    python backend/scripts/migrate_auth_columns.py

SQLModel's `create_all` only creates tables that do not exist yet; it never
alters one that does. Without this, an existing database keeps the old `users`
shape and every query touching a new column fails at runtime with
"no such column" -- and it fails on the login path, so the whole product is
down rather than degraded.

Idempotent: existing columns are skipped, so it is safe to run repeatedly and
safe to run as part of a deploy step.

EXISTING ACCOUNTS ARE BACKFILLED AS VERIFIED. They were created by an
administrator through bootstrap_admin.py or the supervisor-only endpoint, which
means a human with authority already vouched for the address. Defaulting them
to unverified would lock every current user out of a system they legitimately
use, to enforce a check that was never part of the bargain when their account
was made. Only self-service registrations start unverified.
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

# (column, SQL type, default clause). Kept as plain DDL rather than an Alembic
# revision because this project has no migration chain -- introducing one for a
# single additive change would be more machinery than the change deserves.
COLUMNS: list[tuple[str, str]] = [
    ("email_verified", "BOOLEAN NOT NULL DEFAULT 0"),
    ("otp_hash", "VARCHAR"),
    ("otp_expires_at", "TIMESTAMP"),
    ("otp_attempts", "INTEGER NOT NULL DEFAULT 0"),
    ("otp_sent_count", "INTEGER NOT NULL DEFAULT 0"),
    ("otp_window_started_at", "TIMESTAMP"),
    ("reset_token_hash", "VARCHAR"),
    ("reset_token_expires_at", "TIMESTAMP"),
    ("failed_login_attempts", "INTEGER NOT NULL DEFAULT 0"),
    ("locked_until", "TIMESTAMP"),
    ("refresh_token_hash", "VARCHAR"),
    ("refresh_token_expires_at", "TIMESTAMP"),
    ("session_epoch", "INTEGER NOT NULL DEFAULT 0"),
    ("last_login_ip", "VARCHAR NOT NULL DEFAULT ''"),
    ("updated_at", "TIMESTAMP"),
]


def existing_columns(conn, table: str = "users") -> set[str]:
    if conn.dialect.name == "sqlite":
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        return {row[1] for row in rows}
    rows = conn.execute(text(
        "SELECT column_name FROM information_schema.columns WHERE table_name = :t"
    ), {"t": table}).fetchall()
    return {row[0] for row in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-existing", action="store_true", default=True,
                    help="mark pre-existing accounts as email-verified (default)")
    ap.add_argument("--no-verify-existing", dest="verify_existing", action="store_false")
    args = ap.parse_args()

    settings = get_settings()
    url = normalise_database_url(settings.database_url)
    engine = build_engine(settings)
    print("=" * 72)
    print("  users table migration")
    print(f"  database: {url.split('@')[-1]}")
    print("=" * 72)

    added, skipped = [], []
    with engine.begin() as conn:
        present = existing_columns(conn)
        if not present:
            print("  no `users` table yet; nothing to migrate "
                  "(create_all will build it with the new shape)")
            return 0

        for name, ddl in COLUMNS:
            if name in present:
                skipped.append(name)
                continue
            conn.execute(text(f"ALTER TABLE users ADD COLUMN {name} {ddl}"))
            added.append(name)

        if added:
            conn.execute(text("UPDATE users SET updated_at = created_at WHERE updated_at IS NULL"))
        if args.verify_existing and "email_verified" in added:
            result = conn.execute(text("UPDATE users SET email_verified = 1"))
            print(f"  backfilled {result.rowcount} pre-existing account(s) as verified")

    print(f"  added   ({len(added)}): {', '.join(added) or 'none'}")
    print(f"  skipped ({len(skipped)}): {', '.join(skipped) or 'none'}")
    print("\n  done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
