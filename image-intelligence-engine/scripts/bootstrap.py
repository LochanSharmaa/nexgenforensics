"""First-run setup: seed the local investigator account.

Local-first deployments still authenticate (ARCHITECTURE §3.1), so the stack
needs one account to exist before anything can be done through the API. This
generates a strong password, prints it **once**, and never stores it in
plaintext.

    python -m scripts.bootstrap
    python -m scripts.bootstrap --email me@example.com

Idempotent: if the account already exists it reports that and changes nothing,
so it is safe to run from a container entrypoint on every start.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import string
import sys

from database.repositories import AuditRepository, UserRepository
from database.session import dispose_engine, session_scope
from shared.config import get_settings
from shared.logging import configure_logging, get_logger

logger = get_logger(__name__)

ALPHABET = string.ascii_letters + string.digits + "-_"


def generate_password(length: int = 24) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


async def bootstrap(email: str, password: str | None, display_name: str) -> int:
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt="console")

    from api.security import hash_password

    async with session_scope() as session:
        users = UserRepository(session)
        audit = AuditRepository(session)

        existing = await users.get_by_email(email)
        if existing is not None:
            print(f"Account {email} already exists — nothing to do.")
            return 0

        total = await users.count()
        secret = password or generate_password()
        user = await users.create(
            email=email,
            password_hash=hash_password(secret),
            display_name=display_name,
            role="admin" if total == 0 else "investigator",
        )
        await audit.record(
            action="user.bootstrap",
            outcome="created",
            actor_id=user.id,
            actor_label=f"bootstrap:{email}",
            detail={"role": user.role, "first_account": total == 0},
        )

    if password is None:
        print("=" * 68)
        print(f"  Account created: {email}")
        print(f"  Password:        {secret}")
        print("")
        print("  This password is shown once and is not recoverable.")
        print("  Store it in a password manager now.")
        print("=" * 68)
    else:
        print(f"Account created: {email} (password supplied by caller)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the local investigator account.")
    parser.add_argument("--email", default="investigator@localhost.localdomain")
    parser.add_argument("--display-name", default="Local Investigator")
    parser.add_argument(
        "--password",
        default=None,
        help="Optional. Omit to have a strong password generated and shown once.",
    )
    args = parser.parse_args()

    try:
        return asyncio.run(
            bootstrap(args.email.strip().lower(), args.password, args.display_name)
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Bootstrap failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        asyncio.run(dispose_engine())


if __name__ == "__main__":
    raise SystemExit(main())
