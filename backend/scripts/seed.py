"""Create the first tenant and administrator.

Run once against a fresh database::

    cd backend
    python scripts/seed.py

Reads NEXGEN_SEED_TENANT, NEXGEN_SEED_ADMIN_EMAIL, and NEXGEN_SEED_ADMIN_PASSWORD.
If no password is set, a strong one is generated and printed -- it is shown once
and never stored in recoverable form, and re-running leaves existing records
alone. If a password IS set, re-running resets the account to that password, so
the configured credential always works.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlmodel import select  # noqa: E402

from imatch_api.core.config import get_settings  # noqa: E402
from imatch_api.core.security import hash_password, validate_password_strength  # noqa: E402
from imatch_api.db.models import Role, Tenant, User  # noqa: E402
from imatch_api.db.session import init_database, session_scope  # noqa: E402


def generate_password() -> str:
    alphabet = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789-_!@#$%"
    while True:
        candidate = "".join(secrets.choice(alphabet) for _ in range(20))
        try:
            validate_password_strength(candidate)
            return candidate
        except ValueError:
            continue


def main() -> int:
    settings = get_settings()
    init_database()

    slug = settings.seed_tenant.strip().lower()
    email = settings.seed_admin_email.strip().lower()
    password = settings.seed_admin_password.strip()
    generated = False

    if not password:
        password = generate_password()
        generated = True
    else:
        try:
            validate_password_strength(password)
        except ValueError as exc:
            print(f"NEXGEN_SEED_ADMIN_PASSWORD is too weak: {exc}", file=sys.stderr)
            return 1

    with session_scope() as session:
        tenant = session.exec(select(Tenant).where(Tenant.slug == slug)).first()
        if tenant is None:
            tenant = Tenant(slug=slug, name=slug.replace("-", " ").title())
            session.add(tenant)
            session.flush()
            print(f"Created tenant {slug!r} ({tenant.id}).")
        else:
            print(f"Tenant {slug!r} already exists ({tenant.id}).")

        existing = session.exec(
            select(User).where(User.tenant_id == tenant.id, User.email == email)
        ).first()
        if existing is not None:
            if generated:
                print(f"User {email!r} already exists; leaving it untouched.")
                return 0
            # An explicitly configured password is a statement of intent: this
            # account must be reachable with exactly this credential (e.g. a
            # demo login that has to survive re-deploys). Converge to it.
            existing.password_hash = hash_password(password)
            existing.email_verified = True
            session.add(existing)
            print(f"User {email!r} already exists; reset to the configured password.")
            return 0

        session.add(
            User(
                tenant_id=tenant.id,
                email=email,
                full_name="Platform Administrator",
                password_hash=hash_password(password),
                role=Role.ADMIN,
                # The operator running this script is vouching for the address,
                # same as a supervisor using POST /auth/users. Without this the
                # seeded admin is rejected at login with "verify your email" --
                # and there is no verified account that could approve it.
                email_verified=True,
            )
        )

    print(f"Created administrator {email!r} with the admin role.")
    if generated:
        print("")
        print("  Generated password (shown once, store it now):")
        print(f"      {password}")
        print("")
    print("Sign in at POST /api/auth/login, then change this password.")

    if not settings.template_key.strip():
        print("")
        print("WARNING: NEXGEN_TEMPLATE_KEY is unset. Templates enrolled now become")
        print("         unreadable when the process restarts. Set a persistent key first:")
        print('         python -c "import base64,os; print(base64.b64encode(os.urandom(32)).decode())"')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
