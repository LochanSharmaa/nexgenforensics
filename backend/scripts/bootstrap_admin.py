#!/usr/bin/env python
"""
Create the initial tenant and operator account for imatch_api.

    python backend/scripts/bootstrap_admin.py --password 'S0me-Strong-Pass!'
    python backend/scripts/bootstrap_admin.py --password ... --role investigator

WHY THIS EXISTS
---------------
`imatch_api.core.config.Settings` declares `seed_tenant`, `seed_admin_email`
and `seed_admin_password`, but nothing in the codebase ever reads them. There
was therefore no way to create the first account: every authenticated endpoint
was unreachable on a fresh database, including the 1:1 verify the console
depends on.

This is deliberately a script rather than startup auto-seeding. Auto-creating
an admin whenever an env var happens to be set means a leaked or inherited
environment silently provisions a privileged account in production. Running it
is an explicit, auditable act.

Idempotent: re-running with an existing email updates nothing and reports the
account already exists. Use --reset-password to rotate the credential, or
--ensure to restore a login that has stopped working -- it resets the password
and clears the three states that block a correct one (unverified address,
deactivated account, brute-force lockout).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from sqlmodel import Session, select  # noqa: E402

from imatch_api.core.config import get_settings  # noqa: E402
from imatch_api.core.security import hash_password  # noqa: E402
from imatch_api.db.models import Role, Tenant, User  # noqa: E402
from imatch_api.db.session import get_engine, init_database  # noqa: E402


def main() -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default=settings.seed_admin_email)
    ap.add_argument("--password", default=settings.seed_admin_password)
    ap.add_argument("--tenant", default=settings.seed_tenant)
    ap.add_argument("--tenant-name", default="NexGen Demo")
    ap.add_argument("--full-name", default="Bootstrap Administrator")
    ap.add_argument("--role", default="admin", choices=[r.value for r in Role])
    ap.add_argument("--reset-password", action="store_true",
                    help="rotate the password of an existing account")
    ap.add_argument("--ensure", action="store_true",
                    help="restore this login: set the password, mark the address "
                         "verified, reactivate, and clear any lockout")
    args = ap.parse_args()

    if not args.password:
        print("A password is required. Pass --password or set NEXGEN_SEED_ADMIN_PASSWORD.")
        return 2
    if len(args.password) < 12:
        print("Refusing to create an account with a password under 12 characters.")
        return 2

    init_database()
    with Session(get_engine()) as session:
        tenant = session.exec(select(Tenant).where(Tenant.slug == args.tenant)).first()
        if tenant is None:
            tenant = Tenant(slug=args.tenant, name=args.tenant_name)
            session.add(tenant)
            session.commit()
            session.refresh(tenant)
            print(f"created tenant  {tenant.slug} ({tenant.id})")
        else:
            print(f"tenant exists   {tenant.slug} ({tenant.id})")

        email = args.email.lower().strip()
        user = session.exec(
            select(User).where(User.tenant_id == tenant.id, User.email == email)
        ).first()

        if user is not None:
            if not (args.reset_password or args.ensure):
                print(f"user exists     {email} (role={user.role}); "
                      "pass --ensure to restore this login")
                return 0
            user.password_hash = hash_password(args.password)
            user.active = True
            # The three states that silently block a correct password: an
            # unconfirmed address, a deactivated account, and a brute-force
            # lockout. --ensure exists to clear all of them in one go, so
            # restoring access never needs a hand-written UPDATE.
            if args.ensure:
                user.email_verified = True
                user.locked_until = None
                user.failed_login_attempts = 0
            session.add(user)
            session.commit()
            print(f"restored login  {email} (role={user.role})")
            return 0

        user = User(
            tenant_id=tenant.id,
            email=email,
            full_name=args.full_name,
            password_hash=hash_password(args.password),
            role=Role(args.role),
            active=True,
            # Bootstrapped by an operator with database access; there is nobody
            # to send a verification code to yet.
            email_verified=True,
        )
        session.add(user)
        session.commit()
        print(f"created user    {email} (role={args.role})")
        print(f"\nLog in with tenant={tenant.slug} email={email}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
