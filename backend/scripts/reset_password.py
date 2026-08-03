"""Reset admin password to Admin@1234 directly in the database."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlmodel import Session, select
from imatch_api.core.security import hash_password
from imatch_api.db.models import User
from imatch_api.db.session import get_engine, init_database

init_database()
engine = get_engine()

with Session(engine) as session:
    user = session.exec(select(User).where(User.email == "admin@nexgen.local")).first()
    if user is None:
        print("User not found. Running seed first...")
        sys.exit(1)
    user.password_hash = hash_password("Admin@1234")
    user.email_verified = True
    user.active = True
    session.add(user)
    session.commit()
    print(f"Password reset for {user.email} -> Admin@1234")
    print(f"Role: {user.role}, Active: {user.active}, Verified: {user.email_verified}")
