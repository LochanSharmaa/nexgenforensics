from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TenantRecord:
    tenant_id: str
    name: str


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    tenant_id: str
    email: str
    role: str
    password_hash: str


class Database:
    def __init__(self, path: str | Path = "runtime/nexgen.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def migrate(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                  tenant_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS users (
                  user_id TEXT PRIMARY KEY,
                  tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
                  email TEXT NOT NULL UNIQUE,
                  role TEXT NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS enrolled_identities (
                  identity_id TEXT NOT NULL,
                  tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                  PRIMARY KEY(identity_id, tenant_id)
                );
                CREATE TABLE IF NOT EXISTS biometric_templates (
                  template_id TEXT PRIMARY KEY,
                  identity_id TEXT NOT NULL,
                  tenant_id TEXT NOT NULL,
                  dimensions INTEGER NOT NULL,
                  encrypted_payload TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS audit_logs (
                  audit_id TEXT PRIMARY KEY,
                  tenant_id TEXT NOT NULL,
                  actor_id TEXT NOT NULL,
                  action TEXT NOT NULL,
                  resource TEXT NOT NULL,
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS jobs (
                  job_id TEXT PRIMARY KEY,
                  tenant_id TEXT NOT NULL,
                  job_type TEXT NOT NULL,
                  status TEXT NOT NULL,
                  payload_json TEXT NOT NULL DEFAULT '{}',
                  error TEXT NOT NULL DEFAULT '',
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS model_versions (
                  version_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  status TEXT NOT NULL,
                  metrics_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS index_versions (
                  version_id TEXT PRIMARY KEY,
                  tenant_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def upsert_tenant(self, tenant_id: str, name: str) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO tenants(tenant_id, name) VALUES(?, ?)", (tenant_id, name))

    def create_user(self, record: UserRecord) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO users(user_id, tenant_id, email, role, password_hash) VALUES(?, ?, ?, ?, ?)",
                (record.user_id, record.tenant_id, record.email, record.role, record.password_hash),
            )

    def get_user_by_email(self, email: str) -> UserRecord | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            return None
        return UserRecord(row["user_id"], row["tenant_id"], row["email"], row["role"], row["password_hash"])

    def create_job(self, job_id: str, tenant_id: str, job_type: str, payload_json: str) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO jobs(job_id, tenant_id, job_type, status, payload_json) VALUES(?, ?, ?, 'queued', ?)",
                (job_id, tenant_id, job_type, payload_json),
            )

    def update_job(self, job_id: str, status: str, error: str = "") -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?",
                (status, error, job_id),
            )

    def job_status(self, job_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None
