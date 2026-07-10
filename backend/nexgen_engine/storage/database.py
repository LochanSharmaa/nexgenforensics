from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


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
    def __init__(self, path: str | Path = "runtime/nexgen.db", database_url: str | None = None) -> None:
        self.database_url = database_url or _url_from_path(path)
        self.backend = _backend_from_url(self.database_url)
        if self.backend != "sqlite":
            raise RuntimeError(
                "The synchronous Database adapter currently supports SQLite only. "
                "Use AsyncDatabase for postgresql+asyncpg production connections."
            )
        self.path = _sqlite_path(self.database_url)
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
                CREATE TABLE IF NOT EXISTS agency_partitions (
                  agency_id TEXT PRIMARY KEY,
                  tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
                  name TEXT NOT NULL,
                  isolation_policy TEXT NOT NULL DEFAULT 'tenant_scoped',
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS model_accuracy_history (
                  record_id TEXT PRIMARY KEY,
                  version_id TEXT NOT NULL,
                  benchmark_name TEXT NOT NULL,
                  accuracy REAL NOT NULL,
                  false_accept_rate REAL NOT NULL DEFAULT 0,
                  false_reject_rate REAL NOT NULL DEFAULT 0,
                  metrics_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS auto_improve_jobs (
                  job_id TEXT PRIMARY KEY,
                  tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
                  dataset_upload_id TEXT NOT NULL DEFAULT '',
                  status TEXT NOT NULL,
                  progress REAL NOT NULL DEFAULT 0,
                  current_stage TEXT NOT NULL DEFAULT 'queued',
                  result_json TEXT NOT NULL DEFAULT '{}',
                  error TEXT NOT NULL DEFAULT '',
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS dataset_uploads (
                  upload_id TEXT PRIMARY KEY,
                  tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
                  filename TEXT NOT NULL,
                  storage_path TEXT NOT NULL,
                  size_bytes INTEGER NOT NULL,
                  checksum_sha256 TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'uploaded',
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

    def store_enrolled_identity(self, identity_id: str, tenant_id: str, metadata_json: str = "{}") -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO enrolled_identities(identity_id, tenant_id, metadata_json)
                VALUES(?, ?, ?)
                """,
                (identity_id, tenant_id, metadata_json),
            )

    def store_template(
        self,
        template_id: str,
        identity_id: str,
        tenant_id: str,
        dimensions: int,
        encrypted_payload: str,
    ) -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO biometric_templates(
                  template_id, identity_id, tenant_id, dimensions, encrypted_payload
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (template_id, identity_id, tenant_id, dimensions, encrypted_payload),
            )


class AsyncDatabase:
    def __init__(self, database_url: str, pool_min: int = 5, pool_max: int = 20) -> None:
        self.database_url = database_url
        self.pool_min = pool_min
        self.pool_max = pool_max
        self._engine = None
        self._sessionmaker = None

    @property
    def available(self) -> bool:
        return self._engine is not None and self._sessionmaker is not None

    async def connect(self) -> None:
        try:
            from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        except Exception as exc:
            raise RuntimeError("SQLAlchemy asyncio support is required for Postgres production mode.") from exc
        self._engine = create_async_engine(
            self.database_url,
            pool_size=self.pool_min,
            max_overflow=max(0, self.pool_max - self.pool_min),
            pool_pre_ping=True,
        )
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)

    async def disconnect(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
        self._engine = None
        self._sessionmaker = None

    async def healthcheck(self) -> bool:
        if self._engine is None:
            await self.connect()
        assert self._engine is not None
        from sqlalchemy import text

        async with self._engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True

    def session(self):
        if self._sessionmaker is None:
            raise RuntimeError("AsyncDatabase.connect() must be called before opening sessions.")
        return self._sessionmaker()


def _url_from_path(path: str | Path) -> str:
    text = str(path)
    if "://" in text:
        return text
    return f"sqlite:///{text}"


def _backend_from_url(database_url: str) -> str:
    scheme = urlparse(database_url).scheme
    if scheme.startswith("sqlite"):
        return "sqlite"
    if scheme.startswith("postgresql"):
        return "postgres"
    return scheme


def _sqlite_path(database_url: str) -> Path:
    parsed = urlparse(database_url)
    if parsed.scheme == "sqlite" and parsed.path:
        if parsed.path.startswith("/") and ":" in parsed.path[:4]:
            raw_path = parsed.path[1:]
        elif parsed.path.startswith("/") and not parsed.netloc:
            raw_path = parsed.path[1:]
        else:
            raw_path = parsed.path
        return Path(raw_path)
    return Path(database_url.replace("sqlite:///", "", 1))
