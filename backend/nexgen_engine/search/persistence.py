"""
Durable storage for biometric templates and the audit trail.

DESIGN: SQLITE IS THE SOURCE OF TRUTH, THE VECTOR INDEX IS A DERIVED CACHE
--------------------------------------------------------------------------
The obvious design -- a FAISS index file on disk plus a separate metadata
table -- has a failure mode that matters for forensic use: the two artifacts
are written separately, so a crash between the two writes leaves an index row
with no metadata (or worse, metadata pointing at a vector slot that shifted).
Recovering from that requires knowing which of the two is stale, and nothing
records that.

Here the embedding is stored *in the same transactional row* as its metadata.
One fsync, one commit, no desync possible. At startup every row is streamed
back into the in-memory GalleryIndex, which keeps its existing optional FAISS
acceleration for search -- that index is a pure cache, and it can always be
rebuilt from SQLite. Nothing is lost if it is deleted.

This is fast enough well past the scale this system targets: rebuilding a
100k-template index is a single indexed SELECT plus one numpy reshape.

ENCRYPTION AT REST
------------------
A face embedding is biometric data: it is derived from a person's body, it is
not revocable, and published inversion attacks can reconstruct a recognizable
face image from a 512-d ArcFace vector. Storing it as plaintext float32 is
therefore not acceptable for a system claiming forensic use.

Templates are encrypted per-row with AES-256-GCM (authenticated, so tampering
is detected rather than silently decrypted into garbage). The key comes from
NEXGEN_TEMPLATE_KEY. When no key is configured the store runs in plaintext
mode and says so loudly -- it does not silently pretend to be encrypted.

SCHEMA VERSIONING
-----------------
`schema_version` is stamped on every row and `_meta` records the version the
file was created with. Readers refuse to open a file written by a newer schema
than they understand rather than silently misreading columns.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_ENV_KEY = "NEXGEN_TEMPLATE_KEY"


class SchemaVersionError(RuntimeError):
    """Raised when the on-disk schema is newer than this code understands."""


class TemplateDecryptionError(RuntimeError):
    """Raised when a stored template fails authenticated decryption."""


# ---------------------------------------------------------------- crypto ---


class TemplateCipher:
    """AES-256-GCM field-level encryption for embedding blobs.

    A fresh 96-bit nonce is generated per write and stored alongside the
    ciphertext. GCM authenticates, so a corrupted or tampered row raises
    instead of decrypting to plausible-looking noise -- which for a biometric
    template would mean silently matching the wrong person.
    """

    def __init__(self, key: bytes | None) -> None:
        self._aes = None
        if key is None:
            return
        if len(key) != 32:
            raise ValueError(
                f"{_ENV_KEY} must decode to exactly 32 bytes (AES-256); got {len(key)}"
            )
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        self._aes = AESGCM(key)

    @classmethod
    def from_env(cls) -> TemplateCipher:
        import base64

        raw = os.environ.get(_ENV_KEY, "").strip()
        if not raw:
            logger.warning(
                "%s is not set - biometric templates will be stored UNENCRYPTED. "
                "Generate one with: python -c \"import os,base64; "
                "print(base64.b64encode(os.urandom(32)).decode())\"",
                _ENV_KEY,
            )
            return cls(None)
        try:
            key = base64.b64decode(raw, validate=True)
        except Exception as exc:
            raise ValueError(f"{_ENV_KEY} must be base64-encoded: {exc}") from exc
        return cls(key)

    @property
    def enabled(self) -> bool:
        return self._aes is not None

    def encrypt(self, plaintext: bytes) -> bytes:
        if self._aes is None:
            return plaintext
        nonce = os.urandom(12)
        return nonce + self._aes.encrypt(nonce, plaintext, None)

    def decrypt(self, blob: bytes) -> bytes:
        if self._aes is None:
            return blob
        if len(blob) < 13:
            raise TemplateDecryptionError("ciphertext too short to contain a nonce")
        try:
            return self._aes.decrypt(blob[:12], blob[12:], None)
        except Exception as exc:
            raise TemplateDecryptionError(
                "template failed authenticated decryption - the row is corrupt, "
                f"or {_ENV_KEY} does not match the key it was written with"
            ) from exc


# ----------------------------------------------------------------- rows ---


@dataclass(frozen=True)
class TemplateRow:
    tenant_id: str
    template_id: str
    subject_id: str
    embedding: np.ndarray
    metadata: dict[str, Any]
    source_sha256: str
    model_version: str
    created_utc: str


@dataclass(frozen=True)
class AuditRow:
    audit_hash: str
    operation: str
    operator_id: str
    tenant_id: str
    decision: str
    score: float | None
    model_version: str
    subject_id: str | None
    created_utc: str
    detail: dict[str, Any]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS _meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS biometric_template (
    tenant_id      TEXT NOT NULL,
    template_id    TEXT NOT NULL,
    subject_id     TEXT NOT NULL,
    embedding      BLOB NOT NULL,
    dimensions     INTEGER NOT NULL,
    encrypted      INTEGER NOT NULL,
    metadata       TEXT NOT NULL,
    source_sha256  TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    created_utc    TEXT NOT NULL,
    PRIMARY KEY (tenant_id, template_id)
);

CREATE INDEX IF NOT EXISTS idx_template_subject
    ON biometric_template (tenant_id, subject_id);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_hash     TEXT NOT NULL,
    operation      TEXT NOT NULL,
    operator_id    TEXT NOT NULL,
    tenant_id      TEXT NOT NULL,
    decision       TEXT NOT NULL,
    score          REAL,
    model_version  TEXT NOT NULL,
    subject_id     TEXT,
    schema_version INTEGER NOT NULL,
    created_utc    TEXT NOT NULL,
    detail         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_hash    ON audit_log (audit_hash);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log (created_utc);
CREATE INDEX IF NOT EXISTS idx_audit_op      ON audit_log (tenant_id, operator_id);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class BiometricStore:
    """Durable template + audit storage backed by SQLite."""

    def __init__(self, db_path: str | Path, cipher: TemplateCipher | None = None) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cipher = cipher if cipher is not None else TemplateCipher.from_env()
        self._lock = threading.RLock()
        # check_same_thread=False: FastAPI serves requests on a thread pool.
        # Every access is serialized by self._lock, so this is safe.
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        # WAL: readers never block the writer. durability stays at the default
        # FULL synchronous level -- an audit log that can lose its last rows on
        # power loss is not an audit log.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(_SCHEMA)
        self._check_schema_version()

    def _check_schema_version(self) -> None:
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM _meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                self._db.execute(
                    "INSERT INTO _meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                self._db.commit()
                return
            found = int(row["value"])
            if found > SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{self.path} was written with schema v{found}, but this build "
                    f"understands only v{SCHEMA_VERSION}. Upgrade the application "
                    "rather than letting it misread the columns."
                )
            if found < SCHEMA_VERSION:
                # Migration hook. v1 is the initial schema, so there is nothing
                # to migrate yet; future versions add their ALTER TABLE steps
                # here and then bump the stored value.
                logger.info("migrating template store v%d -> v%d", found, SCHEMA_VERSION)
                self._db.execute(
                    "UPDATE _meta SET value=? WHERE key='schema_version'",
                    (str(SCHEMA_VERSION),),
                )
                self._db.commit()

    # ------------------------------------------------------- templates ---

    def put_template(
        self,
        tenant_id: str,
        template_id: str,
        subject_id: str,
        embedding: np.ndarray,
        metadata: dict[str, Any],
        source_sha256: str,
        model_version: str,
    ) -> None:
        vec = np.ascontiguousarray(embedding, dtype=np.float32)
        blob = self.cipher.encrypt(vec.tobytes())
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO biometric_template "
                "(tenant_id, template_id, subject_id, embedding, dimensions, "
                " encrypted, metadata, source_sha256, model_version, "
                " schema_version, created_utc) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    tenant_id,
                    template_id,
                    subject_id,
                    blob,
                    int(vec.size),
                    int(self.cipher.enabled),
                    json.dumps(metadata, default=str),
                    source_sha256,
                    model_version,
                    SCHEMA_VERSION,
                    _utcnow(),
                ),
            )
            self._db.commit()

    def iter_templates(self, tenant_id: str | None = None) -> Iterator[TemplateRow]:
        sql = (
            "SELECT * FROM biometric_template"
            + (" WHERE tenant_id=?" if tenant_id else "")
            + " ORDER BY created_utc"
        )
        args = (tenant_id,) if tenant_id else ()
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        for row in rows:
            raw = self.cipher.decrypt(bytes(row["embedding"])) if row["encrypted"] else bytes(row["embedding"])
            vec = np.frombuffer(raw, dtype=np.float32)
            if vec.size != row["dimensions"]:
                raise TemplateDecryptionError(
                    f"template {row['template_id']} decoded to {vec.size} floats "
                    f"but the row records {row['dimensions']}"
                )
            yield TemplateRow(
                tenant_id=row["tenant_id"],
                template_id=row["template_id"],
                subject_id=row["subject_id"],
                embedding=vec.copy(),
                metadata=json.loads(row["metadata"]),
                source_sha256=row["source_sha256"],
                model_version=row["model_version"],
                created_utc=row["created_utc"],
            )

    def delete_template(self, tenant_id: str, template_id: str) -> bool:
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM biometric_template WHERE tenant_id=? AND template_id=?",
                (tenant_id, template_id),
            )
            self._db.commit()
            return cur.rowcount > 0

    def count_templates(self, tenant_id: str | None = None) -> int:
        sql = "SELECT COUNT(*) c FROM biometric_template" + (
            " WHERE tenant_id=?" if tenant_id else ""
        )
        args = (tenant_id,) if tenant_id else ()
        with self._lock:
            return int(self._db.execute(sql, args).fetchone()["c"])

    # ----------------------------------------------------------- audit ---

    def write_audit(
        self,
        audit_hash: str,
        operation: str,
        operator_id: str,
        tenant_id: str,
        decision: str,
        model_version: str,
        score: float | None = None,
        subject_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Append one row to the audit trail. Never updates or deletes."""
        with self._lock:
            self._db.execute(
                "INSERT INTO audit_log "
                "(audit_hash, operation, operator_id, tenant_id, decision, score, "
                " model_version, subject_id, schema_version, created_utc, detail) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    audit_hash,
                    operation,
                    operator_id,
                    tenant_id,
                    decision,
                    float(score) if score is not None else None,
                    model_version,
                    subject_id,
                    SCHEMA_VERSION,
                    _utcnow(),
                    json.dumps(detail or {}, default=str),
                ),
            )
            self._db.commit()

    def get_audit(self, audit_hash: str) -> list[AuditRow]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM audit_log WHERE audit_hash=? ORDER BY created_utc",
                (audit_hash,),
            ).fetchall()
        return [
            AuditRow(
                audit_hash=r["audit_hash"],
                operation=r["operation"],
                operator_id=r["operator_id"],
                tenant_id=r["tenant_id"],
                decision=r["decision"],
                score=r["score"],
                model_version=r["model_version"],
                subject_id=r["subject_id"],
                created_utc=r["created_utc"],
                detail=json.loads(r["detail"]),
            )
            for r in rows
        ]

    def count_audit(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) c FROM audit_log").fetchone()["c"])

    def close(self) -> None:
        with self._lock:
            self._db.close()


def restore_into(store: BiometricStore, index: Any) -> int:
    """Rebuild an in-memory GalleryIndex from durable storage. Returns rows loaded.

    Called at startup. The index is a derived cache; this is the only thing
    that makes enrollments survive a restart.
    """
    count = 0
    for row in store.iter_templates():
        index.add(
            tenant_id=row.tenant_id,
            template_id=row.template_id,
            subject_id=row.subject_id,
            embedding=row.embedding,
            metadata=row.metadata,
        )
        count += 1
    logger.info("restored %d templates from %s", count, store.path)
    return count
