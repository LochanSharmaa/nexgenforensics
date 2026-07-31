"""Durability tests for the biometric template store and audit log.

The central claim under test is the one the product page makes: an enrolled
identity survives a process restart, and an audit hash handed back to a user
can actually be looked up later.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.search.gallery_index import GalleryIndex  # noqa: E402
from nexgen_engine.search.persistence import (  # noqa: E402
    BiometricStore,
    TemplateCipher,
    TemplateDecryptionError,
    restore_into,
)


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _key() -> bytes:
    return base64.b64encode(os.urandom(32))


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "templates.db"


def test_enroll_survives_restart(db_path: Path):
    """Enroll, drop the process state, reopen -> the template is still there."""
    cipher = TemplateCipher(base64.b64decode(_key()))
    emb = _vec(1)

    store = BiometricStore(db_path, cipher=cipher)
    store.put_template(
        tenant_id="t1",
        template_id="tpl-1",
        subject_id="alice",
        embedding=emb,
        metadata={"source": "upload"},
        source_sha256="a" * 64,
        model_version="ensemble_v1",
    )
    store.close()

    # --- simulated restart: brand new objects, nothing carried over ---
    store2 = BiometricStore(db_path, cipher=cipher)
    index = GalleryIndex(dimensions=512)
    assert index.size("t1") == 0, "fresh index must start empty"

    loaded = restore_into(store2, index)
    assert loaded == 1
    assert index.size("t1") == 1

    rows = list(store2.iter_templates())
    assert rows[0].subject_id == "alice"
    # round-trip must be bit-exact; a lossy template silently degrades matching
    np.testing.assert_array_equal(rows[0].embedding, emb)
    store2.close()


def test_templates_are_encrypted_on_disk(db_path: Path):
    """The raw float32 bytes must not be readable in the DB file."""
    raw_key = base64.b64decode(_key())
    emb = _vec(2)

    store = BiometricStore(db_path, cipher=TemplateCipher(raw_key))
    store.put_template("t1", "tpl-1", "bob", emb, {}, "b" * 64, "m")
    store.close()

    blob = db_path.read_bytes()
    assert emb.tobytes() not in blob, "plaintext embedding found in the database file"


def test_wrong_key_is_detected_not_silently_wrong(db_path: Path):
    """A mismatched key must raise, never decrypt into a plausible vector."""
    store = BiometricStore(db_path, cipher=TemplateCipher(base64.b64decode(_key())))
    store.put_template("t1", "tpl-1", "carol", _vec(3), {}, "c" * 64, "m")
    store.close()

    other = BiometricStore(db_path, cipher=TemplateCipher(base64.b64decode(_key())))
    with pytest.raises(TemplateDecryptionError):
        list(other.iter_templates())
    other.close()


def test_tenant_isolation_on_restore(db_path: Path):
    """Restored templates land in their own tenant shard, not a shared one."""
    store = BiometricStore(db_path, cipher=TemplateCipher(None))
    store.put_template("tenant-a", "tpl-a", "alice", _vec(4), {}, "d" * 64, "m")
    store.put_template("tenant-b", "tpl-b", "bob", _vec(5), {}, "e" * 64, "m")
    store.close()

    store2 = BiometricStore(db_path, cipher=TemplateCipher(None))
    index = GalleryIndex(dimensions=512)
    restore_into(store2, index)

    assert index.size("tenant-a") == 1
    assert index.size("tenant-b") == 1
    assert index.size("tenant-c") == 0
    store2.close()


def test_audit_row_is_queryable_by_hash(db_path: Path):
    """An audit hash returned to a caller must resolve to a real record."""
    store = BiometricStore(db_path, cipher=TemplateCipher(None))
    store.write_audit(
        audit_hash="deadbeef",
        operation="verify",
        operator_id="op-7",
        tenant_id="t1",
        decision="match",
        model_version="ensemble_v1",
        score=0.84,
        detail={"threshold": 0.28},
    )
    store.close()

    store2 = BiometricStore(db_path, cipher=TemplateCipher(None))
    rows = store2.get_audit("deadbeef")
    assert len(rows) == 1
    assert rows[0].operation == "verify"
    assert rows[0].operator_id == "op-7"
    assert rows[0].decision == "match"
    assert rows[0].score == pytest.approx(0.84)
    assert rows[0].detail["threshold"] == 0.28
    assert store2.get_audit("nonexistent") == []
    store2.close()


def test_audit_survives_restart_and_appends(db_path: Path):
    store = BiometricStore(db_path, cipher=TemplateCipher(None))
    for i in range(3):
        store.write_audit(f"h{i}", "identify", "op", "t1", "no_match", "m", score=0.1 * i)
    store.close()

    store2 = BiometricStore(db_path, cipher=TemplateCipher(None))
    assert store2.count_audit() == 3
    store2.write_audit("h3", "identify", "op", "t1", "match", "m", score=0.9)
    assert store2.count_audit() == 4
    store2.close()


def test_rejects_newer_schema(db_path: Path):
    from nexgen_engine.search.persistence import SchemaVersionError

    store = BiometricStore(db_path, cipher=TemplateCipher(None))
    store._db.execute("UPDATE _meta SET value='999' WHERE key='schema_version'")
    store._db.commit()
    store.close()

    with pytest.raises(SchemaVersionError):
        BiometricStore(db_path, cipher=TemplateCipher(None))
