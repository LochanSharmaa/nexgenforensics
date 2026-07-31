"""End-to-end durability test against the real EngineService.

This is the test the brief asks for in Phase 4 step 4: enroll an identity,
restart the server, confirm /identify still finds it. It exercises the real
recognition pipeline on GPU, so it is slower than the unit tests in
test_persistence.py -- but it is the only test that proves the wiring, not just
the storage layer, actually persists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _face_bytes(seed: int) -> bytes:
    """A real face crop from AgeDB, so the pipeline's face check passes."""
    import io

    agedb = _BACKEND.parent / "src_extracted/AgeDB/AgeDB"
    files = sorted(agedb.glob("*.jpg"))
    if not files:
        pytest.skip("AgeDB images not available")
    img = Image.open(files[seed % len(files)]).convert("RGB").resize((256, 256))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.slow
def test_enrollment_survives_restart(tmp_path):
    from nexgen_engine.api.service import EngineService

    audit = tmp_path / "audit.jsonl"
    db = tmp_path / "templates.db"
    photo = _face_bytes(0)

    svc = EngineService(audit_path=audit, store_path=db)
    assert svc.restored_count == 0, "a fresh store must start empty"

    res = svc.enroll(photo, "subject-alpha", {"case": "TEST-1"})
    assert res.decision == "enrolled"
    assert svc.store.count_templates() == 1

    # identify against the live index
    before = svc.identify(photo, operator_id="op-1", top_k=5)
    assert before.matches, "probe should match the identity just enrolled"
    assert before.matches[0].identity_id == "subject-alpha"
    top_before = before.matches[0].confidence

    svc.store.close()
    del svc

    # ---- simulated server restart: brand new service, same paths ----
    svc2 = EngineService(audit_path=audit, store_path=db)
    assert svc2.restored_count == 1, "template was not restored from disk"

    after = svc2.identify(photo, operator_id="op-1", top_k=5)
    assert after.matches, "enrollment did NOT survive restart"
    assert after.matches[0].identity_id == "subject-alpha"
    # restored template must be bit-identical, so the score must match exactly
    assert after.matches[0].confidence == pytest.approx(top_before, abs=1e-6)

    svc2.store.close()


@pytest.mark.slow
def test_audit_hash_is_resolvable(tmp_path):
    """The hash handed back to a caller must resolve to a stored record."""
    from nexgen_engine.api.service import EngineService

    svc = EngineService(audit_path=tmp_path / "a.jsonl", store_path=tmp_path / "t.db")
    ref, probe = _face_bytes(1), _face_bytes(2)

    res = svc.verify(ref, probe, operator_id="op-9")
    rows = svc.store.get_audit(res.audit_hash)

    assert len(rows) == 1, "verify() did not write a durable audit row"
    assert rows[0].operation == "verify"
    assert rows[0].operator_id == "op-9"
    assert rows[0].score == pytest.approx(res.score, abs=1e-5)
    assert rows[0].model_version == EngineService.MODEL_VERSION
    assert "ref_sha256" in rows[0].detail
    svc.store.close()


@pytest.mark.slow
def test_identify_writes_audit(tmp_path):
    from nexgen_engine.api.service import EngineService

    svc = EngineService(audit_path=tmp_path / "a.jsonl", store_path=tmp_path / "t.db")
    svc.enroll(_face_bytes(3), "subject-beta", {})
    res = svc.identify(_face_bytes(3), operator_id="op-5")

    rows = svc.store.get_audit(res.audit_hash)
    assert len(rows) == 1
    assert rows[0].operation == "identify"
    assert rows[0].detail["gallery_size"] >= 1
    svc.store.close()
