import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from measurements.engine import compute_continuous_measurements, compute_discrete_measurements


def _fixture_68():
    """Simple synthetic 68-landmark set with known distances for asserting values."""
    pts = {i: (float(i), 0.0) for i in range(68)}  # all on a line, spaced 1px apart
    # override the ones our measurement defs actually use, with clean values
    overrides = {
        0: (0, 0), 16: (100, 0),      # face_width -> 100
        27: (50, 0), 8: (50, 120),    # face_height -> 120
        39: (45, 50), 42: (55, 50),   # interocular -> 10
        36: (30, 50), 39: (45, 50),   # eye_width_r -> 15
        42: (55, 50), 45: (70, 50),   # eye_width_l -> 15
        17: (10, 30), 21: (30, 30),   # eyebrow_r length -> 20
        22: (70, 30), 26: (90, 30),   # eyebrow_l length -> 20
        31: (40, 65), 35: (60, 65),   # nose_width -> 20
        33: (50, 80),                  # nose_height p2 (27 already set) -> 80
        48: (35, 100), 54: (65, 100), # mouth_width -> 30
        51: (50, 95), 57: (50, 110),  # mouth_height -> 15
        4: (15, 90), 12: (85, 90),    # jaw_width -> 70
    }
    pts.update(overrides)
    return [{"id": i, "x": x, "y": y, "confidence": 0.95} for i, (x, y) in pts.items()]


def test_face_width_distance():
    landmarks = _fixture_68()
    results = compute_continuous_measurements("68", landmarks)
    by_id = {m["feature_id"]: m for m in results}
    assert by_id["face_width"]["value"] == 100.0


def test_interocular_distance():
    landmarks = _fixture_68()
    results = compute_continuous_measurements("68", landmarks)
    by_id = {m["feature_id"]: m for m in results}
    assert by_id["interocular_distance"]["value"] == 10.0


def test_missing_landmark_is_omitted_not_fabricated():
    landmarks = [p for p in _fixture_68() if p["id"] != 16]  # drop a point face_width needs
    results = compute_continuous_measurements("68", landmarks)
    ids = {m["feature_id"] for m in results}
    assert "face_width" not in ids


def test_unknown_schema_returns_empty():
    assert compute_continuous_measurements("999", _fixture_68()) == []


def test_discrete_bucketing_is_deterministic():
    landmarks = _fixture_68()
    continuous = compute_continuous_measurements("68", landmarks)
    discrete_a = compute_discrete_measurements(continuous)
    discrete_b = compute_discrete_measurements(continuous)
    assert discrete_a == discrete_b
    assert all("category" in d for d in discrete_a)
