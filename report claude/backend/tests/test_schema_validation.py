import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.validator import validate_report

MINIMAL_VALID = {
    "administrative": {
        "report_id": "r1", "schema_version": "1.0.0",
        "created_at": "2026-01-01T00:00:00+00:00", "software_version": "0.0.1",
    },
    "evidence": {"probe_image_id": "p.png", "candidate_image_id": "c.png"},
    "image_metadata": {"filename": "p.png", "width": 10, "height": 10, "format": "png", "capture_date": None},
    "quality_metrics": {"blur_score": 100.0, "detection_confidence": 0.9,
                         "brightness": 100.0, "contrast": 30.0,
                         "face_visibility_pct": 90.0},
    "face_detection": {"bounding_box": {"x": 0, "y": 0, "w": 10, "h": 10}, "detection_confidence": 0.9},
    "landmarks": {"schema": "68", "points": []},
    "measurements_continuous": [],
    "measurements_discrete": [],
    "morphological_features": [],
    "embedding_metrics": {},
    "similarity_metrics": {"cosine_similarity": 0.5, "decision_category": "below_threshold"},
    "limitations": [],
    "technical_info": {},
    "audit": {"stages": []},
}


def test_minimal_valid_report_passes():
    errors = validate_report(MINIMAL_VALID)
    assert errors == []


def test_missing_required_field_fails():
    bad = {k: v for k, v in MINIMAL_VALID.items() if k != "similarity_metrics"}
    errors = validate_report(bad)
    assert any("similarity_metrics" in e for e in errors)


def test_bad_landmark_schema_enum_fails():
    bad = dict(MINIMAL_VALID)
    bad["landmarks"] = {"schema": "not-a-valid-schema", "points": []}
    errors = validate_report(bad)
    assert len(errors) > 0
