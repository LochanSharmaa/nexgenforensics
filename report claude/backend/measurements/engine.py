"""
measurements/engine.py

Pure-function anthropometric measurement engine. Given a landmark schema and
list of {id, x, y, confidence} points, computes:

  - continuous measurements (pixel distances / angles), config-driven
  - discrete (forensic categorical) buckets derived from the continuous
    values, config-driven, never hardcoded

No I/O, no model calls — fully unit-testable with synthetic landmark
fixtures. Based on the feature methodology in
Tome et al., "Facial Soft Biometric Features for Forensic Face Recognition".
"""
from __future__ import annotations

import math
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_yaml(name: str) -> dict:
    with open(_CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def _points_by_id(landmarks: list[dict]) -> dict[int, tuple[float, float]]:
    return {p["id"]: (p["x"], p["y"]) for p in landmarks}


def _distance(p1, p2) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _angle_deg(p1, p2) -> float:
    """Angle of the segment p1->p2 relative to the horizontal, in degrees."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def compute_continuous_measurements(landmark_schema: str, landmarks: list[dict]) -> list[dict]:
    defs = _load_yaml("measurement_definitions.yaml")
    schema_defs = defs.get(landmark_schema)
    if not schema_defs:
        return []

    pts = _points_by_id(landmarks)
    results = []
    for d in schema_defs:
        ids = d["points"]
        if any(i not in pts for i in ids):
            continue  # missing landmark -> omit, never fabricate
        p1, p2 = pts[ids[0]], pts[ids[1]]
        if d["type"] == "distance":
            value = _distance(p1, p2)
        elif d["type"] == "angle":
            value = _angle_deg(p1, p2)
        else:
            continue
        results.append({
            "feature_id": d["feature_id"],
            "name": d["name"],
            "region": d["region"],
            "value": round(value, 2),
            "unit": d["unit"],
            "landmarks_used": ids,
            "formula_ref": d["formula_ref"],
        })
    return results


def compute_discrete_measurements(continuous_measurements: list[dict]) -> list[dict]:
    thresholds = _load_yaml("discrete_thresholds.yaml")
    normalize_by = thresholds["normalize_by"]
    by_id = {m["feature_id"]: m for m in continuous_measurements}

    norm_measure = by_id.get(normalize_by)
    if norm_measure is None or norm_measure["value"] == 0:
        return []  # can't normalize -> omit discrete features, don't guess
    norm_value = norm_measure["value"]

    results = []
    for feature_id, spec in thresholds["features"].items():
        m = by_id.get(feature_id)
        if m is None:
            continue
        ratio = m["value"] / norm_value
        bounds = spec["ratio_bounds"]
        categories = spec["categories"]

        idx = 0
        for b in bounds:
            if ratio >= b:
                idx += 1
        idx = min(idx, len(categories) - 1)
        category = categories[idx]

        results.append({
            "feature_id": f"{feature_id}_category",
            "name": f"{m['name']} (category)",
            "region": m["region"],
            "category": category,
            "threshold_basis": (
                f"ratio to {normalize_by} = {ratio:.3f}, "
                f"bounds={bounds}, normalize_by={normalize_by}"
            ),
            "landmarks_used": m["landmarks_used"],
        })
    return results
