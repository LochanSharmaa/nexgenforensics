"""
measurements/morphology.py

Compares probe vs. candidate discrete measurement categories, region by
region, and produces the morphological_features section of the report JSON.
Comparison labels are derived purely from category equality/adjacency —
never from free-text generation.
"""
from __future__ import annotations


def _category_distance(categories: list[str], cat_a: str, cat_b: str) -> int:
    try:
        return abs(categories.index(cat_a) - categories.index(cat_b))
    except ValueError:
        return 99


def compare_morphology(probe_discrete: list[dict], candidate_discrete: list[dict],
                        discrete_thresholds: dict) -> list[dict]:
    probe_by_id = {m["feature_id"]: m for m in probe_discrete}
    cand_by_id = {m["feature_id"]: m for m in candidate_discrete}
    features_cfg = discrete_thresholds["features"]

    results = []
    for base_feature_id, spec in features_cfg.items():
        fid = f"{base_feature_id}_category"
        p = probe_by_id.get(fid)
        c = cand_by_id.get(fid)
        if p is None or c is None:
            results.append({
                "region": spec.get("region", base_feature_id),
                "probe_observation": p["category"] if p else "not_assessable",
                "candidate_observation": c["category"] if c else "not_assessable",
                "comparison_label": "not_assessable",
                "notes": "One or both images lacked the landmarks needed for this feature.",
            })
            continue

        dist = _category_distance(spec["categories"], p["category"], c["category"])
        if dist == 0:
            label = "similar"
        elif dist == 1:
            label = "minor_variation"
        else:
            label = "observed_variation"

        results.append({
            "region": p["region"],
            "probe_observation": p["category"],
            "candidate_observation": c["category"],
            "comparison_label": label,
            "notes": f"Category distance = {dist} on a {len(spec['categories'])}-level scale.",
        })
    return results
