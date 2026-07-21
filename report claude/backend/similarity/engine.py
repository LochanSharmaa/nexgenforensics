"""
similarity/engine.py

Two independent, clearly-separated similarity computations:

1. Primary: embedding-based similarity from the EXISTING recognition
   model's vectors (cosine, euclidean, angular). This is the number the
   report leads with.
2. Supplementary: soft-biometric feature-vector similarity (Euclidean /
   Mahalanobis / Hamming) computed purely from this system's own
   anthropometric measurements, per Tome et al. Always labeled as
   corroborating/supplementary — never merged into the primary score.
"""
from __future__ import annotations

import numpy as np


def embedding_similarity(vec_a: np.ndarray, vec_b: np.ndarray, threshold: float) -> dict:
    vec_a = vec_a / (np.linalg.norm(vec_a) or 1.0)
    vec_b = vec_b / (np.linalg.norm(vec_b) or 1.0)

    cosine = float(np.dot(vec_a, vec_b))
    euclidean = float(np.linalg.norm(vec_a - vec_b))
    cos_clamped = max(-1.0, min(1.0, cosine))
    angular = float(np.degrees(np.arccos(cos_clamped)))

    decision = "above_threshold" if cosine >= threshold else "below_threshold"

    return {
        "cosine_similarity": round(cosine, 4),
        "euclidean_distance": round(euclidean, 4),
        "angular_distance": round(angular, 2),
        "threshold_used": threshold,
        "decision_category": decision,
        "candidate_rank": None,
    }


COVARIANCE_DIAGONAL = {
    "face_width": 0.04,
    "face_height": 0.05,
    "interocular_distance": 0.005,
    "eye_width_r": 0.005,
    "eye_width_l": 0.005,
    "eyebrow_length_r": 0.01,
    "eyebrow_length_l": 0.01,
    "eyebrow_angle_r": 25.0,  # angle in degrees, higher variance
    "eyebrow_angle_l": 25.0,
    "nose_width": 0.02,
    "nose_height": 0.03,
    "mouth_width": 0.03,
    "mouth_height": 0.01,
    "mouth_angle": 15.0,      # angle in degrees
    "jaw_width": 0.04,
    "chin_height": 0.02,
    "forehead_height": 0.03,
    "forehead_width": 0.05,
    "ear_height_l": 0.01,
    "ear_height_r": 0.01,
}


def soft_feature_similarity(
    probe_continuous: list[dict],
    candidate_continuous: list[dict],
    probe_discrete: list[dict] | None = None,
    candidate_discrete: list[dict] | None = None
) -> dict:
    """Scale-normalized Euclidean, Mahalanobis, and Hamming distances over shared features.
    Supplementary only — see module docstring."""
    p_feat = {m["feature_id"]: m for m in probe_continuous}
    c_feat = {m["feature_id"]: m for m in candidate_continuous}
    shared = sorted(set(p_feat) & set(c_feat))
    if not shared:
        return {"euclidean": None, "mahalanobis": None, "hamming": None}

    # Extract interocular distance for scale normalization of distance features
    p_iod = p_feat.get("interocular_distance", {}).get("value", 1.0) or 1.0
    c_iod = c_feat.get("interocular_distance", {}).get("value", 1.0) or 1.0

    p_norm = {}
    c_norm = {}
    for k in shared:
        p_val = p_feat[k]["value"]
        c_val = c_feat[k]["value"]
        # Normalize only distance/pixel features, keep angles as-is
        is_angle = "angle" in k or p_feat[k].get("unit") == "deg"
        if is_angle:
            p_norm[k] = p_val
            c_norm[k] = c_val
        else:
            p_norm[k] = p_val / p_iod
            c_norm[k] = c_val / c_iod

    # 1. Euclidean Distance
    diffs = np.array([p_norm[k] - c_norm[k] for k in shared])
    euclidean = float(np.linalg.norm(diffs))

    # 2. Mahalanobis Distance (diagonal covariance matrix approach)
    mahalanobis_sum = 0.0
    for k in shared:
        diff = p_norm[k] - c_norm[k]
        is_angle = "angle" in k or p_feat[k].get("unit") == "deg"
        fallback_var = 10.0 if is_angle else 0.02
        variance = COVARIANCE_DIAGONAL.get(k, fallback_var)
        mahalanobis_sum += (diff ** 2) / variance
    mahalanobis = float(np.sqrt(mahalanobis_sum))

    # 3. Hamming Distance over categorical attributes
    hamming = None
    if probe_discrete and candidate_discrete:
        p_disc = {m["feature_id"].replace("_category", ""): m["category"] for m in probe_discrete}
        c_disc = {m["feature_id"].replace("_category", ""): m["category"] for m in candidate_discrete}
        shared_disc = sorted(set(p_disc) & set(c_disc))
        if shared_disc:
            mismatches = sum(1 for k in shared_disc if p_disc[k] != c_disc[k])
            hamming = float(mismatches) / len(shared_disc)

    return {
        "euclidean": round(euclidean, 3),
        "mahalanobis": round(mahalanobis, 3),
        "hamming": round(hamming, 3) if hamming is not None else None,
    }

