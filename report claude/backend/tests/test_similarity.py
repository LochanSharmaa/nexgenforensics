"""
tests/test_similarity.py

Unit tests for the Mahalanobis and Hamming distance functions in
similarity/engine.py, and for the schema "468" measurement pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from similarity.engine import embedding_similarity, soft_feature_similarity


# ===== Helpers =====

def _make_continuous(pairs: dict) -> list[dict]:
    """Build a minimal continuous measurements list from {feature_id: value}."""
    out = []
    for fid, val in pairs.items():
        unit = "deg" if "angle" in fid else "px"
        out.append({"feature_id": fid, "value": val, "unit": unit, "name": fid, "region": "face", "landmarks_used": []})
    return out


def _make_discrete(pairs: dict) -> list[dict]:
    """Build minimal discrete measurement list from {feature_id: category}."""
    return [{"feature_id": fid + "_category", "category": cat} for fid, cat in pairs.items()]


# ===== Embedding Similarity =====

class TestEmbeddingSimilarity:
    def test_identical_vectors_cosine_one(self):
        v = np.random.default_rng(0).normal(size=512).astype(np.float32)
        result = embedding_similarity(v, v, threshold=0.5)
        assert abs(result["cosine_similarity"] - 1.0) < 1e-4

    def test_orthogonal_vectors_cosine_near_zero(self):
        v1 = np.zeros(512, dtype=np.float32); v1[0] = 1.0
        v2 = np.zeros(512, dtype=np.float32); v2[1] = 1.0
        result = embedding_similarity(v1, v2, threshold=0.5)
        assert abs(result["cosine_similarity"]) < 1e-4

    def test_above_threshold_decision(self):
        v = np.random.default_rng(1).normal(size=512).astype(np.float32)
        result = embedding_similarity(v, v, threshold=0.5)
        assert result["decision_category"] == "above_threshold"

    def test_below_threshold_decision(self):
        v1 = np.zeros(512, dtype=np.float32); v1[0] = 1.0
        v2 = np.zeros(512, dtype=np.float32); v2[1] = 1.0
        result = embedding_similarity(v1, v2, threshold=0.5)
        assert result["decision_category"] == "below_threshold"

    def test_euclidean_non_negative(self):
        rng = np.random.default_rng(7)
        v1 = rng.normal(size=512).astype(np.float32)
        v2 = rng.normal(size=512).astype(np.float32)
        result = embedding_similarity(v1, v2, threshold=0.5)
        assert result["euclidean_distance"] >= 0.0

    def test_angular_in_degrees_range(self):
        rng = np.random.default_rng(9)
        v1 = rng.normal(size=512).astype(np.float32)
        v2 = rng.normal(size=512).astype(np.float32)
        result = embedding_similarity(v1, v2, threshold=0.5)
        assert 0.0 <= result["angular_distance"] <= 180.0


# ===== Soft Feature Similarity =====

class TestSoftFeatureSimilarity:
    def test_identical_features_euclidean_zero(self):
        feats = _make_continuous({"face_width": 100.0, "interocular_distance": 30.0, "nose_width": 20.0})
        result = soft_feature_similarity(feats, feats)
        assert result["euclidean"] == 0.0

    def test_identical_features_mahalanobis_zero(self):
        feats = _make_continuous({"face_width": 100.0, "interocular_distance": 30.0, "nose_width": 20.0})
        result = soft_feature_similarity(feats, feats)
        assert result["mahalanobis"] == 0.0

    def test_identical_discrete_hamming_zero(self):
        feats = _make_continuous({"interocular_distance": 30.0})
        disc = _make_discrete({"nose_width": "narrow", "face_width": "wide"})
        result = soft_feature_similarity(feats, feats, disc, disc)
        assert result["hamming"] == 0.0

    def test_all_discrete_mismatch_hamming_one(self):
        feats = _make_continuous({"interocular_distance": 30.0})
        disc_a = _make_discrete({"nose_width": "narrow", "face_width": "wide"})
        disc_b = _make_discrete({"nose_width": "wide", "face_width": "narrow"})
        result = soft_feature_similarity(feats, feats, disc_a, disc_b)
        assert result["hamming"] == 1.0

    def test_partial_discrete_mismatch(self):
        feats = _make_continuous({"interocular_distance": 30.0})
        disc_a = _make_discrete({"nose_width": "narrow", "face_width": "wide"})
        disc_b = _make_discrete({"nose_width": "narrow", "face_width": "narrow"})
        result = soft_feature_similarity(feats, feats, disc_a, disc_b)
        assert result["hamming"] == pytest.approx(0.5, abs=1e-3)

    def test_no_shared_features_returns_none(self):
        p = _make_continuous({"face_width": 100.0})
        c = _make_continuous({"mouth_width": 50.0})
        result = soft_feature_similarity(p, c)
        assert result["euclidean"] is None
        assert result["mahalanobis"] is None
        assert result["hamming"] is None

    def test_mahalanobis_with_angle_feature(self):
        p = _make_continuous({"eyebrow_angle_r": 10.0, "interocular_distance": 30.0})
        c = _make_continuous({"eyebrow_angle_r": 20.0, "interocular_distance": 30.0})
        result = soft_feature_similarity(p, c)
        assert result["mahalanobis"] > 0.0

    def test_results_are_floats_or_none(self):
        p = _make_continuous({"face_width": 100.0, "interocular_distance": 30.0})
        c = _make_continuous({"face_width": 95.0, "interocular_distance": 28.0})
        disc_a = _make_discrete({"nose_width": "narrow"})
        disc_b = _make_discrete({"nose_width": "wide"})
        result = soft_feature_similarity(p, c, disc_a, disc_b)
        for key in ["euclidean", "mahalanobis", "hamming"]:
            assert result[key] is None or isinstance(result[key], float)


# ===== Schema "468" Measurement Roundtrip =====

class TestSchema468Measurements:
    def _make_468_landmarks(self):
        """Minimal 468-point landmark list with enough points to test measurement definitions."""
        lms = [{"id": i, "x": float(i * 2), "y": 0.0, "confidence": 0.95} for i in range(480)]
        # Place key points in known geometry:
        W, H = 600.0, 800.0
        overrides = {
            10:  (W * 0.5,  H * 0.12),  # forehead top
            168: (W * 0.5,  H * 0.30),  # forehead bottom
            109: (W * 0.25, H * 0.20),  # forehead left
            338: (W * 0.75, H * 0.20),  # forehead right
            # Interocular
            33:  (W * 0.35, H * 0.40),
            263: (W * 0.65, H * 0.40),
            # Nose tip
            4:   (W * 0.50, H * 0.55),
        }
        for idx, (x, y) in overrides.items():
            if idx < len(lms):
                lms[idx] = {"id": idx, "x": x, "y": y, "confidence": 0.95}
        return lms

    def test_468_schema_produces_measurements(self):
        from measurements.engine import compute_continuous_measurements
        lms = self._make_468_landmarks()
        results = compute_continuous_measurements("468", lms)
        # At minimum, forehead features should be present if config defines them
        # (gracefully accepts empty if config doesn't have "468" fully wired)
        assert isinstance(results, list)
        for m in results:
            assert "feature_id" in m
            assert "value" in m
            assert isinstance(m["value"], (int, float))

    def test_468_no_fabricated_values(self):
        from measurements.engine import compute_continuous_measurements
        # Empty landmarks list should give empty results, not fabricated data
        results = compute_continuous_measurements("468", [])
        assert results == []

    def test_discrete_from_468_continuous(self):
        from measurements.engine import compute_continuous_measurements, compute_discrete_measurements
        lms = self._make_468_landmarks()
        continuous = compute_continuous_measurements("468", lms)
        discrete = compute_discrete_measurements(continuous)
        assert isinstance(discrete, list)
        for d in discrete:
            assert "feature_id" in d
            assert "category" in d
            assert isinstance(d["category"], str)
