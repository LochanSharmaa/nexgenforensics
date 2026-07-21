"""
tests/test_pose_quality.py

Unit tests for the solvePnP pose estimator and quality metric functions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import pytest

from quality.assessment import (
    _estimate_pose,
    _crude_pose_fallback,
    _estimate_noise,
    _estimate_compression,
    _estimate_lighting_uniformity,
    assess_quality,
)


def _make_lm(idx, x, y, conf=0.95):
    return {"id": idx, "x": float(x), "y": float(y), "confidence": conf}


def _frontal_468_landmarks(w=600, h=800):
    """Approximate frontal-pose 468-schema landmarks for testing solvePnP."""
    # key indices used by _estimate_pose for schema "468":
    # left_eye=33, right_eye=263, nose=4, mouth_l=61, mouth_r=291, chin=152
    lms = [_make_lm(i, w / 2, h / 2) for i in range(468)]  # filler defaults
    lms[33]  = _make_lm(33,  w * 0.35, h * 0.40)   # left eye outer
    lms[263] = _make_lm(263, w * 0.65, h * 0.40)   # right eye outer
    lms[4]   = _make_lm(4,   w * 0.50, h * 0.55)   # nose tip
    lms[61]  = _make_lm(61,  w * 0.42, h * 0.70)   # mouth left
    lms[291] = _make_lm(291, w * 0.58, h * 0.70)   # mouth right
    lms[152] = _make_lm(152, w * 0.50, h * 0.85)   # chin
    return lms


class TestEstimatePose:
    def test_frontal_yaw_near_zero(self):
        """Symmetric frontal face should yield yaw close to 0."""
        lms = _frontal_468_landmarks()
        result = _estimate_pose(lms, w=600, h=800)
        assert abs(result["yaw"]) < 30.0, f"Expected near-zero yaw, got {result['yaw']}"

    def test_empty_landmarks_returns_zeros(self):
        result = _estimate_pose([], w=600, h=800)
        assert result == {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    def test_output_within_bounds(self):
        lms = _frontal_468_landmarks()
        result = _estimate_pose(lms, w=600, h=800)
        assert -90.0 <= result["yaw"]   <= 90.0
        assert -90.0 <= result["pitch"] <= 90.0
        assert -90.0 <= result["roll"]  <= 90.0

    def test_fallback_on_missing_key_points(self):
        """When key landmarks are absent, falls back to crude proxy — must not raise."""
        lms = [_make_lm(i, 300, 400) for i in range(10)]  # only 10 pts, missing 33/263 etc.
        result = _estimate_pose(lms, w=600, h=800)
        assert isinstance(result["yaw"], float)

    def test_68_schema_mapping(self):
        """68-schema uses indices 36/45/30/48/54/8."""
        lms = [_make_lm(i, 300, 400) for i in range(68)]
        lms[36] = _make_lm(36, 200, 300)
        lms[45] = _make_lm(45, 400, 300)
        lms[30] = _make_lm(30, 300, 360)
        lms[48] = _make_lm(48, 260, 460)
        lms[54] = _make_lm(54, 340, 460)
        lms[8]  = _make_lm(8,  300, 540)
        result = _estimate_pose(lms, w=600, h=800)
        assert -90.0 <= result["yaw"] <= 90.0


class TestNoiseEstimator:
    def test_clean_image_low_noise(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        assert _estimate_noise(gray) == 0.0

    def test_noisy_image_higher_noise(self):
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 255, (100, 100), dtype=np.uint8)
        score = _estimate_noise(gray)
        assert score > 5.0, f"Noisy image should have noise > 5, got {score}"

    def test_returns_float(self):
        gray = np.ones((50, 50), dtype=np.uint8) * 128
        assert isinstance(_estimate_noise(gray), float)


class TestCompressionEstimator:
    def test_uniform_image_near_zero(self):
        gray = np.ones((64, 64), dtype=np.uint8) * 128
        score = _estimate_compression(gray)
        assert 0.0 <= score <= 0.1, f"Uniform image should have low blockiness, got {score}"

    def test_score_in_range(self):
        rng = np.random.default_rng(7)
        gray = rng.integers(0, 255, (64, 64), dtype=np.uint8)
        score = _estimate_compression(gray)
        assert 0.0 <= score <= 1.0

    def test_small_image_returns_zero(self):
        gray = np.ones((8, 8), dtype=np.uint8) * 100
        assert _estimate_compression(gray) == 0.0


class TestLightingUniformity:
    def test_uniform_image_is_one(self):
        gray = np.ones((100, 100), dtype=np.uint8) * 150
        assert _estimate_lighting_uniformity(gray) == 1.0

    def test_extreme_gradient_low_uniformity(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[:, 50:] = 255
        score = _estimate_lighting_uniformity(gray)
        assert score < 0.5, f"High gradient should have low uniformity, got {score}"

    def test_return_in_range(self):
        rng = np.random.default_rng(13)
        gray = rng.integers(0, 255, (80, 80), dtype=np.uint8)
        score = _estimate_lighting_uniformity(gray)
        assert 0.0 <= score <= 1.0


class TestAssessQuality:
    def _make_bgr(self, h=200, w=200):
        img = np.ones((h, w, 3), dtype=np.uint8) * 128
        return img

    def _make_lms_68(self):
        return [{"id": i, "x": float(100 + i), "y": 100.0, "confidence": 0.9} for i in range(68)]

    def test_returns_all_required_keys(self):
        bgr = self._make_bgr()
        lms = self._make_lms_68()
        result = assess_quality(bgr, 0.95, lms)
        for key in ["resolution", "blur_score", "noise", "compression_artifact_score",
                    "brightness", "contrast", "lighting_uniformity", "pose",
                    "occlusion_score", "face_visibility_pct", "detection_confidence"]:
            assert key in result, f"Missing key: {key}"

    def test_resolution_format(self):
        bgr = self._make_bgr(h=300, w=400)
        result = assess_quality(bgr, 0.9, [])
        assert result["resolution"] == "400x300"

    def test_noise_is_non_negative(self):
        bgr = self._make_bgr()
        result = assess_quality(bgr, 0.9, [])
        assert result["noise"] >= 0.0

    def test_compression_in_range(self):
        bgr = self._make_bgr()
        result = assess_quality(bgr, 0.9, [])
        assert 0.0 <= result["compression_artifact_score"] <= 1.0
