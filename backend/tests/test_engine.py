from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from nexgen_engine.config import EngineConfig
from nexgen_engine.data.quality_filter import ImageQualityFilter, laplacian_variance
from nexgen_engine.detection.alignment import (
    ARCFACE_REFERENCE_5PT,
    estimate_pose,
    norm_crop,
    umeyama_similarity,
)
from nexgen_engine.detection.types import DetectedFace, FaceBox
from nexgen_engine.inference.cohort_normalizer import CohortNormalizer
from nexgen_engine.inference.pipeline import InvalidImageError, decode_image
from nexgen_engine.inference.score_fusion import (
    DECISION_MATCH,
    DECISION_NO_MATCH,
    DECISION_REJECTED,
    DECISION_REVIEW,
    DECISION_UNAVAILABLE,
    DecisionEngine,
    ScoreNormalizer,
)
from nexgen_engine.security.template_encryption import (
    TemplateDecryptionError,
    TemplateEncryptor,
)

from .conftest import image_bytes, noise_image

# Pipeline and recognition behaviour live in test_recognition_engine.py, which
# runs the real model against real faces. Everything here is pure logic that
# needs no weights: geometry, quality metrics, decoding, crypto, decisions.


# ------------------------------------------------------------- alignment ----


class TestAlignment:
    def test_umeyama_recovers_a_known_transform(self):
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        angle = np.pi / 6
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        target = (source @ rotation.T) * 2.5 + np.array([7.0, -3.0])

        transform = umeyama_similarity(source, target)
        recovered = source @ transform[:2, :2].T + transform[:2, 2]

        np.testing.assert_allclose(recovered, target, atol=1e-9)

    def test_umeyama_rejects_degenerate_points(self):
        collinear = np.zeros((5, 2))
        with pytest.raises(ValueError):
            umeyama_similarity(collinear, ARCFACE_REFERENCE_5PT)

    def test_norm_crop_maps_landmarks_onto_the_reference_layout(self):
        """The whole point of alignment: landmarks must land where ArcFace expects."""
        image = noise_image(seed=1, size=400)
        # Landmarks scaled and shifted away from the canonical layout.
        landmarks = ARCFACE_REFERENCE_5PT * 2.0 + np.array([60.0, 40.0])

        transform = umeyama_similarity(landmarks, ARCFACE_REFERENCE_5PT)
        mapped = landmarks @ transform[:2, :2].T + transform[:2, 2]
        np.testing.assert_allclose(mapped, ARCFACE_REFERENCE_5PT, atol=1e-6)

        crop = norm_crop(image, landmarks)
        assert crop.size == (112, 112)

    def test_pose_is_neutral_for_a_frontal_layout(self):
        pose = estimate_pose(ARCFACE_REFERENCE_5PT)
        assert abs(pose.yaw) < 5.0
        assert abs(pose.roll) < 5.0

    def test_pose_detects_a_turned_head(self):
        turned = ARCFACE_REFERENCE_5PT.copy()
        turned[2, 0] += 18.0  # push the nose toward one eye
        assert estimate_pose(turned).yaw > 15.0

    def test_pose_detects_roll(self):
        angle = np.pi / 8
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rolled = ARCFACE_REFERENCE_5PT @ rotation.T
        assert abs(estimate_pose(rolled).roll) > 15.0


# --------------------------------------------------------------- quality ----


class TestQuality:
    def test_laplacian_variance_separates_sharp_from_blurred(self):
        from PIL import ImageFilter

        original = noise_image(seed=2)
        sharp = np.asarray(original.convert("L"), dtype=np.float64)
        blurred = np.asarray(
            original.filter(ImageFilter.GaussianBlur(6)).convert("L"), dtype=np.float64
        )
        assert laplacian_variance(sharp) > laplacian_variance(blurred) * 2

    def test_uniform_image_is_rejected(self):
        flat = Image.new("RGB", (300, 300), (128, 128, 128))
        report = ImageQualityFilter().evaluate(flat)
        assert not report.accepted
        assert "low_contrast" in report.reasons or "blur_risk" in report.reasons

    def test_dark_image_flags_brightness(self):
        dark = Image.new("RGB", (300, 300), (8, 8, 8))
        assert "brightness_out_of_range" in ImageQualityFilter().evaluate(dark).reasons

    def test_small_face_is_flagged(self):
        image = noise_image(seed=3, size=400)
        tiny = DetectedFace(box=FaceBox(10, 10, 40, 40, 0.99), confidence=0.99)
        assert "face_too_small" in ImageQualityFilter().evaluate(image, tiny).reasons

    def test_quality_measures_the_face_not_the_frame(self):
        """A sharp face on a noisy background must not be scored on the background."""
        image = noise_image(seed=4, size=400)
        whole = ImageQualityFilter().evaluate(image)
        face = DetectedFace(box=FaceBox(100, 100, 300, 300, 0.99), confidence=0.99)
        cropped = ImageQualityFilter().evaluate(image, face)
        assert cropped.face_pixels == 200
        assert whole.face_pixels == 400


# -------------------------------------------------------------- decoding ----


class TestDecoding:
    def test_rejects_empty_payload(self):
        with pytest.raises(InvalidImageError):
            decode_image(b"")

    def test_rejects_non_image_bytes(self):
        with pytest.raises(InvalidImageError):
            decode_image(b"this is definitely not a JPEG")

    def test_decodes_common_formats(self):
        for fmt in ("JPEG", "PNG", "BMP"):
            decoded = decode_image(image_bytes(noise_image(seed=5), fmt))
            assert decoded.mode == "RGB"


# ------------------------------------------------------------- templates ----


class TestTemplateEncryption:
    def test_round_trip(self):
        encryptor = TemplateEncryptor.generate()
        template = np.random.default_rng(0).normal(size=512).astype(np.float32)
        sealed = encryptor.encrypt(template, "tenant-a")
        np.testing.assert_allclose(encryptor.decrypt(sealed), template, rtol=1e-6)

    def test_ciphertext_differs_across_calls(self):
        """A fixed nonce would leak that two subjects share an enrolment image."""
        encryptor = TemplateEncryptor.generate()
        template = np.ones(512, dtype=np.float32)
        first = encryptor.encrypt(template, "tenant-a")
        second = encryptor.encrypt(template, "tenant-a")
        assert first.ciphertext != second.ciphertext

    def test_cross_tenant_ciphertext_fails_to_decrypt(self):
        """Moving a row between tenants must break, not silently succeed."""
        encryptor = TemplateEncryptor.generate()
        sealed = encryptor.encrypt(np.ones(512, dtype=np.float32), "tenant-a")
        forged = type(sealed)(
            nonce=sealed.nonce,
            ciphertext=sealed.ciphertext,
            dimensions=sealed.dimensions,
            tenant_id="tenant-b",
        )
        with pytest.raises(TemplateDecryptionError):
            encryptor.decrypt(forged)

    def test_wrong_key_fails(self):
        sealed = TemplateEncryptor.generate().encrypt(np.ones(512, dtype=np.float32), "tenant-a")
        with pytest.raises(TemplateDecryptionError):
            TemplateEncryptor.generate().decrypt(sealed)

    def test_tampered_ciphertext_is_detected(self):
        import base64

        encryptor = TemplateEncryptor.generate()
        sealed = encryptor.encrypt(np.ones(512, dtype=np.float32), "tenant-a")
        raw = bytearray(base64.b64decode(sealed.ciphertext))
        raw[0] ^= 0xFF
        tampered = type(sealed)(
            nonce=sealed.nonce,
            ciphertext=base64.b64encode(bytes(raw)).decode(),
            dimensions=sealed.dimensions,
            tenant_id=sealed.tenant_id,
        )
        with pytest.raises(TemplateDecryptionError):
            encryptor.decrypt(tampered)

    def test_rejects_wrong_key_length(self):
        with pytest.raises(ValueError):
            TemplateEncryptor(b"too-short")


# -------------------------------------------------------------- decisions ---


class TestDecisionEngine:
    @pytest.fixture
    def decisions(self):
        return DecisionEngine(EngineConfig())

    def test_unavailable_engine_never_claims_a_match(self, decisions):
        decision = decisions.decide(
            top_score=0.99, recognition_capable=False, probe_accepted=True, gallery_size=100
        )
        assert decision.label == DECISION_UNAVAILABLE
        assert decision.confidence == 0.0

    def test_rejected_probe_is_not_searched(self, decisions):
        decision = decisions.decide(
            top_score=0.9,
            recognition_capable=True,
            probe_accepted=False,
            probe_reasons=("blur_risk",),
            gallery_size=100,
        )
        assert decision.label == DECISION_REJECTED

    def test_empty_gallery_returns_no_match(self, decisions):
        decision = decisions.decide(
            top_score=0.0, recognition_capable=True, probe_accepted=True, gallery_size=0
        )
        assert decision.label == DECISION_NO_MATCH

    def test_low_score_is_no_match(self, decisions):
        decision = decisions.decide(
            top_score=0.15, recognition_capable=True, probe_accepted=True, gallery_size=100, margin=0.1
        )
        assert decision.label == DECISION_NO_MATCH

    def test_borderline_score_goes_to_review(self, decisions):
        """A score inside the review band must reach a human.

        The probe score is DERIVED from the configured thresholds, not
        hardcoded. It previously used a literal 0.36, which sat in the review
        band only while the thresholds were 0.32/0.42. When they were
        recalibrated to 0.2153/0.2871 for false-match control (BENCHMARKS.md
        section 5c), 0.36 became a clear match and this test failed -- it was
        asserting a number, not a behaviour.
        """
        t = decisions.config.thresholds
        midband = (t.review + t.match) / 2
        assert t.review < midband < t.match, "fixture assumption: a review band exists"

        decision = decisions.decide(
            top_score=midband,
            recognition_capable=True,
            probe_accepted=True,
            gallery_size=100,
            margin=0.1,
        )
        assert decision.label == DECISION_REVIEW
        assert "score_in_review_band" in decision.reasons

    def test_clear_score_is_a_candidate_match(self, decisions):
        decision = decisions.decide(
            top_score=0.72, recognition_capable=True, probe_accepted=True, gallery_size=100, margin=0.2
        )
        assert decision.label == DECISION_MATCH
        assert "not a positive identification" in decision.explanation

    def test_near_tie_on_a_large_gallery_forces_review(self, decisions):
        """A high score that barely beats the runner-up is the shape of a false match."""
        decision = decisions.decide(
            top_score=0.72,
            recognition_capable=True,
            probe_accepted=True,
            gallery_size=5000,
            margin=0.01,
            candidate_count=4,
        )
        assert decision.label == DECISION_REVIEW
        assert "low_margin_over_runner_up" in decision.reasons

    def test_a_lone_candidate_is_not_treated_as_a_tie(self, decisions):
        """With one candidate the margin is 0.0, which means "nothing else came
        close" -- the opposite of a dead heat. Reading it as a tie would send
        the strongest possible result to review and word it alarmingly."""
        decision = decisions.decide(
            top_score=0.72,
            recognition_capable=True,
            probe_accepted=True,
            gallery_size=5000,
            margin=0.0,
            candidate_count=1,
        )
        assert decision.label == DECISION_MATCH
        assert "low_margin_over_runner_up" not in decision.reasons
        assert "no other enrolled subject scored above" in decision.explanation

    def test_probe_flags_downgrade_a_passing_score(self, decisions):
        decision = decisions.decide(
            top_score=0.80,
            recognition_capable=True,
            probe_accepted=True,
            probe_reasons=("liveness_below_threshold",),
            gallery_size=100,
            margin=0.3,
        )
        assert decision.label == DECISION_REVIEW


class TestScoreNormalizer:
    def test_margin_is_the_top_two_gap(self):
        assert ScoreNormalizer.margin(np.array([0.9, 0.6, 0.2])) == pytest.approx(0.3)

    def test_margin_of_single_score_is_zero(self):
        assert ScoreNormalizer.margin(np.array([0.9])) == 0.0

    def test_z_scores_are_centred(self):
        scores = ScoreNormalizer.z_scores(np.array([0.1, 0.2, 0.3, 0.4]))
        assert scores.mean() == pytest.approx(0.0, abs=1e-9)

    def test_identical_scores_do_not_divide_by_zero(self):
        np.testing.assert_array_equal(ScoreNormalizer.z_scores(np.ones(5)), np.zeros(5))


class TestCohortNormalizer:
    def test_small_cohort_returns_the_raw_score(self):
        normalizer = CohortNormalizer()
        normalizer.set_cohort(np.random.default_rng(0).normal(size=(3, 512)).astype(np.float32))
        assert normalizer.normalize_score(np.ones(512, dtype=np.float32), 0.7) == 0.7

    def test_normalization_does_not_mutate_the_template(self):
        """Regression: an earlier version adjusted the embedding itself, which made
        a stored identity depend on unrelated search history."""
        normalizer = CohortNormalizer()
        normalizer.set_cohort(np.random.default_rng(1).normal(size=(50, 512)).astype(np.float32))
        probe = np.random.default_rng(2).normal(size=512).astype(np.float32)
        before = probe.copy()
        normalizer.normalize_score(probe, 0.5)
        np.testing.assert_array_equal(probe, before)


