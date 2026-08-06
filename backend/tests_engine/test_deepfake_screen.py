"""Tests for the multi-signal synthetic-media screen.

All imagery here is synthesised with controlled statistics, because the claims
under test are physical ("a pasted denoised face mismatches its background's
noise"), not semantic ("this looks like a person"). Threshold calibration was
performed against real AgeDB photographs -- see the slope and periodicity
bounds in deepfake_detector.py -- and the natural-image construction below
stays inside those photographic ranges by design.
"""

from __future__ import annotations

import json
from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageFilter, PngImagePlugin

from nexgen_engine.detection.types import FaceBox
from nexgen_engine.security.deepfake_detector import DeepfakeDetector, DeepfakeReport


@pytest.fixture(scope="module")
def detector() -> DeepfakeDetector:
    return DeepfakeDetector()


def natural_image(side: int = 512, seed: int = 0, exponent: float = 1.6) -> Image.Image:
    """A photograph-statistics stand-in: 1/f power-law content plus sensor noise.

    Built by FILTERING white noise rather than assigning spectrum amplitudes
    directly, so the spectrum keeps the chi-square roughness a real capture
    has. The exponent puts the power-spectrum slope in the middle of the
    measured photographic range, and the additive Gaussian noise gives it the
    sensor noise floor a real capture carries.
    """
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, (side, side))
    frequency = np.fft.fftfreq(side)
    fx, fy = np.meshgrid(frequency, frequency)
    radius = np.sqrt(fx**2 + fy**2)
    radius[0, 0] = abs(frequency[1])
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * radius**-exponent))
    field = (field - field.min()) / (field.max() - field.min() + 1e-12) * 200 + 20
    field = field + rng.normal(0, 4.0, field.shape)
    gray = Image.fromarray(np.clip(field, 0, 255).astype(np.uint8))
    return gray.convert("RGB")


def centered_box(image: Image.Image, fraction: float = 0.5) -> FaceBox:
    width, height = image.size
    box_w, box_h = int(width * fraction), int(height * fraction)
    left, top = (width - box_w) // 2, (height - box_h) // 2
    return FaceBox(left=left, top=top, right=left + box_w, bottom=top + box_h, confidence=0.99)


# ------------------------------------------------------------- provenance ----


class TestProvenanceMetadata:
    def test_generation_parameters_chunk_is_decisive(self, detector):
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "portrait, Steps: 30, Sampler: DPM++ 2M, Seed: 1234")
        buffer = BytesIO()
        natural_image(seed=1).save(buffer, "PNG", pnginfo=info)
        raw = buffer.getvalue()
        report = detector.analyze(Image.open(BytesIO(raw)), raw_bytes=raw)

        assert report.flagged
        assert report.band == "high"
        assert "generative_metadata_present" in report.reasons
        assert "synthetic_media_risk" in report.reasons

    def test_generator_named_in_exif_software_is_decisive(self, detector):
        exif = Image.Exif()
        exif[305] = "Stable Diffusion XL 1.0"  # Software
        buffer = BytesIO()
        natural_image(seed=2).save(buffer, "JPEG", quality=92, exif=exif)
        raw = buffer.getvalue()
        report = detector.analyze(Image.open(BytesIO(raw)), raw_bytes=raw)

        assert report.flagged
        assert "generative_metadata_present" in report.reasons

    def test_iptc_trained_algorithmic_media_marker_is_decisive(self, detector):
        info = PngImagePlugin.PngInfo()
        info.add_text("Description", "digitalSourceType: trainedAlgorithmicMedia")
        buffer = BytesIO()
        natural_image(seed=3).save(buffer, "PNG", pnginfo=info)
        raw = buffer.getvalue()
        report = detector.analyze(Image.open(BytesIO(raw)), raw_bytes=raw)

        assert report.flagged
        assert "generative_metadata_present" in report.reasons

    def test_ordinary_editing_software_does_not_flag(self, detector):
        """A scan cropped in Photoshop is still a genuine photograph."""
        exif = Image.Exif()
        exif[305] = "Adobe Photoshop 2026"
        buffer = BytesIO()
        natural_image(seed=4).save(buffer, "JPEG", quality=92, exif=exif)
        raw = buffer.getvalue()
        report = detector.analyze(Image.open(BytesIO(raw)), raw_bytes=raw)

        assert not report.flagged
        assert "generative_metadata_present" not in report.reasons

    def test_camera_exif_reads_as_camera_provenance(self, detector):
        exif = Image.Exif()
        exif[271] = "Canon"  # Make
        exif[272] = "Canon EOS R5"  # Model
        buffer = BytesIO()
        natural_image(seed=5).save(buffer, "JPEG", quality=92, exif=exif)
        raw = buffer.getvalue()
        report = detector.analyze(Image.open(BytesIO(raw)), raw_bytes=raw)

        assert not report.flagged
        provenance = next(s for s in report.signals if s.name == "provenance")
        assert "camera EXIF" in provenance.detail

    def test_marker_needs_word_boundaries(self, detector):
        """REGRESSION GUARD: 'roop' must not fire inside 'drooped', etc."""
        exif = Image.Exif()
        exif[305] = "drooped organ dallesport imagenes"
        buffer = BytesIO()
        natural_image(seed=6).save(buffer, "JPEG", quality=92, exif=exif)
        raw = buffer.getvalue()
        report = detector.analyze(Image.open(BytesIO(raw)), raw_bytes=raw)

        assert "generative_metadata_present" not in report.reasons


# --------------------------------------------------------- physical signals --


class TestPhysicalSignals:
    def test_natural_statistics_pass_clean(self, detector):
        report = detector.analyze(natural_image(seed=10), face_box=None)
        assert not report.flagged
        assert not report.review_advised

    def test_natural_statistics_pass_clean_with_box(self, detector):
        image = natural_image(seed=11)
        report = detector.analyze(image, face_box=centered_box(image))
        assert not report.flagged
        assert not report.review_advised

    def test_denoised_face_on_noisy_background_flags(self, detector):
        """The face-swap signature: the pasted face lacks the background's noise."""
        image = natural_image(seed=12)
        box = centered_box(image)
        face = (
            image.crop((box.left, box.top, box.right, box.bottom))
            .filter(ImageFilter.MedianFilter(5))
            .filter(ImageFilter.GaussianBlur(1.0))
        )
        composite = image.copy()
        composite.paste(face, (box.left, box.top))

        report = detector.analyze(composite, face_box=box)
        assert "face_background_noise_mismatch" in report.reasons
        assert report.flagged

    def test_noise_mismatch_is_directional(self, detector):
        """A noisy face on a flat background is ordinary photography (bokeh,
        studio walls) and must NOT fire the mismatch signal."""
        image = natural_image(seed=13)
        box = centered_box(image)
        background = image.filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.GaussianBlur(1.0))
        composite = background.copy()
        composite.paste(image.crop((box.left, box.top, box.right, box.bottom)), (box.left, box.top))

        report = detector.analyze(composite, face_box=box)
        assert "face_background_noise_mismatch" not in report.reasons

    def test_checkerboard_artefact_flags(self, detector):
        """The transposed-convolution checkerboard: a faint periodic grid
        riding on natural content concentrates energy into an isolated
        spectral tone no photograph produces."""
        image = natural_image(seed=14)
        array = np.asarray(image, dtype=np.float64)
        yy, xx = np.meshgrid(np.arange(image.height), np.arange(image.width), indexing="ij")
        checker = 5.0 * ((((xx // 2) + (yy // 2)) % 2) * 2 - 1)
        fake = Image.fromarray(np.clip(array + checker[:, :, None], 0, 255).astype(np.uint8))

        report = detector.analyze(fake, face_box=centered_box(fake))
        assert "periodic_upsampling_artefacts" in report.reasons
        assert report.flagged

    def test_oversmoothing_asks_for_review_without_hard_flag(self, detector):
        """Blur alone is a quality problem, not proof of synthesis: the screen
        must ask for review, and must not brand every blurry surveillance
        still a deepfake."""
        blurred = natural_image(seed=15).filter(ImageFilter.GaussianBlur(2.5))
        report = detector.analyze(blurred, face_box=centered_box(blurred))
        assert report.review_advised
        assert report.score < 0.85  # never near-certain from smoothness alone


# ---------------------------------------------------------------- contract ---


class TestReportContract:
    def test_report_is_json_serialisable(self, detector):
        report = detector.analyze(natural_image(seed=20))
        payload = json.loads(json.dumps(report.as_dict()))
        assert set(payload) >= {
            "score",
            "band",
            "flagged",
            "review_advised",
            "signals",
            "reasons",
            "method",
            "certified",
        }
        assert payload["certified"] is False

    def test_risk_score_compat_matches_analyze(self, detector):
        image = natural_image(seed=21)
        assert detector.risk_score(image) == detector.analyze(image).score

    def test_score_is_bounded_and_deterministic(self, detector):
        image = natural_image(seed=22)
        first = detector.analyze(image, face_box=centered_box(image))
        second = detector.analyze(image, face_box=centered_box(image))
        assert 0.0 <= first.score <= 1.0
        assert first.score == second.score

    def test_band_tracks_thresholds(self):
        detector = DeepfakeDetector(alert_threshold=0.65, review_threshold=0.45)
        assert detector._band(0.9) == "high"
        assert detector._band(0.5) == "elevated"
        assert detector._band(0.3) == "moderate"
        assert detector._band(0.1) == "minimal"

    def test_review_threshold_never_exceeds_alert(self):
        detector = DeepfakeDetector(alert_threshold=0.5, review_threshold=0.9)
        assert detector.review_threshold <= detector.alert_threshold

    def test_tiny_image_does_not_crash(self, detector):
        tiny = natural_image(seed=23).resize((40, 40))
        report = detector.analyze(tiny)
        assert isinstance(report, DeepfakeReport)

    def test_degenerate_box_is_ignored(self, detector):
        image = natural_image(seed=24)
        sliver = FaceBox(left=0, top=0, right=8, bottom=8, confidence=0.9)
        report = detector.analyze(image, face_box=sliver)
        assert isinstance(report, DeepfakeReport)
