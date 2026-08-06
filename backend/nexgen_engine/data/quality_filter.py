from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

from ..config import QualityConfig
from ..detection.types import DetectedFace
from ..utils import clamp

# 3x3 discrete Laplacian. Variance of the response is the standard blur proxy:
# a sharp image has strong second derivatives, a blurred one has almost none.
_LAPLACIAN_KERNEL = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# Sharpness is measured after resizing the face region so its shorter edge is
# 112px -- the scale the recognizer actually consumes. Raw Laplacian variance is
# strongly resolution-dependent: the identical scene rendered at 112px and
# 1600px measures 3311 and 280, a 12x drift, so one fixed cutoff cannot be
# right at more than one size. In production that surfaced as tack-sharp ID
# photographs with 250-400px faces measuring 3-7 (scored 1-3% sharp, flagged
# severe_blur, refused) while their content was fine. At the fixed 112px scale
# the same scene measures within +/-2% from 112px to 1600px of source size.
SHARPNESS_MATCH_EDGE = 112

# Light Gaussian before the Laplacian. Sensor noise is pure high-frequency
# energy: a defocused frame with mild noise out-measured a sharp one 2881 vs
# 1340 at native scale. The pre-blur costs a genuinely sharp image little (its
# edges survive) and stops noise reading as detail.
_SHARPNESS_PRE_BLUR = 0.6


# Conditions that make an image unusable, as opposed to merely imperfect.
# Only these veto an enrolment or a search.
BLOCKING_REASONS = frozenset(
    {
        "face_too_small",
        "brightness_out_of_range",
        "detection_confidence_below_minimum",
        "severe_blur",
    }
)


@dataclass(frozen=True)
class QualityReport:
    """Measured properties of the face region, not of the whole photograph.

    Scores are 0-1 where higher is better. ``score`` is the weighted aggregate;
    the components are exposed so an analyst can see *why* an image was refused
    rather than being handed one opaque number.

    Reasons are split deliberately. Blocking reasons mean the image cannot
    support a reliable comparison at all. Advisory reasons -- pose, moderate
    blur, low contrast -- mean the result deserves closer examiner attention but
    are not grounds for refusing the image: treating every flag as fatal caused
    perfectly usable enrolment photographs to be rejected on a single soft
    signal, and the pose estimate in particular is an approximation.

    ``laplacian_variance`` is measured at the fixed 112px match scale (see
    ``match_scale_sharpness``), not at the native crop size, so the same number
    means the same thing regardless of upload resolution.
    """

    score: float
    resolution_score: float
    brightness_score: float
    contrast_score: float
    sharpness_score: float
    pose_score: float
    detection_confidence: float
    face_pixels: int
    laplacian_variance: float
    accepted: bool
    reasons: tuple[str, ...]

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        return tuple(reason for reason in self.reasons if reason in BLOCKING_REASONS)

    @property
    def advisory_reasons(self) -> tuple[str, ...]:
        return tuple(reason for reason in self.reasons if reason not in BLOCKING_REASONS)

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "resolution": self.resolution_score,
            "brightness": self.brightness_score,
            "contrast": self.contrast_score,
            "sharpness": self.sharpness_score,
            "pose": self.pose_score,
            "detection_confidence": self.detection_confidence,
            "face_pixels": self.face_pixels,
            "laplacian_variance": self.laplacian_variance,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "blocking_reasons": list(self.blocking_reasons),
            "advisory_reasons": list(self.advisory_reasons),
        }


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian response, computed without OpenCV.

    Scale-dependent by nature: the same content measures higher at smaller
    sizes. Comparing values against a fixed threshold is only meaningful when
    every input is measured at one scale -- use ``match_scale_sharpness`` for
    that.
    """
    if gray.ndim != 2 or min(gray.shape) < 3:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(gray, (3, 3))
    response = np.einsum("ijkl,kl->ij", windows, _LAPLACIAN_KERNEL)
    return float(response.var())


def match_scale_sharpness(region: Image.Image) -> float:
    """Laplacian variance at the fixed 112px match scale, noise-suppressed.

    The region's shorter edge is resized to ``SHARPNESS_MATCH_EDGE`` (both up
    and down -- one measurement scale, so one threshold means one thing), then
    lightly smoothed so sensor noise does not read as detail. Values are
    comparable across input resolutions and against the fixed 220 normaliser.
    """
    gray = region.convert("L")
    width, height = gray.size
    short = min(width, height)
    if short < 3:
        return 0.0
    if short != SHARPNESS_MATCH_EDGE:
        scale = SHARPNESS_MATCH_EDGE / short
        gray = gray.resize(
            (
                max(SHARPNESS_MATCH_EDGE, round(width * scale)),
                max(SHARPNESS_MATCH_EDGE, round(height * scale)),
            ),
            Image.LANCZOS,
        )
    gray = gray.filter(ImageFilter.GaussianBlur(_SHARPNESS_PRE_BLUR))
    return laplacian_variance(np.asarray(gray, dtype=np.float64))


class ImageQualityFilter:
    """Scores whether a face region is good enough to search on."""

    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()

    def evaluate_bytes(self, image_bytes: bytes, face: DetectedFace | None = None) -> QualityReport:
        with Image.open(BytesIO(image_bytes)) as image:
            return self.evaluate(image, face)

    def evaluate(self, image: Image.Image, face: DetectedFace | None = None) -> QualityReport:
        """Measure the face region if one was detected, else the whole frame."""
        region = image
        face_pixels = min(image.size)
        detection_confidence = 0.0
        pose_score = 1.0
        reasons: list[str] = []

        if face is not None:
            box = face.box.clipped(*image.size)
            if box.width >= 2 and box.height >= 2:
                region = image.crop((box.left, box.top, box.right, box.bottom))
                face_pixels = min(box.width, box.height)
            detection_confidence = float(face.confidence)
            pose_score = self._pose_score(face)

            if detection_confidence < self.config.min_detection_confidence:
                reasons.append("detection_confidence_below_minimum")
            if abs(face.pose.yaw) > self.config.max_yaw:
                reasons.append("yaw_out_of_range")
            if abs(face.pose.pitch) > self.config.max_pitch:
                reasons.append("pitch_out_of_range")
            if abs(face.pose.roll) > self.config.max_roll:
                reasons.append("roll_out_of_range")

        gray = np.asarray(region.convert("L"), dtype=np.float64)
        brightness = float(gray.mean()) if gray.size else 0.0
        contrast = float(gray.std()) if gray.size else 0.0
        # Mean and deviation are scale-invariant so the native crop is fine for
        # them; the blur measure is not, so it is taken at the fixed match scale.
        blur = match_scale_sharpness(region)

        # Face height in pixels drives how much identity signal survives; below
        # ~60px ArcFace accuracy degrades sharply, and 220px is where it plateaus.
        resolution_score = clamp((face_pixels - self.config.min_face_pixels) / 160.0)
        brightness_score = 1.0 - clamp(abs(brightness - 128.0) / 128.0)
        contrast_score = clamp(contrast / 64.0)
        sharpness_score = clamp(blur / 220.0)

        score = (
            resolution_score * 0.24
            + brightness_score * 0.16
            + contrast_score * 0.16
            + sharpness_score * 0.24
            + pose_score * 0.20
        )

        if face_pixels < self.config.min_face_pixels:
            reasons.append("face_too_small")
        if not (self.config.min_brightness <= brightness <= self.config.max_brightness):
            reasons.append("brightness_out_of_range")
        if contrast_score < self.config.min_contrast_score:
            reasons.append("low_contrast")
        if sharpness_score < self.config.min_sharpness_score:
            # Mild softness is advisory; a near-total absence of high-frequency
            # detail means there is no identity information left to compare.
            reasons.append("severe_blur" if sharpness_score < 0.10 else "blur_risk")

        accepted = score >= self.config.min_quality_score and not (
            set(reasons) & BLOCKING_REASONS
        )

        return QualityReport(
            score=round(float(score), 4),
            resolution_score=round(resolution_score, 4),
            brightness_score=round(brightness_score, 4),
            contrast_score=round(contrast_score, 4),
            sharpness_score=round(sharpness_score, 4),
            pose_score=round(pose_score, 4),
            detection_confidence=round(detection_confidence, 4),
            face_pixels=int(face_pixels),
            laplacian_variance=round(blur, 2),
            accepted=bool(accepted),
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def keep_top(self, reports: list[QualityReport]) -> list[int]:
        """Indices of the best ``keep_top_fraction`` reports, best first."""
        if not reports:
            return []
        keep_count = max(1, int(np.ceil(len(reports) * self.config.keep_top_fraction)))
        ranked = sorted(enumerate(reports), key=lambda item: item[1].score, reverse=True)
        return [index for index, _ in ranked[:keep_count]]

    def _pose_score(self, face: DetectedFace) -> float:
        if not face.has_landmarks:
            # No landmarks means pose is unknown, not good. Score it neutrally so
            # it neither rewards nor punishes the aggregate.
            return 0.5
        yaw = clamp(1.0 - abs(face.pose.yaw) / max(self.config.max_yaw, 1e-6))
        pitch = clamp(1.0 - abs(face.pose.pitch) / max(self.config.max_pitch, 1e-6))
        roll = clamp(1.0 - abs(face.pose.roll) / max(self.config.max_roll, 1e-6))
        return float((yaw + pitch + roll) / 3.0)


__all__ = [
    "BLOCKING_REASONS",
    "SHARPNESS_MATCH_EDGE",
    "ImageQualityFilter",
    "QualityReport",
    "laplacian_variance",
    "match_scale_sharpness",
]
