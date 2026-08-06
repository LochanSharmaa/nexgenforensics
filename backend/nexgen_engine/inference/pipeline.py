from __future__ import annotations

import time
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import EngineConfig
from ..data.quality_filter import ImageQualityFilter, QualityReport
from ..detection.alignment import FaceAligner
from ..detection.types import DetectedFace
from ..runtime import EngineRuntime
from ..security.deepfake_detector import DeepfakeDetector, DeepfakeReport
from ..security.liveness import LivenessDetector, LivenessReport
from ..utils import l2_normalize


class NoFaceDetectedError(ValueError):
    """Raised when an image contains no usable face."""


class InvalidImageError(ValueError):
    """Raised when the supplied bytes are not a decodable image."""


@dataclass(frozen=True)
class StageTimings:
    """Per-stage wall-clock cost, in milliseconds."""

    decode_ms: float = 0.0
    detect_ms: float = 0.0
    align_ms: float = 0.0
    embed_ms: float = 0.0
    quality_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "decode_ms": self.decode_ms,
            "detect_ms": self.detect_ms,
            "align_ms": self.align_ms,
            "embed_ms": self.embed_ms,
            "quality_ms": self.quality_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True)
class RecognitionResult:
    """Everything derived from one probe image.

    ``embedding`` is the L2-normalized template used for matching: the raw model
    output averaged over flip-TTA and renormalized. It is deliberately not
    adjusted by any query-dependent state, so the same image always produces the
    same template regardless of what was searched before it.
    """

    embedding: np.ndarray
    face: DetectedFace
    quality: QualityReport
    liveness: LivenessReport
    deepfake_risk: float
    faces_detected: int
    detector_name: str
    padded_detection: bool
    timings: StageTimings
    review_required: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    # Retained for API compatibility. The engine cannot start without a real
    # model, so reaching this object at all means recognition is live.
    recognition_capable: bool = True

    # Full synthetic-media screen behind ``deepfake_risk``. Optional so that
    # results constructed by older callers stay valid; the pipeline always
    # populates it.
    deepfake: DeepfakeReport | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "quality": self.quality.as_dict(),
            "liveness": self.liveness.as_dict(),
            "deepfake_risk": self.deepfake_risk,
            "deepfake": self.deepfake.as_dict() if self.deepfake else None,
            "faces_detected": self.faces_detected,
            "recognition_capable": True,
            "detector": self.detector_name,
            "padded_detection": self.padded_detection,
            "timings": self.timings.as_dict(),
            "review_required": self.review_required,
            "reasons": list(self.reasons),
            "box": {
                "left": self.face.box.left,
                "top": self.face.box.top,
                "right": self.face.box.right,
                "bottom": self.face.box.bottom,
            },
            "pose": {
                "yaw": self.face.pose.yaw,
                "pitch": self.face.pose.pitch,
                "roll": self.face.pose.roll,
            },
        }


class FacialRecognitionPipeline:
    """Image bytes in, comparable biometric template out.

    Stages: decode -> detect -> select face -> quality -> align -> embed.

    Two things this deliberately does NOT do, because both corrupt matching:

    * It does not fuse unrelated backbones through a random projection. A
      projection never trained jointly with the encoder destroys the metric
      structure ArcFace learned.
    * It does not adjust the stored template using statistics from previous
      queries. Cohort normalization belongs at the score level, where it cannot
      make a template depend on search history.
    """

    def __init__(self, config: EngineConfig | None = None, runtime: EngineRuntime | None = None) -> None:
        self.config = config or EngineConfig()
        self.runtime = runtime or EngineRuntime(self.config)
        self.quality_filter = ImageQualityFilter(self.config.quality)
        self.aligner = FaceAligner()
        self.liveness = LivenessDetector(self.config.security.liveness_threshold)
        self.deepfake = DeepfakeDetector(
            self.config.security.deepfake_threshold,
            self.config.security.deepfake_review_threshold,
        )

    # ------------------------------------------------------------------ api --

    def encode_bytes(self, image_bytes: bytes) -> RecognitionResult:
        started = time.perf_counter()
        image = decode_image(image_bytes)
        decode_ms = (time.perf_counter() - started) * 1000
        # The raw bytes travel along so the synthetic-media screen can read
        # provenance metadata (EXIF, PNG chunks, C2PA) that decoding discards.
        return self.encode_image(image, decode_ms=decode_ms, started=started, raw_bytes=image_bytes)

    def encode_image(
        self,
        image: Image.Image,
        decode_ms: float = 0.0,
        started: float | None = None,
        raw_bytes: bytes | None = None,
    ) -> RecognitionResult:
        started = started if started is not None else time.perf_counter()
        image = image.convert("RGB")

        outcome = self.runtime.detector.detect_detailed(image)
        if not outcome.faces:
            raise NoFaceDetectedError(
                "No face was detected. The image may not contain a face, the face may be too "
                "small, or the detection confidence threshold may be too high."
            )

        face = self._select_face(list(outcome.faces))

        mark = time.perf_counter()
        quality = self.quality_filter.evaluate(image, face)
        quality_ms = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        crop = self.aligner.align(image, face)
        align_ms = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        embedding = self._embed(crop)
        embed_ms = (time.perf_counter() - mark) * 1000

        liveness = self.liveness.analyze(crop)
        # The screen runs on the ORIGINAL image around the detected box, plus
        # the raw bytes for provenance metadata. The aligned 112 px crop it
        # used to receive had already been resampled below the resolution
        # where generator artefacts survive.
        deepfake = self.deepfake.analyze(image, face_box=face.box, raw_bytes=raw_bytes)

        reasons = list(quality.reasons)
        if not liveness.passed:
            reasons.append("liveness_below_threshold")
            reasons.extend(liveness.reasons)
        reasons.extend(deepfake.reasons)
        if len(outcome.faces) > 1:
            reasons.append("multiple_faces_detected")

        return RecognitionResult(
            embedding=embedding,
            face=face,
            quality=quality,
            liveness=liveness,
            deepfake_risk=deepfake.score,
            deepfake=deepfake,
            faces_detected=len(outcome.faces),
            detector_name=self.runtime.detector.name,
            padded_detection=outcome.padded,
            timings=StageTimings(
                decode_ms=round(decode_ms, 2),
                detect_ms=outcome.elapsed_ms,
                align_ms=round(align_ms, 2),
                embed_ms=round(embed_ms, 2),
                quality_ms=round(quality_ms, 2),
                total_ms=round((time.perf_counter() - started) * 1000, 2),
            ),
            review_required=bool(reasons),
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def encode_all_faces(self, image: Image.Image) -> list[RecognitionResult]:
        """Encode every detected face. Used by group-photo and batch intake.

        All crops go through the recognizer in a single batch, which is
        substantially cheaper than one call per face.
        """
        image = image.convert("RGB")
        outcome = self.runtime.detector.detect_detailed(image)
        if not outcome.faces:
            return []

        faces = list(outcome.faces)
        crops = [self.aligner.align(image, face) for face in faces]

        mark = time.perf_counter()
        embeddings = self._embed_many(crops)
        embed_ms = (time.perf_counter() - mark) * 1000

        results: list[RecognitionResult] = []
        for face, crop, embedding in zip(faces, crops, embeddings):
            quality = self.quality_filter.evaluate(image, face)
            liveness = self.liveness.analyze(crop)
            deepfake = self.deepfake.analyze(image, face_box=face.box)
            reasons = list(quality.reasons)
            if not liveness.passed:
                reasons.append("liveness_below_threshold")
            reasons.extend(deepfake.reasons)
            results.append(
                RecognitionResult(
                    embedding=embedding,
                    face=face,
                    quality=quality,
                    liveness=liveness,
                    deepfake_risk=deepfake.score,
                    deepfake=deepfake,
                    faces_detected=len(faces),
                    detector_name=self.runtime.detector.name,
                    padded_detection=outcome.padded,
                    timings=StageTimings(
                        detect_ms=outcome.elapsed_ms,
                        embed_ms=round(embed_ms / max(len(faces), 1), 2),
                    ),
                    review_required=bool(reasons),
                    reasons=tuple(dict.fromkeys(reasons)),
                )
            )
        return results

    # -------------------------------------------------------------- internal --

    def _select_face(self, faces: list[DetectedFace]) -> DetectedFace:
        """Pick the subject face: the largest that clears the confidence gate."""
        for face in faces:
            if face.confidence >= self.config.quality.min_detection_confidence:
                return face
        return faces[0]

    def _embed(self, crop: Image.Image) -> np.ndarray:
        """Template for one aligned crop, averaged with its mirror.

        Flip-TTA is a genuine small gain for ArcFace: a face and its mirror
        should encode to the same identity, so averaging cancels part of the
        pose-specific noise. Both crops go through in one batch.
        """
        crops = [crop, ImageOps.mirror(crop)] if self.config.use_flip_tta else [crop]
        embeddings = self.runtime.recognizer.embed_batch(crops)
        return l2_normalize(embeddings.mean(axis=0).astype(np.float32))

    def _embed_many(self, crops: list[Image.Image]) -> list[np.ndarray]:
        """Batch-embed several crops, applying flip-TTA in the same batch."""
        if not crops:
            return []
        if not self.config.use_flip_tta:
            return list(self.runtime.recognizer.embed_batch(crops))

        batch = list(crops) + [ImageOps.mirror(crop) for crop in crops]
        embeddings = self.runtime.recognizer.embed_batch(batch)
        count = len(crops)
        return [
            l2_normalize(((embeddings[i] + embeddings[i + count]) / 2.0).astype(np.float32))
            for i in range(count)
        ]


def decode_image(image_bytes: bytes) -> Image.Image:
    """Decode bytes to RGB, honouring EXIF orientation.

    Phone photos routinely carry an EXIF rotation flag. Ignoring it feeds the
    detector a sideways face and silently tanks the match rate.
    """
    if not image_bytes:
        raise InvalidImageError("Empty image payload.")
    try:
        with Image.open(BytesIO(image_bytes)) as handle:
            handle.load()
            image = ImageOps.exif_transpose(handle).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Could not decode image: {exc}") from exc

    # Reject degenerate geometry HERE, where it can still be a typed error.
    #
    # SCRFD scales the image to fit the detector input while preserving aspect
    # ratio: new_height = int(new_width * height / width). For a 4000x1 strip
    # that rounds to 0, and cv2.resize then raises a bare cv2.error
    # ("inv_scale_x > 0") from deep inside insightface. That exception is not
    # one the API maps to a 4xx, so it surfaced as a 500 -- an unreadable
    # failure for the operator and, in a batch, one that killed the request.
    #
    # Found by tests_engine/test_adversarial_input.py. A 1px-tall strip is a
    # realistic accident from a bad crop, not just a synthetic attack.
    #
    # The floor is the detector's minimum useful edge, not 1px: an image
    # smaller than this cannot contain a detectable face, so rejecting it with
    # a clear message beats letting the detector return nothing.
    min_edge = 16
    if image.width < min_edge or image.height < min_edge:
        raise InvalidImageError(
            f"Image is {image.width}x{image.height}; each side must be at least "
            f"{min_edge}px for face detection to be possible."
        )

    # Guard the ratio itself as well as the sides: 4000x20 clears the floor
    # above but still scales to a sub-pixel height at detector input size.
    longest, shortest = max(image.width, image.height), min(image.width, image.height)
    max_ratio = 50
    if longest / shortest > max_ratio:
        raise InvalidImageError(
            f"Image aspect ratio is {longest / shortest:.0f}:1 "
            f"({image.width}x{image.height}); the maximum supported is {max_ratio}:1. "
            "This usually means the image was cropped incorrectly."
        )

    return image


__all__ = [
    "FacialRecognitionPipeline",
    "InvalidImageError",
    "NoFaceDetectedError",
    "RecognitionResult",
    "StageTimings",
    "decode_image",
]
