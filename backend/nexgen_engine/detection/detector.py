from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .alignment import estimate_pose
from .types import DetectedFace, FaceBox, HeadPose

logger = logging.getLogger(__name__)

# Detector input sizes, smallest first. Running a 640x640 detector over a 112px
# thumbnail costs ~10x more than it needs to; picking the smallest size that
# comfortably contains the image was the single biggest inference speed-up
# measured on this hardware.
_DET_SIZES = ((320, 320), (480, 480), (640, 640))

# Fraction of the image added as a border when the first detection pass finds
# nothing. See PaddedRetry below for why this is required, not optional.
_PAD_FRACTION = 0.4

# At or below this size an image is almost certainly an already-cropped face
# (thumbnails, mugshots, AgeDB-style databases), so the unpadded pass is skipped
# rather than run and discarded.
_CROP_SIZE_HINT = 200


@dataclass(frozen=True)
class DetectionOutcome:
    faces: tuple[DetectedFace, ...]
    padded: bool
    det_size: tuple[int, int]
    elapsed_ms: float


class InsightFaceDetector:
    """SCRFD/RetinaFace detection with five-point landmarks.

    Shares the loaded ``FaceAnalysis`` session with the recognizer so the model
    pack is paid for once.

    **Pad-and-retry.** A detector trained on photographs expects a face to
    occupy part of a scene. Hand it an image that is *only* a face -- a mugshot,
    a database thumbnail, a previously cropped probe -- and it finds nothing:
    there is no surrounding context and the face exceeds the largest anchor.
    Measured on AgeDB (112x112 crops), the first pass detected 0 of 120 faces;
    re-running with a 40% replicated border detected all 120 at ~0.8 confidence.
    Forensic probes are frequently already cropped, so this path is common, not
    exceptional.

    The border is replicated rather than filled, and every coordinate is mapped
    back to the original image, so nothing downstream sees invented pixels.
    """

    name = "scrfd_retinaface"
    produces_landmarks = True

    def __init__(self, analysis_app, min_confidence: float = 0.5) -> None:  # noqa: ANN001
        self._app = analysis_app
        # The detection model directly, NOT FaceAnalysis.get(). get() also runs
        # the recognition network over every detected face and attaches an
        # embedding that this pipeline throws away and recomputes from its own
        # aligned crop -- measured at ~3x the necessary ArcFace work per image.
        self._model = analysis_app.models["detection"]
        self.min_confidence = min_confidence

    def detect(self, image: Image.Image) -> list[DetectedFace]:
        return list(self.detect_detailed(image).faces)

    def detect_detailed(self, image: Image.Image) -> DetectionOutcome:
        import time

        started = time.perf_counter()
        bgr = _to_bgr(image)
        det_size = self._choose_det_size(image.size)

        # A face filling the whole frame is undetectable, so for images that are
        # already face-sized there is no point paying for a pass that will fail.
        pad_first = max(image.size) <= _CROP_SIZE_HINT
        faces: list[DetectedFace] = []
        padded = False

        if not pad_first:
            faces = self._run(bgr, det_size, offset=(0, 0))

        if not faces:
            pad = max(8, int(max(bgr.shape[:2]) * _PAD_FRACTION))
            import cv2

            bordered = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            # Coordinates come back in padded space; subtract the offset so the
            # caller only ever sees boxes and landmarks in original-image space.
            faces = self._run(bordered, det_size, offset=(-pad, -pad))
            padded = bool(faces)

        clipped = [
            DetectedFace(
                box=face.box.clipped(*image.size),
                confidence=face.confidence,
                landmarks=face.landmarks,
                pose=face.pose,
                detector=self.name,
            )
            for face in faces
        ]
        clipped.sort(key=lambda face: face.box.area, reverse=True)

        return DetectionOutcome(
            faces=tuple(clipped),
            padded=padded,
            det_size=det_size,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    # ----------------------------------------------------------- internal ---

    def _run(
        self,
        bgr: np.ndarray,
        det_size: tuple[int, int],
        offset: tuple[int, int],
    ) -> list[DetectedFace]:
        """One detection pass. Returns boxes, scores, and five landmarks."""
        # SCRFD reads its score threshold from the model, not from detect(), so
        # it is set once here rather than passed per call.
        if getattr(self._model, "det_thresh", None) != self.min_confidence:
            self._model.det_thresh = self.min_confidence

        boxes, landmark_sets = self._model.detect(bgr, input_size=det_size, metric="default")
        if boxes is None or len(boxes) == 0:
            return []

        shift = np.asarray(offset, dtype=np.float64)
        detected: list[DetectedFace] = []

        for position, row in enumerate(boxes):
            score = float(row[4])
            if score < self.min_confidence:
                continue

            landmarks = None
            if landmark_sets is not None and position < len(landmark_sets):
                landmarks = np.asarray(landmark_sets[position], dtype=np.float64) + shift

            detected.append(
                DetectedFace(
                    box=FaceBox(
                        left=int(round(float(row[0]) + shift[0])),
                        top=int(round(float(row[1]) + shift[1])),
                        right=int(round(float(row[2]) + shift[0])),
                        bottom=int(round(float(row[3]) + shift[1])),
                        confidence=score,
                    ),
                    confidence=score,
                    landmarks=landmarks,
                    pose=estimate_pose(landmarks)
                    if landmarks is not None and landmarks.shape == (5, 2)
                    else HeadPose(),
                    detector=self.name,
                )
            )
        return detected

    def _choose_det_size(self, size: tuple[int, int]) -> tuple[int, int]:
        """Smallest detector input that still covers the image, plus padding room."""
        longest = max(size) * (1 + 2 * _PAD_FRACTION)
        for candidate in _DET_SIZES:
            if longest <= candidate[0]:
                return candidate
        return _DET_SIZES[-1]


def build_detector(analysis_app, min_confidence: float = 0.5) -> InsightFaceDetector:  # noqa: ANN001
    """Build the detector.

    There is exactly one implementation. Earlier revisions fell back to an
    OpenCV Haar cascade (no landmarks, so no proper alignment) and finally to a
    centre crop that asserted a face was present without looking. Both have been
    removed: a centre crop is a fabricated detection, and a landmark-less crop
    silently degrades recognition. If InsightFace is unavailable the engine now
    fails at startup instead.
    """
    if analysis_app is None:
        raise ValueError("An InsightFace FaceAnalysis instance is required.")
    return InsightFaceDetector(analysis_app, min_confidence)


def _to_bgr(image: Image.Image) -> np.ndarray:
    """InsightFace expects OpenCV-style BGR uint8 arrays."""
    return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8)[:, :, ::-1])


__all__ = ["DetectionOutcome", "InsightFaceDetector", "build_detector"]
