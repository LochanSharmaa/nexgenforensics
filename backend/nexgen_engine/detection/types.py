from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FaceBox:
    left: int
    top: int
    right: int
    bottom: int
    confidence: float

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    def clipped(self, image_width: int, image_height: int) -> "FaceBox":
        return FaceBox(
            left=max(0, min(self.left, image_width)),
            top=max(0, min(self.top, image_height)),
            right=max(0, min(self.right, image_width)),
            bottom=max(0, min(self.bottom, image_height)),
            confidence=self.confidence,
        )


@dataclass(frozen=True)
class HeadPose:
    """Coarse head orientation in degrees, estimated from five landmarks.

    This is a geometric approximation, not a calibrated 3D pose solver. It is
    accurate enough to reject grossly off-angle probes but should not be
    reported to an analyst as a measured pose value.
    """

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True)
class DetectedFace:
    """One detected face.

    ``landmarks`` holds the five ArcFace reference points in image pixel
    coordinates, ordered: left eye, right eye, nose tip, left mouth corner,
    right mouth corner. It is ``None`` when the active detector cannot produce
    landmarks, in which case alignment degrades to a bounding-box crop.
    """

    box: FaceBox
    confidence: float
    landmarks: np.ndarray | None = None
    pose: HeadPose = field(default_factory=HeadPose)
    detector: str = "unknown"

    @property
    def has_landmarks(self) -> bool:
        return self.landmarks is not None and np.asarray(self.landmarks).shape == (5, 2)


__all__ = ["DetectedFace", "FaceBox", "HeadPose"]
