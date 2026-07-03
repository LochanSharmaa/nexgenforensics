from __future__ import annotations

from dataclasses import dataclass

from .alignment import CenterCropDetector, FaceDetectorProtocol, InsightFaceRetinaFaceDetector, MtcnnDetector


@dataclass(frozen=True)
class DetectorConfig:
    backend: str = "center_crop"
    allow_fallback: bool = True
    min_confidence: float = 0.95


class DetectorRegistry:
    def create(self, config: DetectorConfig) -> FaceDetectorProtocol:
        backend = config.backend.lower()
        if backend in {"retinaface", "insightface"}:
            return InsightFaceRetinaFaceDetector(min_confidence=config.min_confidence)
        if backend == "mtcnn":
            return MtcnnDetector(min_confidence=config.min_confidence)
        if backend == "center_crop":
            if not config.allow_fallback:
                raise RuntimeError("center_crop detector is local/test only and disabled by production config.")
            return CenterCropDetector()
        raise ValueError(f"Unknown detector backend: {config.backend}")
