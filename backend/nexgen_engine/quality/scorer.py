from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .brisque import BrisqueScorer
from .faceqnet import FaceQNetScorer


@dataclass(frozen=True)
class QualityScorerConfig:
    enable_brisque: bool = True
    enable_faceqnet: bool = True
    faceqnet_checkpoint: str | None = None
    require_faceqnet_checkpoint: bool = False
    min_faceqnet: float = 0.60
    max_brisque: float = 60.0


@dataclass(frozen=True)
class RichQualityReport:
    brisque_score: float | None
    faceqnet_score: float | None
    detector_confidence: float
    landmark_confidence: float
    accepted: bool
    reasons: tuple[str, ...]


class ProductionQualityScorer:
    def __init__(self, config: QualityScorerConfig | None = None) -> None:
        self.config = config or QualityScorerConfig()
        self.brisque = BrisqueScorer() if self.config.enable_brisque else None
        self.faceqnet = (
            FaceQNetScorer(self.config.faceqnet_checkpoint, self.config.require_faceqnet_checkpoint)
            if self.config.enable_faceqnet
            else None
        )

    def evaluate(self, image: Image.Image, detector_confidence: float = 1.0, landmark_confidence: float = 1.0) -> RichQualityReport:
        brisque = self.brisque.score(image) if self.brisque else None
        faceqnet = self.faceqnet.score(image) if self.faceqnet else None
        reasons: list[str] = []
        if brisque is not None and brisque > self.config.max_brisque:
            reasons.append("brisque_above_threshold")
        if faceqnet is not None and faceqnet < self.config.min_faceqnet:
            reasons.append("faceqnet_below_threshold")
        if detector_confidence < 0.95:
            reasons.append("detector_confidence_below_threshold")
        if landmark_confidence < 0.80:
            reasons.append("landmark_confidence_below_threshold")
        return RichQualityReport(brisque, faceqnet, detector_confidence, landmark_confidence, not reasons, tuple(reasons))
