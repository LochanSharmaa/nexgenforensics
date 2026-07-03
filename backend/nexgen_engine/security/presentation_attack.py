from __future__ import annotations

from PIL import Image

from .deepfake_detector import DeepfakeDetector
from .liveness import LivenessDetector
from .morphing_detector import MorphingDetector


class PresentationAttackDetector:
    def __init__(self) -> None:
        self.deepfake = DeepfakeDetector()
        self.morphing = MorphingDetector()
        self.liveness = LivenessDetector()

    def assess(self, image: Image.Image, quality_score: float) -> dict[str, float | bool]:
        liveness_score = self.liveness.score_image(image, quality_score)
        deepfake_risk = self.deepfake.risk_score(image)
        morphing_risk = self.morphing.risk_score(image)
        flagged = liveness_score < self.liveness.threshold or self.deepfake.flagged(image) or self.morphing.flagged(image)
        return {
            "liveness_score": liveness_score,
            "deepfake_risk": deepfake_risk,
            "morphing_risk": morphing_risk,
            "flagged": flagged,
        }
