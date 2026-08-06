from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .deepfake_detector import DeepfakeDetector
from .liveness import LivenessDetector
from .morphing_detector import MorphingDetector


@dataclass(frozen=True)
class IntegrityAssessment:
    liveness_score: float
    deepfake_risk: float
    morphing_risk: float
    flagged: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "liveness_score": self.liveness_score,
            "deepfake_risk": self.deepfake_risk,
            "morphing_risk": self.morphing_risk,
            "flagged": self.flagged,
            "reasons": list(self.reasons),
            "certified": False,
        }


class PresentationAttackDetector:
    """Runs the three passive integrity screens over one aligned crop.

    None of these are certified detectors -- see the individual modules. The
    combined result answers "does anything about this image warrant a closer
    look", not "is this image authentic".
    """

    def __init__(
        self,
        liveness: LivenessDetector | None = None,
        deepfake: DeepfakeDetector | None = None,
        morphing: MorphingDetector | None = None,
    ) -> None:
        self.liveness = liveness or LivenessDetector()
        self.deepfake = deepfake or DeepfakeDetector()
        self.morphing = morphing or MorphingDetector()

    def assess(self, image: Image.Image) -> IntegrityAssessment:
        liveness_report = self.liveness.analyze(image)
        deepfake_report = self.deepfake.analyze(image)
        morphing_risk = self.morphing.risk_score(image)

        reasons: list[str] = list(liveness_report.reasons)
        if not liveness_report.passed:
            reasons.append("liveness_below_threshold")
        reasons.extend(deepfake_report.reasons)
        if morphing_risk >= self.morphing.alert_threshold:
            reasons.append("morphing_risk")

        return IntegrityAssessment(
            liveness_score=liveness_report.score,
            deepfake_risk=deepfake_report.score,
            morphing_risk=morphing_risk,
            flagged=bool(reasons),
            reasons=tuple(dict.fromkeys(reasons)),
        )


__all__ = ["IntegrityAssessment", "PresentationAttackDetector"]
