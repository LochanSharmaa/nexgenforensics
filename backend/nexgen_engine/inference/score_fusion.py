from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EngineConfig, ThresholdConfig

# Decision labels returned to the client. They are deliberately investigative
# rather than conclusive: the system proposes candidates, a human adjudicates.
DECISION_MATCH = "candidate_match"
DECISION_REVIEW = "review_required"
DECISION_NO_MATCH = "no_match"
DECISION_REJECTED = "probe_rejected"
DECISION_UNAVAILABLE = "recognition_unavailable"


@dataclass(frozen=True)
class Decision:
    label: str
    confidence: float
    explanation: str
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.label,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "reasons": list(self.reasons),
        }


class ScoreNormalizer:
    """Cohort (Z-norm) normalization applied at the score level.

    Some probes are "lucky": they score moderately high against everyone because
    of lighting or pose. Comparing a candidate score against the distribution of
    that same probe's scores across the rest of the gallery shows whether the top
    hit actually stands out.

    This operates on scores, never on templates. Normalizing the template itself
    would make a stored identity depend on unrelated search traffic, so the same
    person would enrol differently depending on when they were enrolled.
    """

    @staticmethod
    def z_scores(scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        if values.size < 2:
            return np.zeros_like(values)
        std = float(values.std())
        if std < 1e-9:
            return np.zeros_like(values)
        return (values - float(values.mean())) / std

    @staticmethod
    def margin(scores: np.ndarray) -> float:
        """Gap between the best and second-best candidate.

        A large margin means the top hit is distinctive. A near-zero margin on a
        big gallery is a classic false-match signature and should pull the
        decision toward human review even when the raw score is high.
        """
        values = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
        if values.size < 2:
            return 0.0
        return float(values[0] - values[1])


class DecisionEngine:
    """Turns a similarity score plus probe context into an adjudicable decision."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    @property
    def thresholds(self) -> ThresholdConfig:
        return self.config.thresholds

    def decide(
        self,
        *,
        top_score: float,
        recognition_capable: bool,
        probe_accepted: bool,
        probe_reasons: tuple[str, ...] = (),
        gallery_size: int = 0,
        margin: float = 0.0,
        candidate_count: int = 0,
    ) -> Decision:
        if not recognition_capable:
            return Decision(
                label=DECISION_UNAVAILABLE,
                confidence=0.0,
                explanation=(
                    "No recognition model is loaded, so no identity conclusion can be drawn. "
                    "Install backend/requirements-engine.txt to enable matching."
                ),
                reasons=("recognition_model_unavailable",),
            )

        if not probe_accepted:
            return Decision(
                label=DECISION_REJECTED,
                confidence=0.0,
                explanation="The probe image did not meet the quality gate, so it was not searched.",
                reasons=probe_reasons,
            )

        if gallery_size == 0:
            return Decision(
                label=DECISION_NO_MATCH,
                confidence=0.0,
                explanation="The gallery is empty; there is nothing to compare against.",
                reasons=("empty_gallery",),
            )

        reasons = list(probe_reasons)

        if top_score < self.thresholds.review:
            return Decision(
                label=DECISION_NO_MATCH,
                confidence=round(float(top_score), 6),
                explanation=(
                    f"Best similarity {top_score:.3f} is below the review threshold "
                    f"{self.thresholds.review:.2f}; no enrolled subject resembles the probe."
                ),
                reasons=tuple(reasons),
            )

        if top_score < self.thresholds.match:
            reasons.append("score_in_review_band")
            return Decision(
                label=DECISION_REVIEW,
                confidence=round(float(top_score), 6),
                explanation=(
                    f"Best similarity {top_score:.3f} sits between the review threshold "
                    f"{self.thresholds.review:.2f} and the match threshold "
                    f"{self.thresholds.match:.2f}. An examiner must adjudicate."
                ),
                reasons=tuple(dict.fromkeys(reasons)),
            )

        # A margin only exists when there is something to compare against. With
        # a single candidate the reported margin is 0.0, which must NOT be read
        # as a dead heat -- it means nothing else scored above the reporting
        # floor, which is the strongest possible outcome, not the weakest.
        has_runner_up = candidate_count >= 2

        # Above threshold but indistinct from the runner-up: on a large gallery
        # this is the shape of a false match, so force review rather than
        # presenting it as a clean hit.
        if has_runner_up and gallery_size >= 50 and margin < 0.05:
            reasons.append("low_margin_over_runner_up")
            return Decision(
                label=DECISION_REVIEW,
                confidence=round(float(top_score), 6),
                explanation=(
                    f"Similarity {top_score:.3f} clears the match threshold, but the runner-up is "
                    f"only {margin:.3f} behind across {gallery_size} enrolled subjects. "
                    "A near-tie at this gallery size warrants examiner review."
                ),
                reasons=tuple(dict.fromkeys(reasons)),
            )

        if reasons:
            return Decision(
                label=DECISION_REVIEW,
                confidence=round(float(top_score), 6),
                explanation=(
                    f"Similarity {top_score:.3f} clears the match threshold, but the probe raised "
                    "quality or integrity flags that require examiner review."
                ),
                reasons=tuple(dict.fromkeys(reasons)),
            )

        separation = (
            f"with a {margin:.3f} margin over the next candidate"
            if has_runner_up
            else "and no other enrolled subject scored above the reporting floor"
        )
        return Decision(
            label=DECISION_MATCH,
            confidence=round(float(top_score), 6),
            explanation=(
                f"Similarity {top_score:.3f} exceeds the match threshold "
                f"{self.thresholds.match:.2f} {separation}. This is an investigative lead, "
                "not a positive identification."
            ),
        )


class ScoreFusion:
    """Combines similarity with probe-integrity signals for candidate ranking.

    The fused value is a *ranking aid* only. Adjudication thresholds are always
    applied to raw cosine similarity, because folding quality into the number
    compared against a calibrated threshold would silently move the operating
    point away from where it was calibrated.
    """

    def fuse(self, similarity: float, quality: float, liveness: float, deepfake_risk: float = 0.0) -> float:
        penalty = (1.0 - quality) * 0.10 + (1.0 - liveness) * 0.06 + deepfake_risk * 0.10
        return round(float(np.clip(similarity - penalty, -1.0, 1.0)), 6)


__all__ = [
    "DECISION_MATCH",
    "DECISION_NO_MATCH",
    "DECISION_REJECTED",
    "DECISION_REVIEW",
    "DECISION_UNAVAILABLE",
    "Decision",
    "DecisionEngine",
    "ScoreFusion",
    "ScoreNormalizer",
]
