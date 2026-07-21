from __future__ import annotations

import numpy as np

from ..utils import clamp


class ScoreFusion:
    def fuse(self, similarity: float, quality: float, liveness: float, deepfake_risk: float = 0.0) -> float:
        score = similarity * 0.62 + quality * 0.18 + liveness * 0.16 - deepfake_risk * 0.12
        return round(clamp(float(score), -1.0, 1.0), 6)

    def z_norm(self, score: float, cohort_scores: np.ndarray) -> float:
        if cohort_scores.size == 0:
            return float(score)
        return float((score - float(cohort_scores.mean())) / (float(cohort_scores.std()) + 1e-6))
