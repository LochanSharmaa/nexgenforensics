from __future__ import annotations

import numpy as np

from .metrics import VerificationMetrics, evaluate_verification_scores


def run_aging_eval(scores: np.ndarray, labels: np.ndarray) -> VerificationMetrics:
    return evaluate_verification_scores(scores, labels, target_far=1e-4)
