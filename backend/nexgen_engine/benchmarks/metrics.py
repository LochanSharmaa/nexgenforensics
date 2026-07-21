from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VerificationMetrics:
    threshold: float
    accuracy: float
    far: float
    frr: float
    tar_at_far: float


def evaluate_verification_scores(scores: np.ndarray, same_identity: np.ndarray, target_far: float = 1e-4) -> VerificationMetrics:
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(same_identity, dtype=bool)
    thresholds = np.linspace(-1.0, 1.0, 2001)
    best_accuracy = -1.0
    best_threshold = 0.0
    best_far = 1.0
    best_frr = 1.0
    tar_at_far = 0.0
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = int(np.logical_and(predicted, labels).sum())
        tn = int(np.logical_and(~predicted, ~labels).sum())
        fp = int(np.logical_and(predicted, ~labels).sum())
        fn = int(np.logical_and(~predicted, labels).sum())
        accuracy = (tp + tn) / max(len(labels), 1)
        far = fp / max(int((~labels).sum()), 1)
        frr = fn / max(int(labels.sum()), 1)
        tar = tp / max(int(labels.sum()), 1)
        if far <= target_far:
            tar_at_far = max(tar_at_far, tar)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
            best_far = far
            best_frr = frr
    return VerificationMetrics(best_threshold, float(best_accuracy), float(best_far), float(best_frr), float(tar_at_far))
