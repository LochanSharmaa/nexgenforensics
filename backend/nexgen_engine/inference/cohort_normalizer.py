from __future__ import annotations

import numpy as np

from ..utils import l2_normalize


class CohortNormalizer:
    def __init__(self, cohort_size: int = 200) -> None:
        self.cohort_size = cohort_size
        self._cohort: list[np.ndarray] = []

    def update(self, embedding: np.ndarray) -> None:
        self._cohort.append(l2_normalize(embedding.astype(np.float32)))
        if len(self._cohort) > self.cohort_size:
            self._cohort = self._cohort[-self.cohort_size :]

    def normalize(self, embedding: np.ndarray) -> np.ndarray:
        vector = l2_normalize(embedding.astype(np.float32))
        if len(self._cohort) < 2:
            return vector
        cohort = np.stack(self._cohort, axis=0)
        scores = cohort @ vector
        mean = float(scores.mean())
        std = float(scores.std()) or 1.0
        adjusted = vector * (1.0 + np.clip((scores.max() - mean) / std, -0.2, 0.2) * 0.05)
        return l2_normalize(adjusted)
