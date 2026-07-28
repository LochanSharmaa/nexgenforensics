from __future__ import annotations

import numpy as np

from ..utils import l2_normalize


class CohortNormalizer:
    """Score-level cohort normalization against a reference (impostor) set.

    Holds a fixed cohort of templates that are known NOT to be the subject.
    Scoring a probe against that cohort gives the impostor distribution for that
    specific probe, which is what turns a raw similarity into a comparable one.

    An earlier version of this class applied its adjustment to the *embedding*
    and mutated its cohort with every query. That made a stored template depend
    on unrelated search history, so the same photograph enrolled twice produced
    two different identities. Normalization now returns a score and the cohort
    is only changed through the explicit ``set_cohort`` call.
    """

    def __init__(self, cohort_size: int = 200) -> None:
        self.cohort_size = cohort_size
        self._cohort = np.empty((0, 0), dtype=np.float32)

    @property
    def size(self) -> int:
        return int(self._cohort.shape[0])

    def set_cohort(self, embeddings: np.ndarray) -> None:
        """Replace the cohort with the most recent ``cohort_size`` templates."""
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            self._cohort = np.empty((0, 0), dtype=np.float32)
            return
        self._cohort = l2_normalize(matrix[-self.cohort_size :], axis=1)

    def normalize_score(self, probe: np.ndarray, raw_score: float) -> float:
        """Z-normalize ``raw_score`` against the probe's impostor distribution.

        Returns the raw score unchanged when the cohort is too small to give a
        meaningful mean and standard deviation.
        """
        if self.size < 10:
            return float(raw_score)
        vector = l2_normalize(np.asarray(probe, dtype=np.float32))
        if vector.shape[0] != self._cohort.shape[1]:
            return float(raw_score)
        impostor_scores = self._cohort @ vector
        std = float(impostor_scores.std())
        if std < 1e-9:
            return float(raw_score)
        return float((raw_score - float(impostor_scores.mean())) / std)


__all__ = ["CohortNormalizer"]
