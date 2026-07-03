from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import l2_normalize


@dataclass(frozen=True)
class NegativePair:
    anchor_index: int
    negative_index: int
    similarity: float


class HardNegativeMiner:
    def __init__(self, top_fraction: float = 0.20, min_similarity: float = 0.30, max_similarity: float = 0.70) -> None:
        self.top_fraction = top_fraction
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity

    def mine(self, embeddings: np.ndarray, labels: np.ndarray) -> list[NegativePair]:
        features = l2_normalize(embeddings, axis=1)
        similarity = features @ features.T
        pairs: list[NegativePair] = []
        for anchor in range(features.shape[0]):
            for negative in range(features.shape[0]):
                if anchor == negative or labels[anchor] == labels[negative]:
                    continue
                score = float(similarity[anchor, negative])
                if self.min_similarity <= score <= self.max_similarity:
                    pairs.append(NegativePair(anchor, negative, score))
        pairs.sort(key=lambda item: item.similarity, reverse=True)
        return pairs[: max(1, int(len(pairs) * self.top_fraction))] if pairs else []
