from __future__ import annotations

import numpy as np

from .faiss_index import MatchResult, VectorSearchIndex


class ShardedVectorSearch:
    def __init__(self, shard_count: int = 4, dimensions: int = 512) -> None:
        self.shards = [VectorSearchIndex(dimensions) for _ in range(max(1, shard_count))]

    def add(self, identity_id: str, embedding: np.ndarray, metadata: dict | None = None) -> None:
        shard = self.shards[hash(identity_id) % len(self.shards)]
        shard.add(identity_id, embedding, metadata)

    def search(self, embedding: np.ndarray, top_k: int = 20) -> list[MatchResult]:
        merged: list[MatchResult] = []
        for shard in self.shards:
            merged.extend(shard.search(embedding, top_k))
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:top_k]
