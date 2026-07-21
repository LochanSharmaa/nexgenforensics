from __future__ import annotations

import numpy as np

from ..utils import l2_normalize
from .faiss_index import MatchResult, VectorSearchIndex


class OptionalFaissIndex:
    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions
        self._ids: list[str] = []
        self._metadata: list[dict] = []
        self._fallback = VectorSearchIndex(dimensions)
        self._faiss = None
        self._index = None
        self._load_faiss()

    @property
    def production_loaded(self) -> bool:
        return self._faiss is not None and self._index is not None

    def add(self, identity_id: str, embedding: np.ndarray, metadata: dict | None = None) -> None:
        vector = l2_normalize(np.asarray(embedding, dtype=np.float32)).reshape(1, -1)
        if not self.production_loaded:
            self._fallback.add(identity_id, vector.reshape(-1), metadata)
            return
        self._index.add(vector)
        self._ids.append(identity_id)
        self._metadata.append(metadata or {})

    def search(self, embedding: np.ndarray, top_k: int = 20) -> list[MatchResult]:
        if not self.production_loaded:
            return self._fallback.search(embedding, top_k)
        query = l2_normalize(np.asarray(embedding, dtype=np.float32)).reshape(1, -1)
        scores, indices = self._index.search(query, top_k)
        results: list[MatchResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self._ids):
                continue
            results.append(MatchResult(self._ids[index], round(float(score), 6), self._metadata[index]))
        return results

    def _load_faiss(self) -> None:
        try:
            import faiss

            self._faiss = faiss
            self._index = faiss.IndexFlatIP(self.dimensions)
        except Exception:
            self._faiss = None
            self._index = None
