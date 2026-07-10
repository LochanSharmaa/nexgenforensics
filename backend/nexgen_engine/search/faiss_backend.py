from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..utils import l2_normalize
from .faiss_index import MatchResult, VectorSearchIndex


class OptionalFaissIndex:
    def __init__(self, dimensions: int = 512, index_path: str | Path | None = None) -> None:
        self.dimensions = dimensions
        self.index_path = Path(index_path) if index_path else None
        self.metadata_path = self.index_path.with_suffix(self.index_path.suffix + ".meta.json") if self.index_path else None
        self._ids: list[str] = []
        self._metadata: list[dict] = []
        self._fallback = VectorSearchIndex(dimensions)
        self._faiss = None
        self._index = None
        self._load_faiss()
        self.load()

    @property
    def production_loaded(self) -> bool:
        return self._faiss is not None and self._index is not None

    def add(self, identity_id: str, embedding: np.ndarray, metadata: dict | None = None) -> None:
        vector = l2_normalize(np.asarray(embedding, dtype=np.float32)).reshape(1, -1)
        if not self.production_loaded:
            self._fallback.add(identity_id, vector.reshape(-1), metadata)
            self.save()
            return
        self._index.add(vector)
        self._ids.append(identity_id)
        self._metadata.append(metadata or {})
        self.save()

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

    def save(self) -> None:
        if self.index_path is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.production_loaded:
            self._fallback.save(self.index_path)
            return
        assert self._faiss is not None and self._index is not None and self.metadata_path is not None
        self._faiss.write_index(self._index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps({"dimensions": self.dimensions, "ids": self._ids, "metadata": self._metadata}, sort_keys=True),
            encoding="utf-8",
        )

    def load(self) -> None:
        if self.index_path is None or not self.index_path.exists():
            return
        try:
            if not self.production_loaded:
                self._fallback = VectorSearchIndex.load(self.index_path)
                return
            assert self._faiss is not None
            self._index = self._faiss.read_index(str(self.index_path))
            if self.metadata_path is not None and self.metadata_path.exists():
                payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                self._ids = [str(item) for item in payload.get("ids", [])]
                self._metadata = [dict(item) for item in payload.get("metadata", [])]
        except Exception:
            self._fallback = VectorSearchIndex(self.dimensions)
            if self.production_loaded:
                self._index = self._faiss.IndexFlatIP(self.dimensions)

    def snapshot(self) -> dict[str, object]:
        if not self.production_loaded:
            data = self._fallback.snapshot()
            data["backend"] = "numpy_fallback"
            data["path"] = str(self.index_path) if self.index_path else ""
            return data
        return {
            "backend": "faiss",
            "dimensions": self.dimensions,
            "count": len(self._ids),
            "path": str(self.index_path) if self.index_path else "",
        }
