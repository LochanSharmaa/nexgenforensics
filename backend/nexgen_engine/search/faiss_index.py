from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import l2_normalize


@dataclass(frozen=True)
class MatchResult:
    identity_id: str
    score: float
    metadata: dict[str, Any]


class VectorSearchIndex:
    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions
        self._ids: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._vectors = np.empty((0, dimensions), dtype=np.float32)

    def add(self, identity_id: str, embedding: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        vector = l2_normalize(np.asarray(embedding, dtype=np.float32))
        if vector.shape[0] != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {vector.shape[0]}.")
        self._ids.append(identity_id)
        self._metadata.append(metadata or {})
        self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])

    def search(self, embedding: np.ndarray, top_k: int = 20) -> list[MatchResult]:
        if self._vectors.shape[0] == 0:
            return []
        query = l2_normalize(np.asarray(embedding, dtype=np.float32))
        scores = self._vectors @ query
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            MatchResult(self._ids[index], round(float(scores[index]), 6), self._metadata[index])
            for index in top_indices
        ]

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dimensions": self.dimensions,
            "ids": self._ids,
            "metadata": self._metadata,
            "vectors": self._vectors.tolist(),
        }
        target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "VectorSearchIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(int(payload["dimensions"]))
        index._ids = [str(item) for item in payload.get("ids", [])]
        index._metadata = [dict(item) for item in payload.get("metadata", [])]
        index._vectors = np.asarray(payload.get("vectors", []), dtype=np.float32).reshape((-1, index.dimensions))
        return index

    def snapshot(self) -> dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "count": len(self._ids),
            "ids": list(self._ids),
        }
