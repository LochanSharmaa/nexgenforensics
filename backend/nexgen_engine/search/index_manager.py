from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .faiss_index import VectorSearchIndex


class IndexManager:
    def __init__(self, dimensions: int = 512) -> None:
        self.index = VectorSearchIndex(dimensions)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dimensions": self.index.dimensions,
            "ids": self.index._ids,
            "metadata": self.index._metadata,
            "vectors": self.index._vectors.tolist(),
        }
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    def load(self, path: str | Path) -> VectorSearchIndex:
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        self.index = VectorSearchIndex(int(payload["dimensions"]))
        self.index._ids = list(payload["ids"])
        self.index._metadata = list(payload["metadata"])
        self.index._vectors = np.asarray(payload["vectors"], dtype=np.float32)
        return self.index
