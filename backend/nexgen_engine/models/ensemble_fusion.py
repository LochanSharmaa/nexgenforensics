from __future__ import annotations

import hashlib

import numpy as np

from ..config import EngineConfig
from ..utils import l2_normalize
from .backbones import BackboneOutput


class EnsembleFusion:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.projection = self._projection_matrix(2048, self.config.final_embedding_dim)

    def fuse(self, outputs: list[BackboneOutput]) -> np.ndarray:
        if not outputs:
            raise ValueError("At least one backbone output is required.")
        weights = np.asarray([output.quality_weight for output in outputs], dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-12)
        projected = []
        for output in outputs:
            embedding = output.embedding
            if embedding.shape[0] != self.projection.shape[0]:
                embedding = self._resize_embedding(embedding, self.projection.shape[0])
            projected.append(l2_normalize(embedding @ self.projection))
        return l2_normalize((np.stack(projected, axis=0) * weights[:, None]).sum(axis=0))

    def _projection_matrix(self, in_dim: int, out_dim: int) -> np.ndarray:
        digest = hashlib.sha256(f"nexgen-fusion:{in_dim}:{out_dim}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        matrix = rng.normal(0, 1 / np.sqrt(in_dim), size=(in_dim, out_dim)).astype(np.float32)
        return matrix

    def _resize_embedding(self, embedding: np.ndarray, dimensions: int) -> np.ndarray:
        if embedding.shape[0] > dimensions:
            return embedding[:dimensions]
        return np.pad(embedding, (0, dimensions - embedding.shape[0]))
