from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from ..config import BackboneConfig, EngineConfig
from ..utils import deterministic_vector, l2_normalize


@dataclass(frozen=True)
class BackboneOutput:
    name: str
    embedding: np.ndarray
    quality_weight: float


class DeterministicBackbone:
    def __init__(self, config: BackboneConfig) -> None:
        self.config = config

    def encode(self, image: Image.Image, quality_score: float = 1.0) -> BackboneOutput:
        buffer = BytesIO()
        image.convert("RGB").resize((self.config.image_size, self.config.image_size)).save(buffer, format="PNG")
        seed = hashlib.sha256(self.config.name.encode("utf-8") + buffer.getvalue()).digest()
        embedding = deterministic_vector(seed, self.config.embedding_dim)
        return BackboneOutput(
            name=self.config.name,
            embedding=embedding,
            quality_weight=max(0.01, self.config.weight * max(quality_score, 0.05)),
        )


class BackboneEnsemble:
    """
    Ensemble of buffalo_l (w600k_r50) + antelopev2 (glintr100).

    Fusion: EMBEDDING-SPACE AVERAGING
    -----------------------------------
    Each model produces an independent L2-normalized 512-d ArcFace embedding.
    We compute: fused = L2_normalize( (emb_buffalo + emb_antelope) / 2 )

    This is the natural ensemble operation in cosine-similarity space:
    the averaged embedding points in the direction that is geometrically
    closest to both model outputs simultaneously. It keeps the embedding
    dimension at 512-d so the rest of the pipeline (index, service, search)
    requires zero changes.

    Why not score-level averaging?
    Score-level requires both the gallery and probe to be scored by both models
    at search time — but our VectorSearchIndex stores only one embedding per
    identity and computes similarity in a single pass. Embedding averaging
    is simpler and equally principled for models with the same output space.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        from .insightface_backbone import InsightFaceEnsembleBackbone
        self.ensemble = InsightFaceEnsembleBackbone()

    def encode_all(self, image: Image.Image, quality_score: float = 1.0) -> list[BackboneOutput]:
        return [self.ensemble.encode(image, quality_score)]

    def encode_tta(self, images: list[Image.Image], quality_score: float = 1.0) -> list[BackboneOutput]:
        grouped: dict[str, list[np.ndarray]] = {}
        weights: dict[str, float] = {}
        for image in images:
            for output in self.encode_all(image, quality_score):
                grouped.setdefault(output.name, []).append(output.embedding)
                weights[output.name] = output.quality_weight
        return [
            BackboneOutput(name=name, embedding=l2_normalize(np.mean(vectors, axis=0)), quality_weight=weights[name])
            for name, vectors in grouped.items()
        ]

