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
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        from .torch_backbone import OptionalTorchBackbone

        self.backbones = [OptionalTorchBackbone(backbone_config) for backbone_config in self.config.backbones]

    def encode_all(self, image: Image.Image, quality_score: float = 1.0) -> list[BackboneOutput]:
        return [backbone.encode(image, quality_score) for backbone in self.backbones]

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
