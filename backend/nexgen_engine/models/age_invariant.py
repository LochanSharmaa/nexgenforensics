from __future__ import annotations

from ..config import BackboneConfig
from .backbones import DeterministicBackbone


class AgeInvariantBackbone(DeterministicBackbone):
    def __init__(self) -> None:
        super().__init__(
            BackboneConfig(
                name="age_invariant",
                provider="custom",
                model_id="vit_base_patch16_224",
                embedding_dim=2048,
                image_size=224,
                weight=0.10,
                specialty="aging",
            )
        )
