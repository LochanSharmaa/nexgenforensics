from __future__ import annotations

from ..config import BackboneConfig
from .backbones import DeterministicBackbone


class ViTH14Backbone(DeterministicBackbone):
    def __init__(self) -> None:
        super().__init__(
            BackboneConfig(
                name="vit_h14",
                provider="timm",
                model_id="vit_huge_patch14_224",
                embedding_dim=2048,
                image_size=224,
                weight=0.25,
            )
        )
