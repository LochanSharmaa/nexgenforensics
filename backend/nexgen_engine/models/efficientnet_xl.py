from __future__ import annotations

from ..config import BackboneConfig
from .backbones import DeterministicBackbone


class EfficientNetXLBackbone(DeterministicBackbone):
    def __init__(self) -> None:
        super().__init__(
            BackboneConfig(
                name="efficientnet_xl",
                provider="timm",
                model_id="tf_efficientnetv2_xl",
                embedding_dim=2048,
                image_size=384,
                weight=0.10,
            )
        )
