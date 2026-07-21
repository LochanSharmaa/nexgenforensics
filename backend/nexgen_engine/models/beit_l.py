from __future__ import annotations

from ..config import BackboneConfig
from .backbones import DeterministicBackbone


class BEiTLBackbone(DeterministicBackbone):
    def __init__(self) -> None:
        super().__init__(
            BackboneConfig(
                name="beit_l",
                provider="timm",
                model_id="beit_large_patch16_224",
                embedding_dim=2048,
                image_size=224,
                weight=0.17,
            )
        )
