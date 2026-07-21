from __future__ import annotations

from ..config import BackboneConfig
from .backbones import DeterministicBackbone


class SwinLBackbone(DeterministicBackbone):
    def __init__(self) -> None:
        super().__init__(
            BackboneConfig(
                name="swin_l",
                provider="timm",
                model_id="swin_large_patch4_window12_384",
                embedding_dim=2048,
                image_size=384,
                weight=0.18,
            )
        )
