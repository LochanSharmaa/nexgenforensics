from __future__ import annotations

from ..config import BackboneConfig
from .backbones import DeterministicBackbone


class ResNet100IRBackbone(DeterministicBackbone):
    def __init__(self) -> None:
        super().__init__(
            BackboneConfig(
                name="resnet100",
                provider="insightface",
                model_id="iresnet100",
                embedding_dim=2048,
                image_size=112,
                weight=0.12,
            )
        )
