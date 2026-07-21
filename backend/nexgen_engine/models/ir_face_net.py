from __future__ import annotations

from ..config import BackboneConfig
from .backbones import DeterministicBackbone


class IRFaceNetBackbone(DeterministicBackbone):
    def __init__(self) -> None:
        super().__init__(
            BackboneConfig(
                name="ir_face_net",
                provider="custom",
                model_id="resnet50",
                embedding_dim=2048,
                image_size=112,
                weight=0.03,
                specialty="ir",
            )
        )
