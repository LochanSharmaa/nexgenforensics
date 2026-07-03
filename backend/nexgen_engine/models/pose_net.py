from __future__ import annotations

from ..config import BackboneConfig
from .backbones import DeterministicBackbone


class PoseNetBackbone(DeterministicBackbone):
    def __init__(self) -> None:
        super().__init__(
            BackboneConfig(
                name="pose_net",
                provider="custom",
                model_id="swin_base_patch4_window7_224",
                embedding_dim=2048,
                image_size=224,
                weight=0.05,
                specialty="pose",
            )
        )
