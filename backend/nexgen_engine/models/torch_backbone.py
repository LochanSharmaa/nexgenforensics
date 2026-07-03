from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import BackboneConfig
from ..utils import l2_normalize
from .backbones import BackboneOutput, DeterministicBackbone


class OptionalTorchBackbone:
    def __init__(self, config: BackboneConfig, checkpoint_path: str | Path | None = None) -> None:
        self.config = config
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._fallback = DeterministicBackbone(config)
        self._model = None
        self._torch = None
        self._transform = None
        self._load_optional_model()

    @property
    def production_loaded(self) -> bool:
        return self._model is not None and self._torch is not None

    def encode(self, image: Image.Image, quality_score: float = 1.0) -> BackboneOutput:
        if not self.production_loaded:
            return self._fallback.encode(image, quality_score)
        assert self._torch is not None
        with self._torch.no_grad():
            tensor = self._image_to_tensor(image)
            output = self._model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            embedding = output.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if embedding.shape[0] != self.config.embedding_dim:
            embedding = self._resize(embedding, self.config.embedding_dim)
        return BackboneOutput(
            name=self.config.name,
            embedding=l2_normalize(embedding),
            quality_weight=max(0.01, self.config.weight * max(quality_score, 0.05)),
        )

    def _load_optional_model(self) -> None:
        try:
            import timm
            import torch

            model = timm.create_model(self.config.model_id, pretrained=False, num_classes=0)
            if self.checkpoint_path and self.checkpoint_path.exists():
                state = torch.load(self.checkpoint_path, map_location="cpu")
                model.load_state_dict(state.get("state_dict", state), strict=False)
            model.eval()
            self._model = model
            self._torch = torch
        except Exception:
            self._model = None
            self._torch = None

    def _image_to_tensor(self, image: Image.Image):
        assert self._torch is not None
        array = np.asarray(image.convert("RGB").resize((self.config.image_size, self.config.image_size)), dtype=np.float32)
        array = (array / 255.0 - 0.5) / 0.5
        tensor = self._torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor

    def _resize(self, embedding: np.ndarray, dimensions: int) -> np.ndarray:
        if embedding.shape[0] > dimensions:
            return embedding[:dimensions]
        return np.pad(embedding, (0, dimensions - embedding.shape[0]))
