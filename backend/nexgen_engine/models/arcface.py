from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..utils import l2_normalize

logger = logging.getLogger(__name__)

ARCFACE_INPUT_SIZE = 112
ARCFACE_EMBEDDING_DIM = 512

# Model packs, with the recognition network each one actually ships.
# buffalo_l is the verified default: smaller, faster, and comfortably inside the
# 4 GB budget on a Quadro M1200. antelopev2 carries the R100 network, which is
# more accurate and roughly twice the cost.
MODEL_PACKS = {
    "buffalo_l": {"recognition": "w600k_r50", "detection": "det_10g (SCRFD-10GF)", "dim": 512},
    "buffalo_s": {"recognition": "w600k_mbf", "detection": "det_500m", "dim": 512},
    "antelopev2": {"recognition": "glintr100 (R100)", "detection": "scrfd_10g_bnkps", "dim": 512},
}


class EngineUnavailableError(RuntimeError):
    """Raised when the recognition model cannot be loaded.

    This is deliberately fatal. There is no fallback embedding, because any
    substitute would produce numbers that look like similarity scores and mean
    nothing -- which is far more dangerous in an investigation than an outage.
    """


@dataclass(frozen=True)
class RecognizerInfo:
    """What is actually running, so callers never have to guess."""

    backend: str
    model_pack: str
    recognition_network: str
    embedding_dim: int
    device: str
    providers: tuple[str, ...] = ()

    @property
    def recognition_capable(self) -> bool:
        """Always True once constructed.

        Kept as an explicit field because the API contract exposes it and
        clients branch on it. There is no longer any code path that builds a
        recognizer which cannot recognize.
        """
        return True

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model_pack": self.model_pack,
            "recognition_network": self.recognition_network,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "providers": list(self.providers),
            "recognition_capable": True,
        }


class FaceRecognizer(ABC):
    """Turns aligned 112x112 face crops into comparable templates."""

    info: RecognizerInfo

    @abstractmethod
    def embed_batch(self, crops: list[Image.Image]) -> np.ndarray:
        """Return an (N, D) array of L2-normalized templates."""

    def embed(self, crop: Image.Image) -> np.ndarray:
        return self.embed_batch([crop])[0]


class ArcFaceRecognizer(FaceRecognizer):
    """ArcFace inference through InsightFace's ONNX recognition model.

    Templates are 512-d and L2-normalized, so cosine similarity is a dot product
    in [-1, 1]. Measured on 40 AgeDB identities (CPU, buffalo_l): genuine pairs
    mean 0.49, impostor pairs mean 0.02. Those are properties of this model on
    that dataset, not guarantees -- calibrate on your own imagery with
    ``scripts/calibrate_threshold.py``.
    """

    def __init__(self, recognition_model, model_pack: str, device: str, providers: tuple[str, ...]) -> None:  # noqa: ANN001
        self._model = recognition_model
        pack = MODEL_PACKS.get(model_pack, {})
        self.info = RecognizerInfo(
            backend="insightface_arcface",
            model_pack=model_pack,
            recognition_network=str(pack.get("recognition", "unknown")),
            embedding_dim=int(pack.get("dim", ARCFACE_EMBEDDING_DIM)),
            device=device,
            providers=providers,
        )

    def embed_batch(self, crops: list[Image.Image]) -> np.ndarray:
        """Embed several crops in one call.

        InsightFace batches internally, so passing a list is materially cheaper
        than looping: session setup and memory transfer are paid once rather
        than per image.
        """
        if not crops:
            raise ValueError("At least one aligned crop is required.")
        batch = [_to_bgr_112(crop) for crop in crops]
        features = np.asarray(self._model.get_feat(batch), dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return l2_normalize(features, axis=1)


def _to_bgr_112(crop: Image.Image) -> np.ndarray:
    """InsightFace recognition expects 112x112 BGR uint8."""
    image = crop.convert("RGB")
    if image.size != (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE):
        image = image.resize((ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)[:, :, ::-1]


__all__ = [
    "ARCFACE_EMBEDDING_DIM",
    "ARCFACE_INPUT_SIZE",
    "MODEL_PACKS",
    "ArcFaceRecognizer",
    "EngineUnavailableError",
    "FaceRecognizer",
    "RecognizerInfo",
]
