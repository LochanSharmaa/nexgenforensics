from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from .pipeline import FacialRecognitionPipeline


class EmbeddingExtractor:
    def __init__(self, pipeline: FacialRecognitionPipeline | None = None) -> None:
        self.pipeline = pipeline or FacialRecognitionPipeline()

    def from_bytes(self, image_bytes: bytes) -> np.ndarray:
        return self.pipeline.encode_bytes(image_bytes).embedding

    def from_image(self, image: Image.Image) -> np.ndarray:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return self.from_bytes(buffer.getvalue())
