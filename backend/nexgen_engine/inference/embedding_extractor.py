from __future__ import annotations

import numpy as np
from PIL import Image

from .pipeline import FacialRecognitionPipeline, RecognitionResult


class EmbeddingExtractor:
    """Thin convenience wrapper for callers that only want the template."""

    def __init__(self, pipeline: FacialRecognitionPipeline | None = None) -> None:
        self.pipeline = pipeline or FacialRecognitionPipeline()

    def from_bytes(self, image_bytes: bytes) -> np.ndarray:
        return self.pipeline.encode_bytes(image_bytes).embedding

    def from_image(self, image: Image.Image) -> np.ndarray:
        # Encode the decoded image directly. Re-serializing to PNG just to parse
        # it back, as an earlier version did, cost a full encode/decode per call.
        return self.pipeline.encode_image(image).embedding

    def describe(self, image_bytes: bytes) -> RecognitionResult:
        return self.pipeline.encode_bytes(image_bytes)


__all__ = ["EmbeddingExtractor"]
