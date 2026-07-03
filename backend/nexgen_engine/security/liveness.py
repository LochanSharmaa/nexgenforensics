from __future__ import annotations

import hashlib
from io import BytesIO

from PIL import Image

from ..utils import clamp


class LivenessDetector:
    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold

    def score_image(self, image: Image.Image, quality_score: float) -> float:
        buffer = BytesIO()
        image.convert("RGB").resize((64, 64)).save(buffer, format="PNG")
        digest = hashlib.sha256(buffer.getvalue()).digest()
        texture_signal = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        score = quality_score * 0.72 + texture_signal * 0.18 + 0.10
        return round(clamp(score), 4)

    def passed(self, image: Image.Image, quality_score: float) -> bool:
        return self.score_image(image, quality_score) >= self.threshold
