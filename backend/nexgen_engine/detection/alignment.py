from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class FaceBox:
    left: int
    top: int
    right: int
    bottom: int
    confidence: float


class FaceDetector:
    def detect(self, image: Image.Image) -> list[FaceBox]:
        width, height = image.size
        side = int(min(width, height) * 0.72)
        left = (width - side) // 2
        top = (height - side) // 2
        return [FaceBox(left, top, left + side, top + side, 0.99)]


class FaceAligner:
    def __init__(self, output_size: int = 112) -> None:
        self.output_size = output_size
        self.detector = FaceDetector()

    def align(self, image: Image.Image) -> Image.Image:
        boxes = self.detector.detect(image)
        if not boxes:
            raise ValueError("No face candidate detected.")
        box = max(boxes, key=lambda item: item.confidence)
        crop = image.convert("RGB").crop((box.left, box.top, box.right, box.bottom))
        return crop.resize((self.output_size, self.output_size), Image.Resampling.BICUBIC)
