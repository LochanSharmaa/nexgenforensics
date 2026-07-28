from __future__ import annotations

import random

from PIL import Image, ImageEnhance, ImageFilter

# Note for anyone reaching for this at inference time: don't. Test-time
# augmentation for ArcFace is the horizontal flip and nothing else -- see
# nexgen_engine/inference/tta.py. Photometric jitter belongs in training, where
# it teaches invariance, not in inference, where it drags the template off the
# manifold the encoder learned.


class TrainingAugmenter:
    """Photometric and geometric jitter for fine-tuning a recognition backbone.

    Simulates the conditions forensic probes actually arrive in: CCTV
    compression, low light, motion blur, and off-angle capture. Seeded so a
    training run is reproducible.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def augment(self, image: Image.Image) -> Image.Image:
        result = image.convert("RGB")
        if self.random.random() < 0.50:
            result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.random.random() < 0.40:
            result = ImageEnhance.Brightness(result).enhance(self.random.uniform(0.60, 1.40))
        if self.random.random() < 0.40:
            result = ImageEnhance.Contrast(result).enhance(self.random.uniform(0.65, 1.35))
        if self.random.random() < 0.30:
            result = result.filter(ImageFilter.GaussianBlur(self.random.uniform(0.3, 1.6)))
        if self.random.random() < 0.25:
            result = self._downscale_cycle(result)
        if self.random.random() < 0.20:
            result = result.rotate(self.random.uniform(-12.0, 12.0), resample=Image.Resampling.BICUBIC)
        return result

    def batch(self, image: Image.Image, count: int) -> list[Image.Image]:
        return [self.augment(image) for _ in range(max(0, count))]

    def _downscale_cycle(self, image: Image.Image) -> Image.Image:
        """Round-trip through a smaller resolution to mimic CCTV capture."""
        width, height = image.size
        factor = self.random.uniform(0.30, 0.70)
        small = image.resize(
            (max(8, int(width * factor)), max(8, int(height * factor))),
            Image.Resampling.BILINEAR,
        )
        return small.resize((width, height), Image.Resampling.BICUBIC)


__all__ = ["TrainingAugmenter"]
