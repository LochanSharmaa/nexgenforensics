from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


class SyntheticFaceAugmentor:
    def generate(self, image: Image.Image) -> list[Image.Image]:
        base = image.convert("RGB")
        return [
            ImageOps.mirror(base),
            ImageEnhance.Contrast(base).enhance(1.18),
            ImageEnhance.Contrast(base).enhance(0.88),
            ImageEnhance.Color(base).enhance(0.82),
            ImageEnhance.Brightness(base).enhance(1.15),
            ImageEnhance.Brightness(base).enhance(0.85),
            base.filter(ImageFilter.GaussianBlur(radius=0.6)),
            base.rotate(3, resample=Image.Resampling.BICUBIC, expand=False),
            base.rotate(-3, resample=Image.Resampling.BICUBIC, expand=False),
        ]
