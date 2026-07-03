from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


class TestTimeAugmenter:
    """Six deterministic augmentations used for inference-time embedding stability."""

    def generate(self, image: Image.Image) -> list[Image.Image]:
        aligned = image.convert("RGB").resize((112, 112))
        return [
            aligned,
            ImageOps.mirror(aligned),
            ImageEnhance.Brightness(aligned).enhance(1.10),
            ImageEnhance.Brightness(aligned).enhance(0.90),
            aligned.filter(ImageFilter.SHARPEN),
            image.convert("RGB").resize((128, 128)).resize((112, 112)),
        ]
