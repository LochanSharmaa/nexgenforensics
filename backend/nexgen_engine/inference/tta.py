from __future__ import annotations

from PIL import Image

from ..data.augmentation import TestTimeAugmenter


class TTAProcessor(TestTimeAugmenter):
    def apply(self, image: Image.Image) -> list[Image.Image]:
        return self.generate(image)
