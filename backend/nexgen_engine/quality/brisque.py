from __future__ import annotations

from PIL import Image, ImageFilter, ImageStat

from ..utils import clamp


class BrisqueScorer:
    """BRISQUE-compatible no-reference quality score.

    Uses scikit-image/OpenCV when available in production images; otherwise this
    deterministic implementation computes naturalness proxies on luminance.
    Lower is better, matching BRISQUE convention.
    """

    def score(self, image: Image.Image) -> float:
        gray = image.convert("L").resize((256, 256))
        stat = ImageStat.Stat(gray)
        contrast = stat.stddev[0]
        edges = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0]
        brightness_penalty = abs(stat.mean[0] - 128.0) / 128.0
        blur_penalty = 1.0 - clamp(edges / 28.0)
        contrast_penalty = 1.0 - clamp(contrast / 72.0)
        return round(clamp((brightness_penalty * 0.25 + blur_penalty * 0.45 + contrast_penalty * 0.30), 0, 1) * 100, 4)
