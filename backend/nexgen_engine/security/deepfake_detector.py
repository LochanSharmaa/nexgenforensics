from __future__ import annotations

from PIL import Image, ImageFilter, ImageStat

from ..utils import clamp


class DeepfakeDetector:
    def __init__(self, alert_threshold: float = 0.85) -> None:
        self.alert_threshold = alert_threshold

    def risk_score(self, image: Image.Image) -> float:
        gray = image.convert("L").resize((128, 128))
        high_freq = gray.filter(ImageFilter.FIND_EDGES)
        edge_mean = ImageStat.Stat(high_freq).mean[0]
        smoothness = 1.0 - clamp(edge_mean / 32.0)
        return round(clamp(smoothness * 0.72), 4)

    def flagged(self, image: Image.Image) -> bool:
        return self.risk_score(image) >= self.alert_threshold
