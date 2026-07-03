from __future__ import annotations

from PIL import Image, ImageStat

from ..utils import clamp


class MorphingDetector:
    def __init__(self, alert_threshold: float = 0.30) -> None:
        self.alert_threshold = alert_threshold

    def risk_score(self, image: Image.Image) -> float:
        stat = ImageStat.Stat(image.convert("RGB").resize((96, 96)))
        channel_spread = max(stat.stddev) - min(stat.stddev)
        return round(clamp(channel_spread / 42.0), 4)

    def flagged(self, image: Image.Image) -> bool:
        return self.risk_score(image) >= self.alert_threshold
