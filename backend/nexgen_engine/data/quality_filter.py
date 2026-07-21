from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, ImageStat

from ..config import QualityConfig
from ..utils import clamp


@dataclass(frozen=True)
class QualityReport:
    score: float
    resolution_score: float
    brightness_score: float
    contrast_score: float
    sharpness_score: float
    accepted: bool
    reasons: tuple[str, ...]


class ImageQualityFilter:
    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()

    def evaluate_bytes(self, image_bytes: bytes) -> QualityReport:
        with Image.open(BytesIO(image_bytes)) as image:
            return self.evaluate_image(image)

    def evaluate_image(self, image: Image.Image) -> QualityReport:
        gray = image.convert("L")
        width, height = gray.size
        stat = ImageStat.Stat(gray)
        brightness = stat.mean[0]
        contrast = stat.stddev[0]
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_mean = ImageStat.Stat(edges).mean[0]

        resolution_score = clamp((min(width, height) - self.config.min_resolution) / 432)
        brightness_score = 1.0 - clamp(abs(brightness - 128.0) / 128.0)
        contrast_score = clamp(contrast / 72.0)
        sharpness_score = clamp(edge_mean / 28.0)
        score = (
            resolution_score * 0.30
            + brightness_score * 0.22
            + contrast_score * 0.22
            + sharpness_score * 0.26
        )

        reasons: list[str] = []
        if min(width, height) < self.config.min_resolution:
            reasons.append("resolution_below_minimum")
        if not (self.config.min_brightness <= brightness <= self.config.max_brightness):
            reasons.append("brightness_out_of_range")
        if contrast_score < 0.35:
            reasons.append("low_contrast")
        if sharpness_score < 0.35:
            reasons.append("blur_risk")

        return QualityReport(
            score=round(float(score), 4),
            resolution_score=round(float(resolution_score), 4),
            brightness_score=round(float(brightness_score), 4),
            contrast_score=round(float(contrast_score), 4),
            sharpness_score=round(float(sharpness_score), 4),
            accepted=score >= self.config.min_faceqnet_score and not reasons,
            reasons=tuple(reasons),
        )

    def keep_top(self, reports: list[QualityReport]) -> list[int]:
        if not reports:
            return []
        keep_count = max(1, int(np.ceil(len(reports) * self.config.keep_top_fraction)))
        ranked = sorted(enumerate(reports), key=lambda item: item[1].score, reverse=True)
        return [index for index, _ in ranked[:keep_count]]
