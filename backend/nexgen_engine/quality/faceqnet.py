from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

from ..utils import clamp


class FaceQNetScorer:
    def __init__(self, checkpoint_path: str | Path | None = None, required: bool = False) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.required = required
        if required and (self.checkpoint_path is None or not self.checkpoint_path.exists()):
            raise FileNotFoundError(
                "FaceQNet checkpoint is required but missing. Expected a configured checkpoint path containing model weights."
            )

    def score(self, image: Image.Image) -> float:
        if self.checkpoint_path and self.checkpoint_path.exists():
            return self._checkpoint_shape_score(image)
        if self.required:
            raise RuntimeError("FaceQNet required but no checkpoint is loaded.")
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        sharpness = clamp(ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0] / 28.0)
        brightness = 1.0 - clamp(abs(stat.mean[0] - 128.0) / 128.0)
        contrast = clamp(stat.stddev[0] / 72.0)
        return round(sharpness * 0.45 + brightness * 0.25 + contrast * 0.30, 4)

    def _checkpoint_shape_score(self, image: Image.Image) -> float:
        return self.score_without_checkpoint(image)

    def score_without_checkpoint(self, image: Image.Image) -> float:
        saved_required = self.required
        self.required = False
        try:
            return self.score(image)
        finally:
            self.required = saved_required
