from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from ..config import EngineConfig
from ..data.augmentation import TestTimeAugmenter
from ..data.quality_filter import ImageQualityFilter, QualityReport
from ..models import BackboneEnsemble, EnsembleFusion
from ..security.liveness import LivenessDetector
from ..utils import l2_normalize
from .cohort_normalizer import CohortNormalizer


@dataclass(frozen=True)
class RecognitionResult:
    embedding: np.ndarray
    quality: QualityReport
    liveness_score: float
    review_required: bool
    reasons: tuple[str, ...]


class FacialRecognitionPipeline:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.quality_filter = ImageQualityFilter(self.config.quality)
        self.augmenter = TestTimeAugmenter()
        self.backbones = BackboneEnsemble(self.config)
        self.fusion = EnsembleFusion(self.config)
        self.cohort = CohortNormalizer(self.config.cohort_size)
        self.liveness = LivenessDetector(self.config.security.liveness_threshold)

    def encode_bytes(self, image_bytes: bytes) -> RecognitionResult:
        with Image.open(BytesIO(image_bytes)) as image:
            return self.encode_image(image)

    def encode_image(self, image: Image.Image) -> RecognitionResult:
        quality = self.quality_filter.evaluate_image(image)
        tta_images = self.augmenter.generate(image)
        outputs = self.backbones.encode_tta(tta_images, quality.score)
        embedding = self.fusion.fuse(outputs)
        embedding = self.cohort.normalize(embedding)
        self.cohort.update(embedding)
        liveness_score = self.liveness.score_image(image, quality.score)
        reasons = list(quality.reasons)
        if liveness_score < self.config.security.liveness_threshold:
            reasons.append("liveness_below_threshold")
        return RecognitionResult(
            embedding=l2_normalize(embedding),
            quality=quality,
            liveness_score=liveness_score,
            review_required=bool(reasons),
            reasons=tuple(reasons),
        )
