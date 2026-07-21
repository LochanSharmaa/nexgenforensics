from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metric_losses import AdaFaceLoss, CosFaceLoss, ElasticFaceLoss, TripletLoss, UniformityLoss


@dataclass
class CombinedMetricLoss:
    adaface: AdaFaceLoss = field(default_factory=AdaFaceLoss)
    elasticface: ElasticFaceLoss = field(default_factory=ElasticFaceLoss)
    cosface: CosFaceLoss = field(default_factory=CosFaceLoss)
    triplet: TripletLoss = field(default_factory=TripletLoss)
    uniformity: UniformityLoss = field(default_factory=UniformityLoss)

    def stage_weights(self, stage: int) -> dict[str, float]:
        if stage <= 1:
            return {"cosface": 1.0}
        if stage == 2:
            return {"adaface": 0.58, "elasticface": 0.42}
        if stage >= 4:
            return {"triplet": 0.30, "adaface": 0.70}
        return {"adaface": 0.35, "elasticface": 0.25, "cosface": 0.20, "triplet": 0.15, "uniformity": 0.05}

    def __call__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_weights: np.ndarray,
        stage: int = 3,
        quality: np.ndarray | None = None,
    ) -> dict[str, float]:
        weights = self.stage_weights(stage)
        components: dict[str, float] = {}
        if "adaface" in weights:
            components["adaface"] = self.adaface(embeddings, labels, class_weights, quality)
        if "elasticface" in weights:
            components["elasticface"] = self.elasticface(embeddings, labels, class_weights)
        if "cosface" in weights:
            components["cosface"] = self.cosface(embeddings, labels, class_weights)
        if "triplet" in weights:
            components["triplet"] = self.triplet(embeddings, labels)
        if "uniformity" in weights:
            components["uniformity"] = self.uniformity(embeddings)
        total = sum(components[name] * weights[name] for name in components)
        return {"total": float(total), **components}
