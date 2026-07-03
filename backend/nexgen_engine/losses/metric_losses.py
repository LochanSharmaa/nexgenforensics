from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import l2_normalize


def _one_hot(labels: np.ndarray, class_count: int) -> np.ndarray:
    output = np.zeros((labels.shape[0], class_count), dtype=np.float32)
    output[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
    return output


def _softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    losses = -np.log(np.maximum(probs[np.arange(labels.shape[0]), labels.astype(int)], 1e-12))
    return float(losses.mean())


@dataclass(frozen=True)
class CosFaceLoss:
    scale: float = 64.0
    margin: float = 0.4

    def __call__(self, embeddings: np.ndarray, labels: np.ndarray, class_weights: np.ndarray) -> float:
        features = l2_normalize(embeddings, axis=1)
        weights = l2_normalize(class_weights, axis=1)
        cosine = features @ weights.T
        logits = self.scale * (cosine - _one_hot(labels, weights.shape[0]) * self.margin)
        return _softmax_cross_entropy(logits, labels)


@dataclass(frozen=True)
class ElasticFaceLoss:
    scale: float = 64.0
    margin: float = 0.5
    std: float = 0.0175

    def __call__(self, embeddings: np.ndarray, labels: np.ndarray, class_weights: np.ndarray) -> float:
        features = l2_normalize(embeddings, axis=1)
        weights = l2_normalize(class_weights, axis=1)
        cosine = features @ weights.T
        rng = np.random.default_rng(20260703)
        elastic_margin = rng.normal(self.margin, self.std, size=labels.shape[0]).astype(np.float32)
        margin_matrix = _one_hot(labels, weights.shape[0]) * elastic_margin[:, None]
        return _softmax_cross_entropy(self.scale * (cosine - margin_matrix), labels)


@dataclass(frozen=True)
class AdaFaceLoss:
    scale: float = 64.0
    margin: float = 0.4

    def __call__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_weights: np.ndarray,
        quality: np.ndarray | None = None,
    ) -> float:
        features = l2_normalize(embeddings, axis=1)
        weights = l2_normalize(class_weights, axis=1)
        cosine = features @ weights.T
        if quality is None:
            quality = np.linalg.norm(embeddings, axis=1)
        normalized_quality = (quality - quality.mean()) / (quality.std() + 1e-6)
        adaptive_margin = self.margin * np.clip(1.0 + 0.08 * normalized_quality, 0.72, 1.28)
        logits = self.scale * (cosine - _one_hot(labels, weights.shape[0]) * adaptive_margin[:, None])
        return _softmax_cross_entropy(logits, labels)


@dataclass(frozen=True)
class TripletLoss:
    margin: float = 0.3

    def __call__(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        features = l2_normalize(embeddings, axis=1)
        distance = 1.0 - (features @ features.T)
        losses: list[float] = []
        for index, label in enumerate(labels):
            positive_mask = labels == label
            negative_mask = labels != label
            positives = distance[index][positive_mask]
            negatives = distance[index][negative_mask]
            positives = positives[positives > 1e-8]
            if positives.size == 0 or negatives.size == 0:
                continue
            hardest_positive = float(positives.max())
            semi_hard = negatives[(negatives > hardest_positive) & (negatives < hardest_positive + self.margin)]
            negative = float(semi_hard.min() if semi_hard.size else negatives.min())
            losses.append(max(0.0, hardest_positive - negative + self.margin))
        return float(np.mean(losses)) if losses else 0.0


@dataclass(frozen=True)
class UniformityLoss:
    temperature: float = 2.0

    def __call__(self, embeddings: np.ndarray) -> float:
        features = l2_normalize(embeddings, axis=1)
        if features.shape[0] < 2:
            return 0.0
        diffs = features[:, None, :] - features[None, :, :]
        squared_distance = np.sum(diffs * diffs, axis=-1)
        mask = ~np.eye(features.shape[0], dtype=bool)
        return float(np.log(np.exp(-self.temperature * squared_distance[mask]).mean() + 1e-12))
