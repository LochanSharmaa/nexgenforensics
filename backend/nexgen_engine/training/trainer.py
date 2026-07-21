from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..losses import CombinedMetricLoss
from ..utils import l2_normalize
from .curriculum import CurriculumScheduler
from .hard_negative_miner import HardNegativeMiner
from .scheduler import WarmRestartScheduler


@dataclass(frozen=True)
class TrainingBatch:
    embeddings: np.ndarray
    labels: np.ndarray
    quality: np.ndarray | None = None


@dataclass(frozen=True)
class TrainingReport:
    epoch: int
    stage: int
    learning_rate: float
    loss: dict[str, float]
    hard_negative_count: int


class NumpyMetricTrainer:
    def __init__(self, class_count: int, embedding_dim: int = 512) -> None:
        self.curriculum = CurriculumScheduler()
        self.lr_scheduler = WarmRestartScheduler()
        self.loss = CombinedMetricLoss()
        self.miner = HardNegativeMiner()
        rng = np.random.default_rng(7)
        self.class_weights = l2_normalize(rng.normal(size=(class_count, embedding_dim)).astype(np.float32), axis=1)

    def evaluate_batch(self, batch: TrainingBatch, epoch: int, global_step: int) -> TrainingReport:
        stage = self.curriculum.for_epoch(epoch)
        hard_negatives = self.miner.mine(batch.embeddings, batch.labels)
        loss = self.loss(batch.embeddings, batch.labels, self.class_weights, stage.stage, batch.quality)
        return TrainingReport(
            epoch=epoch,
            stage=stage.stage,
            learning_rate=self.lr_scheduler.learning_rate(global_step),
            loss=loss,
            hard_negative_count=len(hard_negatives),
        )
