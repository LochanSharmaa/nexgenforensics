from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    stage: int
    start_epoch: int
    end_epoch: int
    quality_fraction: float
    learning_rate: float
    loss_profile: str


class CurriculumScheduler:
    def __init__(self) -> None:
        self.stages = (
            CurriculumStage(1, 1, 10, 0.40, 1e-4, "cosface"),
            CurriculumStage(2, 11, 30, 0.70, 1e-3, "adaface_elasticface"),
            CurriculumStage(3, 31, 50, 1.00, 1e-4, "combined"),
            CurriculumStage(4, 51, 60, 1.00, 1e-5, "triplet_adaface"),
        )

    def for_epoch(self, epoch: int) -> CurriculumStage:
        for stage in self.stages:
            if stage.start_epoch <= epoch <= stage.end_epoch:
                return stage
        return self.stages[-1]
