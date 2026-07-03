from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AccuracyTracker:
    scores: list[float] = field(default_factory=list)
    correct: list[bool] = field(default_factory=list)

    def add(self, score: float, is_correct: bool) -> None:
        self.scores.append(float(score))
        self.correct.append(bool(is_correct))

    def summary(self) -> dict[str, float]:
        if not self.scores:
            return {"samples": 0.0, "accuracy": 0.0, "mean_score": 0.0}
        return {
            "samples": float(len(self.scores)),
            "accuracy": sum(self.correct) / len(self.correct),
            "mean_score": sum(self.scores) / len(self.scores),
        }
