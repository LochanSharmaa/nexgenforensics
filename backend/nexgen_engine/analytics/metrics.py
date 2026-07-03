from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UsageMetrics:
    enrollments: int = 0
    identifications: int = 0
    review_required: int = 0
    average_latency_ms: float = 0.0

    @property
    def review_rate(self) -> float:
        total = self.enrollments + self.identifications
        return self.review_required / total if total else 0.0
