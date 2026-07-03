from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class MetricValue:
    count: int = 0
    total: float = 0.0


class MetricsRegistry:
    def __init__(self) -> None:
        self.counters: dict[str, int] = defaultdict(int)
        self.timers: dict[str, MetricValue] = defaultdict(MetricValue)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount

    @contextmanager
    def timer(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            metric = self.timers[name]
            metric.count += 1
            metric.total += elapsed

    def prometheus_text(self) -> str:
        lines: list[str] = []
        for name, value in sorted(self.counters.items()):
            lines.append(f"nexgen_{name}_total {value}")
        for name, value in sorted(self.timers.items()):
            lines.append(f"nexgen_{name}_seconds_count {value.count}")
            lines.append(f"nexgen_{name}_seconds_sum {value.total:.6f}")
        return "\n".join(lines) + "\n"
