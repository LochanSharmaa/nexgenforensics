from __future__ import annotations

import statistics
import time
from collections.abc import Callable


def benchmark_latency_ms(operation: Callable[[], object], iterations: int = 20) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(max(1, iterations)):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[int(len(samples) * 0.95) - 1],
        "max_ms": max(samples),
    }
