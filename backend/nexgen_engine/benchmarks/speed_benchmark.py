from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable


def _percentile(sorted_samples: list[float], q: float) -> float:
    """Nearest-rank percentile. q in [0, 1].

    Nearest-rank rather than interpolated: with the small sample counts used
    for latency runs, an interpolated p99 invents a value that was never
    observed. Every number reported here is a measurement that actually
    happened.
    """
    if not sorted_samples:
        return float("nan")
    rank = max(1, math.ceil(q * len(sorted_samples)))
    return sorted_samples[min(rank, len(sorted_samples)) - 1]


def benchmark_latency_ms(
    operation: Callable[[], object],
    iterations: int = 20,
    warmup: int = 0,
) -> dict[str, float]:
    """Time ``operation`` and return latency percentiles in milliseconds.

    ``warmup`` iterations run first and are discarded. On CUDA the first call
    pays one-off kernel autotuning and allocator growth, which otherwise lands
    entirely in the max and skews p95/p99 for short runs.
    """
    for _ in range(max(0, warmup)):
        operation()

    samples: list[float] = []
    for _ in range(max(1, iterations)):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000)

    ordered = sorted(samples)
    mean = statistics.fmean(samples)
    return {
        "iterations": len(samples),
        "mean_ms": mean,
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": ordered[0],
        "p50_ms": _percentile(ordered, 0.50),
        "p95_ms": _percentile(ordered, 0.95),
        "p99_ms": _percentile(ordered, 0.99),
        "max_ms": ordered[-1],
        "throughput_per_s": (1000.0 / mean) if mean > 0 else float("inf"),
    }
