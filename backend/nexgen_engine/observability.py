"""
Per-stage latency observability.

WHY IN-PROCESS AND BOUNDED
--------------------------
The pipeline already measures every stage of every call (StageTimings in
inference/pipeline.py). Until now those numbers were computed, attached to the
result, and thrown away. This module keeps a bounded window of them so
operational questions can be answered from the running service rather than from
a benchmark run on an idle machine:

    "is detection slower today than last week?"
    "did the p99 move after that model change?"
    "is the GPU actually being used right now?"

Deliberately a fixed-size ring buffer, not a growing list: an always-on
collector that accumulates without limit is a memory leak with a long fuse.
At the default 1024 samples and ~6 floats each this is well under a megabyte
and cannot grow.

Deliberately NOT a Prometheus client or a metrics backend. Adding one is a
deployment decision with its own dependency and scrape surface. This exposes
the same numbers through the existing authenticated API, which needs no new
infrastructure. If a real metrics stack arrives later, read from here.

PERCENTILES ARE NEAREST-RANK
----------------------------
Same convention as benchmarks/speed_benchmark.py, for the same reason: with a
small window an interpolated p99 reports a latency that never actually
occurred. Every figure here is an observation that happened.

THREAD SAFETY
-------------
FastAPI serves requests on a thread pool, so record() is called concurrently.
All mutation is under a lock. The lock is held only for an append, never
across the percentile computation.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any

# Stages reported by the pipeline. Kept explicit rather than discovered so a
# renamed stage shows up as missing instead of silently disappearing.
_STAGES = ("decode_ms", "detect_ms", "align_ms", "embed_ms", "quality_ms", "total_ms")


def _percentile(ordered: list[float], q: float) -> float:
    """Nearest-rank percentile. `ordered` must already be sorted."""
    if not ordered:
        return float("nan")
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


class LatencyCollector:
    """Bounded, thread-safe window of per-stage pipeline timings."""

    def __init__(self, window: int = 1024) -> None:
        self._window = window
        self._samples: deque[dict[str, float]] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._started = time.time()
        self._total = 0
        self._errors = 0

    def record(self, timings: dict[str, float]) -> None:
        """Record one call's stage timings. Never raises.

        Observability must not be able to break the request it is observing,
        so a malformed timings dict is dropped rather than propagated.
        """
        try:
            sample = {k: float(timings.get(k, 0.0)) for k in _STAGES}
        except (TypeError, ValueError):
            return
        with self._lock:
            self._samples.append(sample)
            self._total += 1

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1
            self._total += 1

    def snapshot(self) -> dict[str, Any]:
        """Current percentiles. Safe to call at any time."""
        with self._lock:
            samples = list(self._samples)
            total, errors, started = self._total, self._errors, self._started

        uptime = max(time.time() - started, 1e-9)
        out: dict[str, Any] = {
            "window_size": self._window,
            "samples_in_window": len(samples),
            "calls_total": total,
            "errors_total": errors,
            "error_rate": (errors / total) if total else 0.0,
            "uptime_seconds": round(uptime, 1),
            "calls_per_second": round(total / uptime, 4),
            "percentile_method": "nearest-rank (every value is an observed sample)",
            "stages": {},
        }
        if not samples:
            out["note"] = "No calls recorded yet; percentiles are unavailable."
            return out

        for stage in _STAGES:
            ordered = sorted(s[stage] for s in samples)
            mean = sum(ordered) / len(ordered)
            out["stages"][stage.removesuffix("_ms")] = {
                "mean_ms": round(mean, 3),
                "p50_ms": round(_percentile(ordered, 0.50), 3),
                "p95_ms": round(_percentile(ordered, 0.95), 3),
                "p99_ms": round(_percentile(ordered, 0.99), 3),
                "min_ms": round(ordered[0], 3),
                "max_ms": round(ordered[-1], 3),
            }

        total_mean = out["stages"]["total"]["mean_ms"]
        if total_mean > 0:
            # Share of wall-clock per stage: the fastest way to see WHERE time
            # goes without reading six separate numbers.
            out["stage_share_pct"] = {
                name: round(vals["mean_ms"] / total_mean * 100, 1)
                for name, vals in out["stages"].items()
                if name != "total"
            }
            out["throughput_per_second_single_threaded"] = round(1000.0 / total_mean, 2)
        return out

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._total = 0
            self._errors = 0
            self._started = time.time()


#: Process-wide collector. One recognition pipeline per process, so one
#: collector; a per-request instance would have nothing to aggregate over.
LATENCY = LatencyCollector()


__all__ = ["LATENCY", "LatencyCollector"]
