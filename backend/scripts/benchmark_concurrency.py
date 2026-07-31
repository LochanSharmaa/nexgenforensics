#!/usr/bin/env python
"""
Item 29 — concurrency and request batching. Closes SCORECARD limitation L7.

    python backend/scripts/benchmark_concurrency.py

Every latency figure in BENCHMARKS.md §7b is single-threaded. That describes one
operator on an idle machine, not a service under load, so "65 images/second"
has until now been an unwarranted extrapolation. This measures what actually
happens when requests overlap.

TWO DISTINCT QUESTIONS, MEASURED SEPARATELY
-------------------------------------------
1. THREAD CONCURRENCY. FastAPI serves on a thread pool, so N requests hit the
   ONNX session at once. onnxruntime's InferenceSession.run is thread-safe, but
   thread-safe is not the same as parallel: internal locking and a shared intra-
   op thread pool mean throughput may not scale with workers, and p99 latency
   can degrade badly while mean throughput looks fine.

2. REQUEST BATCHING. Feeding N images to the model as ONE batched call instead
   of N separate calls. This is where a real gain is expected, because the
   per-call overhead (blob construction, session dispatch, memory transfer) is
   paid once rather than N times.

The distinction matters for what to build: if batching wins and threading does
not, the fix is a request-collecting queue in front of the model, not more
uvicorn workers.

HONEST SCOPE
------------
This measures the ENGINE under concurrent load, not the full HTTP stack. Real
end-to-end throughput is additionally bounded by request parsing, base64
decoding and database writes. Treat these as an upper bound on what the
recognition path can sustain, not a service-level SLO.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "scripts"))

_ROOT = _BACKEND.parent


def load_images(n: int) -> list[bytes]:
    d = _ROOT / "src_extracted/AgeDB/AgeDB"
    files = sorted(d.glob("*.jpg"))[:n]
    if not files:
        raise SystemExit(f"no images under {d}")
    return [f.read_bytes() for f in files]


def percentile(ordered: list[float], q: float) -> float:
    import math

    if not ordered:
        return float("nan")
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def run_threaded(pipeline, payloads: list[bytes], workers: int, total: int) -> dict:
    """total encodes spread across `workers` threads."""
    latencies: list[float] = []
    lock = threading.Lock()
    counter = {"i": 0}

    def worker():
        local = []
        while True:
            with lock:
                i = counter["i"]
                if i >= total:
                    break
                counter["i"] = i + 1
            t = time.perf_counter()
            pipeline.encode_bytes(payloads[i % len(payloads)])
            local.append((time.perf_counter() - t) * 1000)
        with lock:
            latencies.extend(local)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in range(workers):
            ex.submit(worker)
    wall = time.perf_counter() - start

    ordered = sorted(latencies)
    return {
        "workers": workers,
        "encodes": len(latencies),
        "wall_s": round(wall, 3),
        "throughput_per_s": round(len(latencies) / wall, 2),
        "mean_ms": round(statistics.fmean(ordered), 2),
        "p50_ms": round(percentile(ordered, 0.50), 2),
        "p95_ms": round(percentile(ordered, 0.95), 2),
        "p99_ms": round(percentile(ordered, 0.99), 2),
    }


def run_batched(recognizer, images, batch_size: int, total: int) -> dict:
    """Same total work, submitted as batches of `batch_size` to the model."""
    n = len(images)
    start = time.perf_counter()
    done = 0
    lat: list[float] = []
    while done < total:
        take = min(batch_size, total - done)
        chunk = [images[(done + k) % n] for k in range(take)]
        t = time.perf_counter()
        recognizer.get_feat(chunk)
        dt = (time.perf_counter() - t) * 1000
        lat.append(dt)
        done += take
    wall = time.perf_counter() - start
    ordered = sorted(lat)
    return {
        "batch_size": batch_size,
        "encodes": done,
        "wall_s": round(wall, 3),
        "throughput_per_s": round(done / wall, 2),
        "ms_per_image": round(wall * 1000 / done, 3),
        "batch_call_p50_ms": round(percentile(ordered, 0.50), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=120, help="encodes per configuration")
    ap.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16, 32, 64])
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/concurrency.json"))
    args = ap.parse_args()

    from nexgen_engine.config import EngineConfig
    from nexgen_engine.inference.pipeline import FacialRecognitionPipeline

    print("=" * 78)
    print("  Item 29 - concurrency and request batching")
    print("=" * 78)

    pipeline = FacialRecognitionPipeline(EngineConfig())
    payloads = load_images(32)
    pipeline.encode_bytes(payloads[0])  # warm up before timing anything
    print(f"  device={pipeline.runtime.device}  images={len(payloads)}  "
          f"encodes per config={args.total}")

    # ---- 1. thread concurrency, full pipeline ----
    print("\n1. Thread concurrency (full pipeline: decode -> detect -> align -> embed)")
    print(f"   {'workers':>8} {'thr/s':>9} {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9}  scaling")
    threaded = []
    base = None
    for w in args.workers:
        r = run_threaded(pipeline, payloads, w, args.total)
        threaded.append(r)
        if base is None:
            base = r["throughput_per_s"]
        scale = r["throughput_per_s"] / base if base else 0
        print(f"   {w:>8} {r['throughput_per_s']:>9.2f} {r['p50_ms']:>9.2f} "
              f"{r['p95_ms']:>9.2f} {r['p99_ms']:>9.2f}  {scale:.2f}x")

    # ---- 2. request batching, recognition model only ----
    # Isolated to the recogniser: detection is per-image and cannot be batched
    # without also batching the detector, which is a separate change.
    print("\n2. Request batching (recognition model only, 112x112 aligned crops)")
    import cv2

    rec = pipeline.runtime.recognizer
    model = getattr(rec, "model", None) or getattr(rec, "_model", None)
    if model is None or not hasattr(model, "get_feat"):
        print("   SKIPPED: could not reach the underlying recogniser's get_feat()")
        batched = []
    else:
        crops = []
        for p in payloads:
            arr = cv2.imdecode(np.frombuffer(p, np.uint8), cv2.IMREAD_COLOR)
            crops.append(cv2.resize(arr, (112, 112)))
        model.get_feat(crops[:2])  # warm up
        print(f"   {'batch':>8} {'thr/s':>9} {'ms/image':>10}  speedup vs batch=1")
        batched = []
        first = None
        for b in args.batches:
            r = run_batched(model, crops, b, args.total)
            batched.append(r)
            if first is None:
                first = r["throughput_per_s"]
            print(f"   {b:>8} {r['throughput_per_s']:>9.2f} {r['ms_per_image']:>10.3f}"
                  f"  {r['throughput_per_s'] / first:.2f}x")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device": pipeline.runtime.device,
        "encodes_per_config": args.total,
        "scope": "engine under load, NOT the full HTTP stack; an upper bound",
        "thread_concurrency": threaded,
        "request_batching": batched,
    }, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
