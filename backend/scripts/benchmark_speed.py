#!/usr/bin/env python
"""
Latency and throughput measurement for the deployed recognition path.

    python backend/scripts/benchmark_speed.py

Closes the "no performance data exists" gap. Every number here is measured on
this host, through the same `FacialRecognitionPipeline` the API serves from --
not a synthetic microbenchmark of the ONNX session in isolation.

WHAT IS MEASURED
    1. Full encode  -- decode -> detect -> align -> quality/liveness/deepfake
                       -> embed. This is what one uploaded image costs.
    2. Per-stage    -- the pipeline already records StageTimings per call, so
                       the breakdown is real instrumentation, not estimation.
    3. Verify (1:1) -- two encodes plus the comparison, i.e. one API request.
    4. Gallery search -- brute-force cosine at several gallery sizes, to show
                       where the current approach stops scaling.

Percentiles are nearest-rank (see speed_benchmark._percentile): every reported
figure is an observation that actually occurred, never an interpolation.

Latency depends on image resolution, so the source image and its dimensions
are recorded alongside the numbers. Quoting these figures without that context
would be meaningless.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.benchmarks.speed_benchmark import benchmark_latency_ms  # noqa: E402
from nexgen_engine.config import EngineConfig  # noqa: E402
from nexgen_engine.inference import FacialRecognitionPipeline  # noqa: E402
from nexgen_engine.search.gallery_index import GalleryIndex  # noqa: E402

_ROOT = _BACKEND.parent


def host_info(pipeline) -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
        else:
            info["gpu"] = "none (CPU)"
    except Exception:
        info["gpu"] = "unknown"
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
    except Exception:
        pass
    try:
        rec = pipeline.runtime.recognizer.info
        info["model_pack"] = rec.model_pack
        info["recognition_network"] = rec.recognition_network
        info["providers"] = list(rec.providers)
        info["device"] = pipeline.runtime.device
    except Exception as exc:
        info["model_info_error"] = str(exc)
    return info


def pick_images(n: int = 8) -> list[Path]:
    agedb = _ROOT / "src_extracted/AgeDB/AgeDB"
    files = sorted(agedb.glob("*.jpg"))[:n]
    if not files:
        raise SystemExit(f"no test images found under {agedb}")
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--gallery-sizes", nargs="+", type=int, default=[100, 1000, 10000, 100000])
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/speed.json"))
    args = ap.parse_args()

    print("=" * 74)
    print("  NexGen iMATCH - latency / throughput")
    print("=" * 74)

    pipeline = FacialRecognitionPipeline(EngineConfig())
    images = pick_images()
    payloads = [p.read_bytes() for p in images]

    # Force model load before timing anything.
    warm = pipeline.encode_bytes(payloads[0])
    info = host_info(pipeline)
    print("\nHost / model")
    for k, v in info.items():
        print(f"  {k:22s} {v}")

    from PIL import Image
    from io import BytesIO

    with Image.open(BytesIO(payloads[0])) as im:
        dims = f"{im.width}x{im.height}"
    print(f"\nTest images: {len(images)} from AgeDB, source resolution {dims}")
    print("  NOTE: latency scales with input resolution. These are pre-cropped")
    print("        112x112 archival images -- a 4000x3000 phone photo costs")
    print("        substantially more in decode and detect.")

    results = {"host": info, "image_resolution": dims, "n_images": len(images)}

    # ---- 1. full encode -------------------------------------------------
    print("\n1. Full encode (decode -> detect -> align -> quality -> embed)")
    counter = {"i": 0}

    def one_encode():
        payload = payloads[counter["i"] % len(payloads)]
        counter["i"] += 1
        return pipeline.encode_bytes(payload)

    enc = benchmark_latency_ms(one_encode, iterations=args.iterations, warmup=args.warmup)
    results["encode"] = enc
    print(f"   p50={enc['p50_ms']:.2f}ms  p95={enc['p95_ms']:.2f}ms  "
          f"p99={enc['p99_ms']:.2f}ms  max={enc['max_ms']:.2f}ms")
    print(f"   mean={enc['mean_ms']:.2f}ms +/- {enc['stdev_ms']:.2f}  "
          f"-> {enc['throughput_per_s']:.1f} images/s single-threaded")

    # ---- 2. per-stage breakdown ----------------------------------------
    print("\n2. Per-stage breakdown (pipeline's own StageTimings)")
    stage_samples: dict[str, list[float]] = {}
    for i in range(args.iterations):
        r = pipeline.encode_bytes(payloads[i % len(payloads)])
        for k, v in r.timings.as_dict().items():
            stage_samples.setdefault(k, []).append(float(v))
    stages = {}
    total_mean = statistics.fmean(stage_samples.get("total_ms", [1.0]))
    for k, v in sorted(stage_samples.items(), key=lambda kv: -statistics.fmean(kv[1])):
        m = statistics.fmean(v)
        stages[k] = {"mean_ms": m, "p50_ms": statistics.median(v)}
        share = (m / total_mean * 100) if total_mean else 0
        bar = "#" * int(share / 3)
        label = k.replace("_ms", "")
        print(f"   {label:10s} mean={m:7.2f}ms  {share:5.1f}%  {bar}")
    results["stages"] = stages

    # ---- 3. verify (1:1), the real API operation ------------------------
    print("\n3. Verify 1:1 (two encodes + cosine) - one API request")

    def one_verify():
        a = pipeline.encode_bytes(payloads[0])
        b = pipeline.encode_bytes(payloads[1])
        return float(np.dot(a.embedding, b.embedding))

    ver = benchmark_latency_ms(one_verify, iterations=max(10, args.iterations // 2), warmup=2)
    results["verify_1to1"] = ver
    print(f"   p50={ver['p50_ms']:.2f}ms  p95={ver['p95_ms']:.2f}ms  "
          f"p99={ver['p99_ms']:.2f}ms")
    print(f"   -> {ver['throughput_per_s']:.1f} verifications/s single-threaded")

    # ---- 4. gallery search scaling --------------------------------------
    print("\n4. Gallery search (brute-force cosine) vs gallery size")
    dim = int(warm.embedding.shape[0])
    rng = np.random.default_rng(0)
    probe = warm.embedding
    search = {}
    for size in args.gallery_sizes:
        index = GalleryIndex(dim)
        vecs = rng.normal(size=(size, dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        index.add_many(
            "bench",
            [(f"t{i}", f"s{i}", vecs[i], {}) for i in range(size)],
        )
        s = benchmark_latency_ms(
            lambda: index.search("bench", probe, top_k=10), iterations=20, warmup=3
        )
        search[str(size)] = s
        print(f"   {size:>7,} templates:  p50={s['p50_ms']:8.3f}ms  "
              f"p95={s['p95_ms']:8.3f}ms  -> {s['throughput_per_s']:8.1f} searches/s")
        del index, vecs
    results["gallery_search"] = search

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
