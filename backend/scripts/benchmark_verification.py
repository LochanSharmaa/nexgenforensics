#!/usr/bin/env python
"""
1:1 VERIFICATION benchmark across the standard InsightFace protocol packs.

    python backend/scripts/benchmark_verification.py --datasets lfw agedb_30 cfp_fp calfw cplfw

Embeddings are extracted ONCE per (dataset, backbone) and cached to disk, then
every fusion configuration is scored from the cache. Re-running to add a new
fusion config therefore costs no GPU time.

Reported accuracy always comes from 10-fold cross-validation where the
threshold is fitted on 9 folds and applied to the held-out fold. See
nexgen_engine/benchmarks/verification.py for the protocol rationale.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.benchmarks.verification import (  # noqa: E402
    EXPECTED_PAIRS,
    decode_pack,
    evaluate_pairs,
    l2n,
    load_pack,
)

_ROOT = _BACKEND.parent

# The .bin packs ship inside several training-set bundles. faces_webface is the
# only one carrying all six, so it is preferred; the others are fallbacks.
PACK_DIRS = [
    _ROOT / "src_extracted/faces_webface_112x112/faces_webface_112x112",
    _ROOT / "src_extracted/faces_megafacetrain_112x112/faces_megafacetrain_112x112",
    _ROOT / "src_extracted/faces_umd/faces_umd",
]

RECOGNITION_MODELS = {
    "w600k_r50": ("buffalo_l", "w600k_r50.onnx", "ResNet-50 / WebFace600K"),
    "glintr100": ("antelopev2", "glintr100.onnx", "ResNet-100 / Glint360K"),
    "w600k_mbf": ("buffalo_s", "w600k_mbf.onnx", "MobileFaceNet / WebFace600K"),
}

CACHE_DIR = _ROOT / "runtime" / "benchmarks" / "embeddings"


def find_pack(dataset: str) -> Path:
    for d in PACK_DIRS:
        p = d / f"{dataset}.bin"
        if p.exists():
            return p
    raise FileNotFoundError(f"{dataset}.bin not found in any of {[str(d) for d in PACK_DIRS]}")


def model_path(key: str) -> Path:
    import os

    pack, filename, _ = RECOGNITION_MODELS[key]
    root = Path(os.environ.get("NEXGEN_MODEL_ROOT", Path.home() / ".insightface" / "models"))
    p = root / pack / filename
    if not p.exists():
        raise FileNotFoundError(f"recognition model missing: {p}")
    return p


def load_recognizer(key: str):
    """Load a bare ArcFace recognition ONNX model (no detector).

    The protocol packs are already ArcFace-aligned 112x112 crops, so the
    detector is not just unnecessary -- running it on a pre-cropped face
    degrades accuracy. Reference implementations feed the crop directly.
    """
    from insightface.model_zoo import get_model

    from nexgen_engine.models.cuda_runtime import (
        CUDA_PROVIDER,
        cuda_expected,
        init_cuda,
        resolve_providers,
        session_provider,
    )

    init_cuda()
    providers, ctx_id = resolve_providers()
    model = get_model(str(model_path(key)), providers=providers)
    model.prepare(ctx_id=ctx_id)

    bound = session_provider(model)
    if cuda_expected() and bound != CUDA_PROVIDER:
        raise RuntimeError(
            f"{key} bound {bound} but CUDA was expected. "
            f"Benchmarks on CPU would take hours and hide a broken GPU setup. "
            f"Run: python scripts/verify_gpu.py"
        )
    print(f"    [{key}] provider={bound}", flush=True)
    return model


def extract(dataset: str, model_key: str, batch_size: int, force: bool) -> dict:
    """Extract (and cache) original + horizontally-flipped embeddings."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{dataset}__{model_key}.npz"
    if cache.exists() and not force:
        d = np.load(cache)
        print(f"    [{model_key}] cache hit ({d['orig'].shape[0]} images)", flush=True)
        return {"orig": d["orig"], "flip": d["flip"], "issame": d["issame"]}

    bins, issame = load_pack(find_pack(dataset))
    expected = EXPECTED_PAIRS.get(dataset)
    if expected and len(issame) != expected:
        raise ValueError(
            f"{dataset}: {len(issame)} pairs but published protocol has {expected}. "
            "Refusing to benchmark on a non-standard pair list."
        )
    images = decode_pack(bins)

    model = load_recognizer(model_key)
    n = len(images)
    orig = np.empty((n, 512), dtype=np.float32)
    flip = np.empty((n, 512), dtype=np.float32)

    t0 = time.time()
    for i in range(0, n, batch_size):
        chunk = images[i : i + batch_size]
        orig[i : i + len(chunk)] = model.get_feat([im for im in chunk])
        flip[i : i + len(chunk)] = model.get_feat([im[:, ::-1] for im in chunk])
        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, n)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"      {done}/{n}  {rate:.0f} img/s", end="\r", flush=True)
    dt = time.time() - t0
    print(f"    [{model_key}] {n} images x2(flip) in {dt:.1f}s ({2 * n / dt:.0f} fwd/s)", flush=True)

    # Release the onnxruntime session + its CUDA arena before the next model
    # loads. This card has 6 GB; three ResNet sessions held simultaneously
    # alongside the decoded image array is enough to trigger an OOM.
    del model
    gc.collect()

    np.savez_compressed(cache, orig=orig, flip=flip, issame=issame)
    return {"orig": orig, "flip": flip, "issame": np.asarray(issame, dtype=bool)}


def fused_embeddings(store: dict[str, dict], use_flip: bool) -> dict[str, np.ndarray]:
    """Per-model per-image embedding, optionally with flip-TTA, L2-normalized."""
    out = {}
    for key, d in store.items():
        e = d["orig"] + d["flip"] if use_flip else d["orig"]
        out[key] = l2n(e.astype(np.float64))
    return out


def build_configs(emb: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """All configurations under test, as per-image embedding matrices."""
    r50, r100, mbf = emb["w600k_r50"], emb["glintr100"], emb["w600k_mbf"]
    cfgs = {
        "single:w600k_r50 (R50)": r50,
        "single:glintr100 (R100)": r100,
        "single:w600k_mbf (MBF)": mbf,
        # production setting currently shipped
        "ensemble:weighted 0.45/0.45/0.10": l2n(0.45 * r50 + 0.45 * r100 + 0.10 * mbf),
        "ensemble:equal 1/3": l2n((r50 + r100 + mbf) / 3.0),
        "ensemble:dual r50+r100": l2n(0.5 * r50 + 0.5 * r100),
        "ensemble:concat 1536-d": np.concatenate([r50, r100, mbf], axis=1) / np.sqrt(3.0),
    }
    return cfgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"])
    ap.add_argument("--models", nargs="+", default=list(RECOGNITION_MODELS))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-flip", action="store_true", help="disable horizontal-flip TTA")
    ap.add_argument("--force", action="store_true", help="ignore embedding cache")
    ap.add_argument("--out", default=str(_ROOT / "runtime" / "benchmarks" / "verification_results.json"))
    args = ap.parse_args()

    use_flip = not args.no_flip
    results = []

    for dataset in args.datasets:
        print(f"\n=== {dataset} ===", flush=True)
        store = {m: extract(dataset, m, args.batch_size, args.force) for m in args.models}
        issame = store[args.models[0]]["issame"]
        emb = fused_embeddings(store, use_flip)
        cfgs = build_configs(emb)

        for name, e in cfgs.items():
            a, b = e[0::2], e[1::2]
            r = evaluate_pairs(a, b, issame, dataset, name)
            results.append(r)
            print(
                f"  {name:36s} acc={r.accuracy_mean * 100:6.2f} ± {r.accuracy_std * 100:4.2f}  "
                f"thr={r.threshold_mean:.4f}  TAR@FAR0.1%={r.tar_at_far_1e3 * 100:6.2f}  "
                f"AUC={r.auc:.5f}",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "10-fold CV, threshold fitted on 9 folds, applied to held-out fold",
        "flip_tta": use_flip,
        "results": [
            {
                **r.__dict__,
                "folds": [{"accuracy": f.accuracy, "threshold": f.threshold} for f in r.folds],
            }
            for r in results
        ],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
