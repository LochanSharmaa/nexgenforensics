#!/usr/bin/env python
"""
Item 28 — approximate nearest-neighbour search: latency AND recall.

    python backend/scripts/benchmark_ann.py

RE-SCOPED FROM "ADD FAISS". READ THIS BEFORE CHANGING THE SEARCH PATH.
----------------------------------------------------------------------
The original plan was "FAISS-backed search". That would not have achieved its
goal. The branch already in gallery_index.py builds `faiss.IndexFlatIP`, an
EXACT inner-product index -- brute force with better SIMD. It buys a constant
factor, not a change in complexity, so 100k templates would still scale
linearly.

Real scaling needs an APPROXIMATE index. Approximation means missed candidates,
and in a forensic system a missed candidate is an investigative lead that
silently never surfaced. So this benchmark reports RECALL alongside latency,
and recall is the number that decides adoption -- not speed.

WHAT IS MEASURED
    exact      numpy matmul, the current production path (ground truth)
    flat       faiss IndexFlatIP, exact, for the constant-factor comparison
    ivfpq      IVF-PQ, approximate, compressed
    hnsw       HNSW, approximate, graph-based

Recall@1  -- fraction of probes whose TOP-1 matches exact search's top-1.
             This is the one that matters: a wrong rank-1 is a wrong lead.
Recall@10 -- fraction of exact top-10 present in the approximate top-10.
             Matters when an examiner reviews a candidate list rather than
             one name.

Vectors are unit-norm 512-d, matching real ArcFace templates, so inner product
is cosine. Synthetic vectors are used deliberately: real galleries at 100k do
not exist here, and recall behaviour is a property of the index geometry, not
of whose faces are in it. That is a limitation, and it is stated in the output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_ROOT = _BACKEND.parent
DIM = 512


def unit(rng, n: int, d: int = DIM) -> np.ndarray:
    v = rng.normal(size=(n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def real_embeddings(model: str = "w600k_r50") -> np.ndarray | None:
    """Pool every cached ArcFace template into one gallery.

    WHY THIS MATTERS MORE THAN THE SYNTHETIC RUN.

    Random unit vectors in 512-d are nearly equidistant (curse of
    dimensionality), so there is no cluster structure for IVF or HNSW to
    exploit and their recall collapses. That is a property of the synthetic
    data, NOT a prediction of production behaviour: real face embeddings
    cluster tightly by identity, which is exactly what these indexes are built
    to exploit.

    Reporting the synthetic recall as the adoption answer would be misleading.
    """
    cache = _ROOT / "runtime" / "benchmarks" / "embeddings"
    parts = []
    for p in sorted(cache.glob(f"*__{model}.npz")):
        d = np.load(p)
        e = (d["orig"] + d["flip"]).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        parts.append(e)
    if not parts:
        return None
    return np.concatenate(parts, axis=0)


def timed(fn, iterations: int, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(iterations):
        t = time.perf_counter()
        fn()
        s.append((time.perf_counter() - t) * 1000)
    s.sort()
    return {
        "p50_ms": round(s[len(s) // 2], 4),
        "p95_ms": round(s[max(0, int(0.95 * len(s)) - 1)], 4),
        "mean_ms": round(sum(s) / len(s), 4),
        "qps": round(1000.0 / (sum(s) / len(s)), 1),
    }


def recall(approx: np.ndarray, exact: np.ndarray, k: int) -> float:
    """Fraction of exact top-k recovered by the approximate index."""
    hits = 0
    for a, e in zip(approx, exact):
        hits += len(set(a[:k].tolist()) & set(e[:k].tolist()))
    return hits / (len(exact) * k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 100000])
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--real", action="store_true",
                    help="use real cached ArcFace templates instead of synthetic vectors")
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/ann_search.json"))
    args = ap.parse_args()

    import faiss

    print("=" * 78)
    print("  Item 28 - ANN search: latency AND recall vs exact")
    print(f"  faiss {faiss.__version__}, {DIM}-d unit vectors, top_k={args.top_k}")
    print("=" * 78)

    rng = np.random.default_rng(0)
    results = {}

    pool = None
    if args.real:
        pool = real_embeddings()
        if pool is None:
            print("no cached embeddings; run benchmark_verification.py first")
            return 1
        print(f"  REAL ArcFace templates available: {pool.shape[0]:,}")
        args.sizes = [n for n in args.sizes if n <= pool.shape[0]]
        if not args.sizes:
            print("  requested sizes exceed the available pool")
            return 1

    for n in args.sizes:
        print(f"\n--- gallery {n:,} {'(REAL templates)' if pool is not None else '(synthetic)'} ---")
        gallery = (
            pool[rng.choice(pool.shape[0], n, replace=False)].copy()
            if pool is not None
            else unit(rng, n)
        )
        # Probes drawn from the SAME distribution as the gallery, which is
        # what a real search does -- a face against a gallery of faces.
        queries = (pool[rng.choice(pool.shape[0], args.queries, replace=False)].copy()
                   if pool is not None else unit(rng, args.queries))
        row = {}

        # ---- ground truth: the current production path ----
        def exact_search():
            return np.argsort(gallery @ queries[0])[::-1][: args.top_k]

        exact_top = np.argsort(queries @ gallery.T, axis=1)[:, ::-1][:, : args.top_k]
        row["exact_numpy"] = {**timed(exact_search, args.iterations), "recall@1": 1.0, "recall@10": 1.0}
        print(f"  exact  (numpy, production)  p50 {row['exact_numpy']['p50_ms']:8.4f}ms  "
              f"{row['exact_numpy']['qps']:>8.1f} qps   recall 1.000 (ground truth)")

        # ---- faiss exact, for the constant-factor comparison ----
        flat = faiss.IndexFlatIP(DIM)
        flat.add(gallery)
        row["faiss_flat_exact"] = {
            **timed(lambda: flat.search(queries[:1], args.top_k), args.iterations),
            "recall@1": 1.0, "recall@10": 1.0,
        }
        print(f"  flat   (faiss, exact)       p50 {row['faiss_flat_exact']['p50_ms']:8.4f}ms  "
              f"{row['faiss_flat_exact']['qps']:>8.1f} qps   recall 1.000")

        # ---- IVF-PQ: approximate + compressed ----
        # nlist ~ sqrt(n) is the usual starting point; m=64 subquantizers over
        # 512 dims = 8 dims each, nbits=8 -> 64 bytes per vector (vs 2048 raw).
        if n >= 4096:
            nlist = max(16, int(np.sqrt(n)))
            ivf = faiss.IndexIVFPQ(faiss.IndexFlatIP(DIM), DIM, nlist, 64, 8)
            ivf.train(gallery)
            ivf.add(gallery)
            for nprobe in (1, 8, 32):
                ivf.nprobe = nprobe
                _, idx = ivf.search(queries, args.top_k)
                row[f"ivfpq_nprobe{nprobe}"] = {
                    **timed(lambda: ivf.search(queries[:1], args.top_k), args.iterations),
                    "recall@1": round(recall(idx, exact_top, 1), 4),
                    "recall@10": round(recall(idx, exact_top, args.top_k), 4),
                    "bytes_per_vector": 64,
                }
                r = row[f"ivfpq_nprobe{nprobe}"]
                print(f"  ivfpq  nprobe={nprobe:<3d}            p50 {r['p50_ms']:8.4f}ms  "
                      f"{r['qps']:>8.1f} qps   recall@1 {r['recall@1']:.3f}  "
                      f"recall@10 {r['recall@10']:.3f}")
        else:
            print("  ivfpq  skipped (needs >=4096 vectors to train meaningfully)")

        # ---- HNSW: approximate, graph-based, uncompressed ----
        hnsw = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = 80
        hnsw.add(gallery)
        for ef in (16, 64, 256):
            hnsw.hnsw.efSearch = ef
            _, idx = hnsw.search(queries, args.top_k)
            row[f"hnsw_ef{ef}"] = {
                **timed(lambda: hnsw.search(queries[:1], args.top_k), args.iterations),
                "recall@1": round(recall(idx, exact_top, 1), 4),
                "recall@10": round(recall(idx, exact_top, args.top_k), 4),
                "bytes_per_vector": DIM * 4,
            }
            r = row[f"hnsw_ef{ef}"]
            print(f"  hnsw   efSearch={ef:<4d}         p50 {r['p50_ms']:8.4f}ms  "
                  f"{r['qps']:>8.1f} qps   recall@1 {r['recall@1']:.3f}  "
                  f"recall@10 {r['recall@10']:.3f}")

        results[str(n)] = row
        del gallery

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "faiss_version": faiss.__version__,
        "dim": DIM,
        "queries": args.queries,
        "top_k": args.top_k,
        "limitation": "Synthetic unit vectors. Recall is a property of index "
                      "geometry, but real ArcFace galleries cluster by identity "
                      "and may recall differently. Re-measure on real templates "
                      "before adopting an approximate index in production.",
        "results": results,
    }, indent=2))
    print(f"\nWrote {out}")
    print("\nRecall, not speed, decides adoption: a missed candidate is an")
    print("investigative lead that silently never surfaced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
