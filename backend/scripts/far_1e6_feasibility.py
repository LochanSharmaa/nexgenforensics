#!/usr/bin/env python
"""How far must the IJB ROC move to reach a given TAR at FAR=1e-6?

    python backend/scripts/far_1e6_feasibility.py --dataset IJBC --model w600k_r50
    python backend/scripts/far_1e6_feasibility.py --dataset IJBC --model vit_kprpe_wf12m

WHY THIS EXISTS
---------------
Written to answer "make it 98.5% TAR @ FAR=1e-6" with arithmetic instead of
opinion. Reading TAR off the ROC at a fixed FAR understates how far away a
target is, because it hides how few impostor pairs define the threshold. This
inverts the question -- at the threshold where TAR hits the target, what FAR do
we actually deliver? -- and puts a confidence interval on the 1e-6 point.

Reads cached embeddings from benchmark_ijb.py. No GPU, no re-embedding.

Four outputs:
  1. Operating points by exact order statistic, with the pair count supporting
     each. At 1e-6 that count is 16 on IJB-C and 8 on IJB-B.
  2. A Poissonised bootstrap CI on the 1e-6 point. The bootstrap resamples the
     impostor tail by drawing Poisson(1) multiplicities for the top scores,
     which is the standard Poissonisation of the nonparametric bootstrap and
     avoids materialising a 15.6M-element resample 2,000 times.
  3. The inversion: FAR actually delivered at the target TAR.
  4. The hard ceiling: TAR above the single highest impostor score, i.e. the
     best achievable if the threshold admitted no impostor at all.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
IJB = _ROOT / "src_extracted" / "ijb" / "ijb"
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
OUT = _ROOT / "runtime" / "forensics"


def pool_templates(feats: np.ndarray, tid: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """Media-aware pooling, identical to benchmark_ijb.pool_templates."""
    feats = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)
    uniq = np.unique(tid)
    out = np.zeros((uniq.size, feats.shape[1]), dtype=np.float32)
    order = np.argsort(tid, kind="stable")
    tid_s = tid[order]
    bounds = np.append(np.searchsorted(tid_s, uniq, side="left"), tid_s.size)
    for k in range(uniq.size):
        sel = order[bounds[k] : bounds[k + 1]]
        media = mid[sel]
        acc = []
        for m in np.unique(media):
            f = feats[sel[media == m]]
            acc.append(f[0] if f.shape[0] == 1 else f.mean(axis=0))
        out[k] = np.sum(np.stack(acc), axis=0)
    out /= np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
    return out, uniq


def build_scores(ds: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    low = ds.lower()
    meta = IJB / ds / "meta"
    tm = np.loadtxt(meta / f"{low}_face_tid_mid.txt", dtype=np.int64, usecols=(1, 2))
    tid, mid = tm[:, 0], tm[:, 1]
    t0 = time.time()
    pairs = np.loadtxt(meta / f"{low}_template_pair_label.txt", dtype=np.int64)
    print(f"  {pairs.shape[0]:,} pairs read in {time.time()-t0:.0f}s", flush=True)

    cache = CACHE / f"{low}__{model}.npz"
    if not cache.exists():
        raise SystemExit(f"no embedding cache at {cache}. Run benchmark_ijb.py first.")
    feats = np.load(cache)["emb"]

    tfeat, uniq = pool_templates(feats, tid, mid)
    pos = {int(t): i for i, t in enumerate(uniq)}
    p1 = np.array([pos[int(x)] for x in pairs[:, 0]])
    p2 = np.array([pos[int(x)] for x in pairs[:, 1]])
    labels = pairs[:, 2].astype(bool)
    scores = np.empty(p1.size, dtype=np.float32)
    for s in range(0, p1.size, 2_000_000):
        e = min(s + 2_000_000, p1.size)
        scores[s:e] = np.einsum("ij,ij->i", tfeat[p1[s:e]], tfeat[p2[s:e]])
    return scores[labels], np.sort(scores[~labels])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="IJBC", choices=["IJBB", "IJBC"])
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--target-tar", type=float, default=0.985)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    gen, imp = build_scores(args.dataset, args.model)
    n_gen, n_imp = gen.size, imp.size
    print(f"\n{args.dataset} / {args.model}: {n_gen:,} genuine, {n_imp:,} impostor\n")

    print("1. OPERATING POINTS (exact order statistic)")
    print(f"{'FAR':>8} {'threshold':>11} {'TAR':>9} {'impostors above':>18}")
    rows = OrderedDict()
    for far in (1e-3, 1e-4, 1e-5, 1e-6):
        k = int(np.ceil(far * n_imp))
        thr = float(imp[-k])
        tar = float((gen > thr).mean())
        rows[far] = (thr, tar, k)
        print(f"{far:>8.0e} {thr:>11.4f} {tar*100:>8.2f}% {k:>18,}")

    print(f"\n2. STABILITY AT 1e-6 (Poissonised bootstrap, {args.boot} reps)")
    rng = np.random.default_rng(args.seed)
    M = 5000
    k6 = int(np.ceil(1e-6 * n_imp))
    top = imp[-M:][::-1]
    boot = np.empty(args.boot)
    for b in range(args.boot):
        cum = np.cumsum(rng.poisson(1.0, size=M))
        thr_b = float(top[min(int(np.searchsorted(cum, k6)), M - 1)])
        boot[b] = (gen[rng.integers(0, n_gen, n_gen)] > thr_b).mean()
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    print(f"  TAR@FAR=1e-6 = {rows[1e-6][1]*100:.2f}%  95% CI [{lo*100:.2f}%, {hi*100:.2f}%]")
    print(f"  CI width {(hi-lo)*100:.2f} points, on a threshold set by {k6} impostor pairs")

    print(f"\n3. INVERSION -- FAR actually delivered at TAR = {args.target_tar*100:.1f}%")
    thr_t = float(np.quantile(gen, 1.0 - args.target_tar))
    n_above = n_imp - int(np.searchsorted(imp, thr_t, side="right"))
    far_t = n_above / n_imp
    print(f"  threshold {thr_t:.4f}   impostors above {n_above:,} of {n_imp:,}")
    print(f"  => FAR {far_t:.3e}  ({far_t/1e-6:,.0f}x the 1e-6 target)")

    print("\n4. HARD CEILING (threshold above every impostor)")
    mx = float(imp[-1])
    ceiling = float((gen > mx).mean())
    print(f"  highest impostor {mx:.4f}   TAR above it {ceiling*100:.2f}%")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "n_genuine": int(n_gen),
        "n_impostor": int(n_imp),
        "operating_points": {
            f"{f:g}": {"threshold": rows[f][0], "tar": rows[f][1], "impostors_above": rows[f][2]}
            for f in rows
        },
        "tar_at_1e-6_ci95": [lo, hi],
        "target": {
            "tar": args.target_tar,
            "threshold": thr_t,
            "actual_far_at_target_tar": far_t,
            "impostors_above": int(n_above),
        },
        "ceiling_far_zero": ceiling,
        "max_impostor_score": mx,
    }
    out = OUT / f"ijb_{args.dataset.lower()}_far1e6_feasibility__{args.model}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
