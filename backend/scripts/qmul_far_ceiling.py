#!/usr/bin/env python
"""The FAR=1e-6 feasibility inversion on QMUL-SurvFace, plus an impostor-tail audit.

    python backend/scripts/qmul_far_ceiling.py --model w600k_r50
    python backend/scripts/qmul_far_ceiling.py --model vit_kprpe_wf12m

The IJB counterpart is far_1e6_feasibility.py; this is the surveillance-corpus
version, and the one that matters operationally, since QMUL is the only corpus
here whose imagery resembles the deployment condition.

REPRODUCES measure_capacity_official.py's QMUL pair construction EXACTLY -- same
seed, same 500k genuine cap, same 50M impostor sample, and critically the same
ORDER of rng draws, because the genuine subsample consumes rng state before the
impostor indices are drawn. Deviating there silently produces a different pool
and the numbers stop being comparable to capacity_official_qmul__*.json.

The tail audit exists because the raw impostor maximum on this corpus is exactly
1.0000: two pairs in 50M are the same image carrying two identity labels. That
does not move any operating point, but it does set the maximum, so the FAR->0
ceiling must be computed on a cleaned pool or it reports a labelling defect as a
recognition result.

CPU only: reads cached embeddings, touches no GPU.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
OUT = _ROOT / "runtime" / "forensics"

DUP_COSINE = 0.999  # above this, two crops are the same image, not two people


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def build_pools(model: str, n_imp: int, seed: int):
    cache = CACHE / f"qmul_ident__{model}.npz"
    if not cache.exists():
        raise SystemExit(f"no embedding cache at {cache}. Run embed_qmul_ident.py first.")
    d = np.load(cache)
    emb = l2n(d["emb"].astype(np.float64))
    split, ids = d["split"], d["ids"]
    gal_emb, gal_ids = emb[split == "gallery"], ids[split == "gallery"]
    prb_emb, prb_ids = emb[split == "mated"], ids[split == "mated"]
    non_emb, non_ids = emb[split == "unmated"], ids[split == "unmated"]
    print(f"QMUL / {model}: gallery {gal_emb.shape[0]:,} / mated {prb_emb.shape[0]:,} "
          f"/ unmated {non_emb.shape[0]:,}")

    rng = np.random.default_rng(seed)
    gal_by = defaultdict(list)
    for k, i in enumerate(gal_ids):
        gal_by[int(i)].append(k)
    ga, pa = [], []
    for k, i in enumerate(prb_ids):
        for g in gal_by.get(int(i), []):
            ga.append(g)
            pa.append(k)
    ga, pa = np.array(ga), np.array(pa)
    if ga.size > 500_000:
        sel = rng.choice(ga.size, 500_000, replace=False)  # consumes rng state; order matters
        ga, pa = ga[sel], pa[sel]
    gen = np.einsum("ij,ij->i", gal_emb[ga], prb_emb[pa])

    a = rng.integers(0, prb_emb.shape[0], size=n_imp)
    b = rng.integers(0, non_emb.shape[0], size=n_imp)
    imp = np.empty(n_imp, dtype=np.float32)
    for s in range(0, n_imp, 1_000_000):
        e = min(s + 1_000_000, n_imp)
        imp[s:e] = np.einsum("ij,ij->i", prb_emb[a[s:e]], non_emb[b[s:e]]).astype(np.float32)
    return gen, imp, a, b, prb_ids, non_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--impostor-samples", type=int, default=50_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-tar", type=float, default=0.985)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    gen, imp, a, b, prb_ids, non_ids = build_pools(
        args.model, args.impostor_samples, args.seed)
    n_gen, n_imp = gen.size, imp.size
    print(f"genuine {n_gen:,} / impostor {n_imp:,}\n")

    # ---- tail audit, before any figure is quoted off the maximum ------------
    k = 200
    idx = np.argpartition(imp, n_imp - k)[n_imp - k :]
    top_a, top_b, top_s = a[idx], b[idx], imp[idx]
    n_dup = int((imp > DUP_COSINE).sum())
    same_label = int((prb_ids[top_a] == non_ids[top_b]).sum())
    print("TAIL AUDIT")
    print(f"  pairs above cosine {DUP_COSINE}: {n_dup}  <- same image, two identity labels")
    print(f"  top {k}: {len(set(prb_ids[top_a].tolist()))} distinct probe identities, "
          f"{len(set(non_ids[top_b].tolist()))} distinct non-mate identities")
    print(f"  top {k} pairs sharing an identity label: {same_label} "
          f"(nonzero would be a protocol violation)")
    del a, b, top_a, top_b, top_s

    imp.sort()
    clean = imp[imp <= DUP_COSINE]

    print("\n1. OPERATING POINTS (exact order statistic)")
    print(f"{'FAR':>8} {'threshold':>11} {'TAR':>10} {'impostors above':>18}")
    rows = {}
    for far in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        kk = int(np.ceil(far * n_imp))
        thr = float(imp[-kk])
        tar = float((gen > thr).mean())
        rows[far] = (thr, tar, kk)
        print(f"{far:>8.0e} {thr:>11.4f} {tar*100:>9.3f}% {kk:>18,}")

    print(f"\n2. STABILITY AT 1e-6 (Poissonised bootstrap, {args.boot} reps)")
    rng = np.random.default_rng(20260804)
    M = 5000
    k6 = int(np.ceil(1e-6 * n_imp))
    top = imp[-M:][::-1]
    boot = np.empty(args.boot)
    for i in range(args.boot):
        cum = np.cumsum(rng.poisson(1.0, size=M))
        thr_b = float(top[min(int(np.searchsorted(cum, k6)), M - 1)])
        boot[i] = (gen[rng.integers(0, n_gen, n_gen)] > thr_b).mean()
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    print(f"  TAR@FAR=1e-6 = {rows[1e-6][1]*100:.3f}%  95% CI [{lo*100:.3f}%, {hi*100:.3f}%]"
          f"  on {k6} pairs")

    print(f"\n3. INVERSION -- FAR actually delivered at TAR = {args.target_tar*100:.1f}%")
    thr_t = float(np.quantile(gen, 1.0 - args.target_tar))
    n_above = n_imp - int(np.searchsorted(imp, thr_t, side="right"))
    far_t = n_above / n_imp
    print(f"  threshold {thr_t:.4f}  =>  FAR {far_t:.4f} "
          f"({far_t*100:.2f}% of impostors accepted, {far_t/1e-6:,.0f}x the target)")

    print("\n4. HARD CEILING (threshold above every impostor, duplicates removed)")
    mx_raw, mx_cl = float(imp[-1]), float(clean[-1])
    ceiling = float((gen > mx_cl).mean())
    print(f"  max impostor {mx_raw:.4f} raw -> {mx_cl:.4f} cleaned")
    print(f"  highest genuine {gen.max():.4f}")
    print(f"  TAR above every impostor: {ceiling*100:.4f}%  "
          f"({int((gen > mx_cl).sum())} of {n_gen:,})")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "qmul_official",
        "model": args.model,
        "n_genuine": int(n_gen),
        "n_impostor": int(n_imp),
        "tail_audit": {
            "dup_cosine_threshold": DUP_COSINE,
            "pairs_above": n_dup,
            "top200_pairs_sharing_identity_label": same_label,
        },
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
        "ceiling_far_zero_cleaned": ceiling,
        "max_impostor_raw": mx_raw,
        "max_impostor_cleaned": mx_cl,
        "max_genuine": float(gen.max()),
    }
    out = OUT / f"qmul_far_ceiling__{args.model}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
