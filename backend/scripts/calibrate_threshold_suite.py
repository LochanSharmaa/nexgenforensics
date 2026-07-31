#!/usr/bin/env python
"""
Calibrate the decision threshold across the WHOLE published benchmark suite.

    python backend/scripts/calibrate_threshold_suite.py --model w600k_r50

WHY NOT scripts/calibrate_threshold.py
--------------------------------------
That script calibrates against a folder-per-identity dataset and constructs its
own pairs. It cannot read the `.bin` protocol packs, so it cannot calibrate
across LFW / AgeDB-30 / CFP-FP / CALFW / CPLFW — the published pair lists this
project reports accuracy on. Calibrating on one dataset and deploying the
result across all of them is exactly the mistake this script exists to avoid.

WHAT IT DOES
    1. Loads the cached embeddings for each dataset (same ones the accuracy
       benchmark used, so the calibration and the reported accuracy cannot
       disagree).
    2. Pools every impostor score across all datasets and finds the single
       threshold giving a target FMR on the combined distribution.
    3. Reports each dataset's own FMR=target threshold, so the spread between
       datasets is visible rather than hidden inside an average.
    4. Reports accuracy / FNMR / FMR per dataset at both the incumbent and the
       proposed threshold.

A combined threshold is a compromise. If per-dataset thresholds differ widely,
no single number serves them all, and that must be visible in the output rather
than discovered later in production.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.benchmarks.verification import l2n  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"


def load_scores(dataset: str, model: str, use_flip: bool = True):
    """Return (scores, labels) for one dataset from the cached embeddings."""
    path = CACHE / f"{dataset}__{model}.npz"
    if not path.exists():
        return None, None
    d = np.load(path)
    e = d["orig"] + d["flip"] if use_flip else d["orig"]
    e = l2n(e.astype(np.float64))
    issame = np.asarray(d["issame"], dtype=bool)
    scores = np.sum(e[0::2] * e[1::2], axis=1)
    return scores, issame


def threshold_at_fmr(impostor_scores: np.ndarray, target_fmr: float) -> float:
    """Lowest threshold whose false-match rate does not exceed target_fmr.

    Nearest-rank on the sorted impostor scores: the returned value is an
    observed score, not an interpolation between two that never occurred.
    """
    s = np.sort(impostor_scores)[::-1]
    k = int(np.floor(target_fmr * s.size))
    if k <= 0:
        return float(s[0]) + 1e-9
    return float(s[k - 1])


def rates(scores, labels, thr):
    gen, imp = scores[labels], scores[~labels]
    fnmr = float((gen <= thr).mean()) if gen.size else float("nan")
    fmr = float((imp > thr).mean()) if imp.size else float("nan")
    acc = float(((scores > thr) == labels).mean())
    return acc, fnmr, fmr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--datasets", nargs="+",
                    default=["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"])
    ap.add_argument("--target-fmr", type=float, default=0.001)
    ap.add_argument("--incumbent", type=float, default=0.20)
    ap.add_argument("--candidate", type=float, default=0.2871,
                    help="threshold proposed from a single dataset, for comparison")
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/threshold_calibration.json"))
    args = ap.parse_args()

    print("=" * 78)
    print(f"  Threshold calibration across the full suite - model {args.model}")
    print(f"  target FMR = {args.target_fmr * 100:.3f}%")
    print("=" * 78)

    per = {}
    all_imp, all_gen = [], []
    missing = []
    for ds in args.datasets:
        s, y = load_scores(ds, args.model)
        if s is None:
            missing.append(ds)
            continue
        per[ds] = (s, y)
        all_imp.append(s[~y])
        all_gen.append(s[y])

    if missing:
        print(f"\n!! no cached embeddings for: {missing}")
        print("   run benchmark_verification.py for these first; they are EXCLUDED below.")
    if not per:
        print("no data")
        return 1

    pooled_imp = np.concatenate(all_imp)
    pooled_gen = np.concatenate(all_gen)
    combined = threshold_at_fmr(pooled_imp, args.target_fmr)

    print(f"\nPooled across {len(per)} datasets: "
          f"{pooled_gen.size:,} genuine + {pooled_imp.size:,} impostor pairs")
    print(f"\n>>> COMBINED threshold at FMR={args.target_fmr * 100:.3f}%: {combined:.4f}")
    print(f"    (single-dataset candidate was {args.candidate:.4f}; "
          f"difference {combined - args.candidate:+.4f})")

    print(f"\nPer-dataset threshold at FMR={args.target_fmr * 100:.3f}% "
          f"(spread shows whether one number can serve all):")
    own = {}
    for ds, (s, y) in per.items():
        t = threshold_at_fmr(s[~y], args.target_fmr)
        own[ds] = t
        print(f"    {ds:10s} {t:.4f}")
    spread = max(own.values()) - min(own.values())
    print(f"    {'spread':10s} {spread:.4f}"
          + ("   <-- WIDE: one threshold cannot serve all datasets equally"
             if spread > 0.05 else "   (tight)"))

    print(f"\n{'dataset':10s} | {'accuracy':>17s} | {'FNMR %':>15s} | {'FMR %':>15s}")
    print(f"{'':10s} | {'old':>8s} {'new':>8s} | {'old':>7s} {'new':>7s} | {'old':>7s} {'new':>7s}")
    print("-" * 66)
    table = {}
    for ds, (s, y) in per.items():
        a0, f0, m0 = rates(s, y, args.incumbent)
        a1, f1, m1 = rates(s, y, combined)
        table[ds] = {
            "incumbent": {"threshold": args.incumbent, "accuracy": a0, "fnmr": f0, "fmr": m0},
            "combined": {"threshold": combined, "accuracy": a1, "fnmr": f1, "fmr": m1},
            "own_fmr_threshold": own[ds],
        }
        print(f"{ds:10s} | {a0 * 100:7.2f}% {a1 * 100:7.2f}% | "
              f"{f0 * 100:6.2f}% {f1 * 100:6.2f}% | {m0 * 100:6.2f}% {m1 * 100:6.2f}%")

    ga0, gf0, gm0 = rates(np.concatenate([s for s, _ in per.values()]),
                          np.concatenate([y for _, y in per.values()]), args.incumbent)
    ga1, gf1, gm1 = rates(np.concatenate([s for s, _ in per.values()]),
                          np.concatenate([y for _, y in per.values()]), combined)
    print("-" * 66)
    print(f"{'POOLED':10s} | {ga0 * 100:7.2f}% {ga1 * 100:7.2f}% | "
          f"{gf0 * 100:6.2f}% {gf1 * 100:6.2f}% | {gm0 * 100:6.2f}% {gm1 * 100:6.2f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "target_fmr": args.target_fmr,
        "combined_threshold": combined,
        "single_dataset_candidate": args.candidate,
        "per_dataset_own_threshold": own,
        "spread": spread,
        "datasets_excluded_no_cache": missing,
        "per_dataset": table,
    }, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
