#!/usr/bin/env python
"""
Item 35 — which NEXGEN_FUSION_METHOD is best, per condition.

    python backend/scripts/benchmark_fusion.py

The production default is `single_glintr100`, chosen in BENCHMARKS.md §3 on
clean-protocol accuracy. But the DEPLOYED pack is `buffalo_l` / `w600k_r50`,
kept because it is far better on degraded imagery (§4). Those two facts sit
uneasily together, and nobody had measured whether the best fusion differs by
condition rather than being one global winner.

This scores every fusion method from the cached embeddings, split by condition:

  clean     LFW, CALFW, CPLFW, CFP-FP, AgeDB-30 — the published protocols
  degraded  TinyFace, if its embeddings are cached

If one method wins everywhere, the default is justified and this is a
one-paragraph confirmation. If the winner differs by condition, then a single
static default is the wrong shape and the engine should select on the measured
quality of the probe.

Runs entirely from cached embeddings, so it costs no GPU time and cannot
disagree with the accuracy figures already published — same inputs, same
harness.
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

from nexgen_engine.benchmarks.verification import evaluate_pairs, l2n  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
CLEAN = ["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"]
MODELS = ["w600k_r50", "glintr100", "w600k_mbf"]


def load(ds: str) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    embs, issame = {}, None
    for m in MODELS:
        p = CACHE / f"{ds}__{m}.npz"
        if not p.exists():
            return None
        d = np.load(p)
        embs[m] = l2n((d["orig"] + d["flip"]).astype(np.float64))
        issame = np.asarray(d["issame"], dtype=bool)
    return embs, issame


def fusions(e: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    r50, r100, mbf = e["w600k_r50"], e["glintr100"], e["w600k_mbf"]
    return {
        "single_r50 (DEPLOYED pack)": r50,
        "single_glintr100 (default)": r100,
        "single_mbf": mbf,
        "dual r50+r100": l2n(0.5 * r50 + 0.5 * r100),
        "weighted .45/.45/.10": l2n(0.45 * r50 + 0.45 * r100 + 0.10 * mbf),
        "equal 1/3": l2n((r50 + r100 + mbf) / 3.0),
        "concat 1536-d": np.concatenate([r50, r100, mbf], axis=1) / np.sqrt(3.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/fusion_selection.json"))
    args = ap.parse_args()

    print("=" * 78)
    print("  Item 35 - fusion method by condition")
    print("=" * 78)

    results: dict[str, dict[str, float]] = {}
    datasets_used = []

    for ds in CLEAN + ["tinyface"]:
        loaded = load(ds)
        if loaded is None:
            print(f"  {ds}: no cached embeddings for all three models; SKIPPED")
            continue
        embs, issame = loaded
        datasets_used.append(ds)
        for name, e in fusions(embs).items():
            r = evaluate_pairs(e[0::2], e[1::2], issame, ds, name)
            results.setdefault(name, {})[ds] = round(r.accuracy_mean * 100, 3)

    if not results:
        print("no cached embeddings; run benchmark_verification.py first")
        return 1

    clean_used = [d for d in datasets_used if d in CLEAN]
    degraded_used = [d for d in datasets_used if d == "tinyface"]

    print(f"\n{'fusion method':30s} " + " ".join(f"{d[:9]:>9s}" for d in datasets_used)
          + f" {'CLEAN avg':>10s}")
    print("-" * (30 + 10 * len(datasets_used) + 11))
    rows = []
    for name, per in results.items():
        clean_avg = float(np.mean([per[d] for d in clean_used])) if clean_used else float("nan")
        rows.append((name, per, clean_avg))
        cells = " ".join(f"{per.get(d, float('nan')):9.2f}" for d in datasets_used)
        print(f"{name:30s} {cells} {clean_avg:10.2f}")

    best_clean = max(rows, key=lambda r: r[2])
    print(f"\n  Best on CLEAN protocols : {best_clean[0]}  ({best_clean[2]:.2f}%)")
    if degraded_used:
        best_deg = max(rows, key=lambda r: r[1].get("tinyface", -1))
        print(f"  Best on DEGRADED        : {best_deg[0]}  "
              f"({best_deg[1]['tinyface']:.2f}%)")
        if best_deg[0] != best_clean[0]:
            print("\n  >>> The winner DIFFERS by condition. A single static default is")
            print("      the wrong shape; selection should follow probe quality.")
        else:
            print("\n  >>> Same winner in both conditions; a static default is justified.")
    else:
        print("  DEGRADED: tinyface embeddings are not cached for all three models,")
        print("  so the condition-dependence question CANNOT be answered here.")
        print("  Run: benchmark_tinyface.py for each model, then re-run this.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "datasets": datasets_used,
        "note": "Scored from cached embeddings, same inputs as BENCHMARKS.md §2.",
        "accuracy_pct": results,
    }, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
