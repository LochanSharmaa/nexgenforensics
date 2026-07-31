#!/usr/bin/env python
"""
Item 32 — regression gate for model / threshold / fusion changes.

    python backend/scripts/regression_check.py              # check, exit 1 on regression
    python backend/scripts/regression_check.py --update     # re-record the baseline

Run this after ANY change to the model pack, decision thresholds, fusion
method, or embedding pipeline. It exits non-zero on regression so it can gate
a merge.

WHAT IT COMPARES, AND WHY BOTH HALVES MATTER
--------------------------------------------
1. ACCURACY, per dataset, recomputed from the cached embeddings the published
   figures came from. Catches "the change made the model worse".

2. CONFIGURATION INVARIANTS -- deployed model pack, the three thresholds, the
   single-source-of-truth wiring. Catches "the numbers are fine but the service
   is no longer running what was measured", which is the failure this project
   actually hit: thresholds existed in four places and drifted, and the API
   reported a decision rule the engine had stopped applying.

A gate that checked only accuracy would have passed every one of those.

TOLERANCE IS DIRECTIONAL
------------------------
An accuracy *drop* beyond tolerance fails. An accuracy *rise* is reported but
does not fail — improvements should not require a ceremony to land. The
tolerance is expressed in accuracy points and defaults to 0.30, comfortably
inside the ±0.26–1.17 fold standard deviations in BENCHMARKS.md §2, so normal
fold noise cannot trip it while a real regression will.

Thresholds and model pack are compared EXACTLY. There is no tolerance on
"which model is deployed".

WHY IT USES CACHED EMBEDDINGS
-----------------------------
Re-embedding 31,000 pairs per run would make this too slow to gate a merge, and
the point is to detect a change in the *scoring* path against a fixed input.
If the embedding cache is stale relative to the model, that is itself caught by
the configuration check. Delete runtime/benchmarks/embeddings/ and re-run
benchmark_verification.py when the model changes.
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
BASELINE = _BACKEND / "regression_baseline.json"

DATASETS = ["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"]


def current_config() -> dict:
    """The configuration the service would actually run with right now."""
    from imatch_api.core.config import Settings
    from nexgen_engine.config import EngineConfig, ThresholdConfig

    t = ThresholdConfig()
    s = Settings()
    e = EngineConfig()
    return {
        "model_pack": e.model_pack,
        "threshold_match": t.match,
        "threshold_review": t.review,
        "threshold_verify": t.verify,
        # The API must DERIVE from the engine, never hold its own copy.
        "api_derives_from_engine": (
            s.match_threshold == t.match
            and s.review_threshold == t.review
            and s.verify_threshold == t.verify
        ),
        "api_model_pack": s.model_pack,
        "use_flip_tta": e.use_flip_tta,
    }


def measure(model: str) -> dict:
    """Accuracy per dataset from cached embeddings. Missing cache -> skipped."""
    out = {}
    for ds in DATASETS:
        p = CACHE / f"{ds}__{model}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        e = l2n((d["orig"] + d["flip"]).astype(np.float64))
        issame = np.asarray(d["issame"], dtype=bool)
        r = evaluate_pairs(e[0::2], e[1::2], issame, ds, model)
        out[ds] = {
            "accuracy": round(r.accuracy_mean * 100, 4),
            "std": round(r.accuracy_std * 100, 4),
            "auc": round(r.auc, 6),
            "pairs": r.n_pairs,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="w600k_r50", help="model whose cached embeddings to score")
    ap.add_argument("--tolerance", type=float, default=0.30, help="accuracy points of allowed drop")
    ap.add_argument("--update", action="store_true", help="re-record the baseline instead of checking")
    args = ap.parse_args()

    config = current_config()
    accuracy = measure(args.model)

    if args.update:
        BASELINE.write_text(json.dumps({
            "model": args.model,
            "note": "Regenerated deliberately. Only update this after reviewing "
                    "WHY the numbers moved -- a baseline refreshed to make a "
                    "failing gate pass defeats the gate.",
            "config": config,
            "accuracy": accuracy,
        }, indent=2))
        print(f"Baseline written to {BASELINE}")
        print(f"  {len(accuracy)} datasets, model {args.model}")
        return 0

    if not BASELINE.exists():
        print(f"No baseline at {BASELINE}. Create one with --update.")
        return 1

    base = json.loads(BASELINE.read_text())
    failures: list[str] = []
    notes: list[str] = []

    print("=" * 74)
    print("  Regression check")
    print("=" * 74)

    # ---- configuration: exact match, no tolerance ----
    print("\nConfiguration")
    for key, expected in base["config"].items():
        actual = config.get(key)
        ok = actual == expected
        print(f"  {'OK  ' if ok else 'FAIL'}  {key:26s} {actual!r}"
              + ("" if ok else f"   expected {expected!r}"))
        if not ok:
            failures.append(f"config.{key}: {actual!r} != baseline {expected!r}")

    if not config["api_derives_from_engine"]:
        failures.append(
            "api_derives_from_engine is False: imatch_api thresholds no longer "
            "derive from nexgen_engine.config.ThresholdConfig. A second copy has "
            "reappeared."
        )

    # ---- accuracy: directional tolerance ----
    print(f"\nAccuracy (tolerance {args.tolerance:.2f} points, drops only)")
    for ds, exp in base["accuracy"].items():
        cur = accuracy.get(ds)
        if cur is None:
            notes.append(f"{ds}: no cached embeddings; SKIPPED (not a pass)")
            print(f"  SKIP  {ds:10s} no cached embeddings")
            continue
        delta = cur["accuracy"] - exp["accuracy"]
        regressed = delta < -args.tolerance
        tag = "FAIL" if regressed else ("OK  " if delta >= 0 else "OK  ")
        print(f"  {tag}  {ds:10s} {cur['accuracy']:7.2f}%  baseline {exp['accuracy']:7.2f}%  "
              f"delta {delta:+.2f}")
        if regressed:
            failures.append(f"{ds}: accuracy {cur['accuracy']:.2f}% is {abs(delta):.2f} "
                            f"points below baseline {exp['accuracy']:.2f}%")
        elif delta > args.tolerance:
            notes.append(f"{ds}: improved {delta:+.2f} points — update the baseline "
                         f"once you know why")

    print("\n" + "=" * 74)
    for n in notes:
        print(f"  NOTE  {n}")
    if failures:
        print(f"  RESULT: FAIL ({len(failures)} regression(s))")
        for f in failures:
            print(f"    - {f}")
        print("\n  Do NOT refresh the baseline to silence this. Establish why it moved.")
        return 1
    print(f"  RESULT: PASS ({len(accuracy)} datasets, config matches baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
