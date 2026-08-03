#!/usr/bin/env python
"""S0.3 enhancement arms: does pre-processing the probe help the existing recogniser?

    # CPU plumbing check -- zero inference, zero GPU, proves the whole path works
    python experiments/S0_3/run_enhancement.py --dataset lfw_synth --embedder stub --pairs 40

    # The real run
    python experiments/S0_3/run_enhancement.py --dataset scface --distance 1 --embedder arcface

NOTHING ABOUT THE RECOGNISER CHANGES. The model, its weights, its thresholds,
the dataset loaders and the metrics are the ones already in run_gpu.py, imported
rather than reimplemented. The only new thing is a transform applied to the
probe before it is embedded.

DECISION RULE, FIXED HERE BEFORE THE RUN so it cannot be adjusted afterwards to
suit the outcome. It is the same rule and the same structure S0.3 used for
B2-B1, which failed its gate and was published as a failure:

    ADOPT       arm E improves TAR@FAR=0.1% over arm A by >= +2.0 points
                AND the bootstrap 95% CI on the paired difference excludes zero
    REJECT      the difference is <= +0.5 points, or the CI includes zero
    otherwise   INCONCLUSIVE -- report it, and do not build on it

Arm A is the untransformed baseline and is recomputed here on the same pairs
rather than read from the earlier result files, because a paired bootstrap needs
both arms scored on the *same* draws. The absolute A figure should reproduce the
published one; if it does not, that is a loader or seed problem and the run is
void.

E0 IS THE CONTROL THAT MAKES THE REST INTERPRETABLE. The embedder resamples
every input to 112x112, so any arm that changes the image size also changes the
resampling path. E0 is Lanczos upscaling and nothing else. An arm that does not
beat E0 has not demonstrated restoration -- it has demonstrated interpolation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_BACKEND = _ROOT / "backend"
for _p in (str(_BACKEND), str(_HERE), str(_BACKEND / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from arms_enhancement import ENHANCEMENT_ARMS, EnhancementArm  # noqa: E402
from run_gpu import LOADERS, _scface_pairs, auc, load_embedder, tar_at_far  # noqa: E402

from nexgen_engine.enhancement.cache import EnhancementCache  # noqa: E402
from nexgen_engine.enhancement.vram import device_report  # noqa: E402

RESULTS = _HERE / "results"
DECISION = {
    "adopt_threshold_points": 2.0,
    "reject_threshold_points": 0.5,
    "metric": "TAR@FAR=0.1% (points)",
    "comparison": "enhancement arm minus arm A (untransformed baseline)",
    "control": "E0 (Lanczos upscale only). An arm must also beat E0 to demonstrate restoration.",
}


def paired_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    far: float = 1e-3,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap the paired difference (b - a) in TAR@FAR, in points.

    Resampling PAIRS, not arms: both arms are scored on the same draw, which is
    what makes the interval a statement about the difference rather than about
    two independent estimates that happen to be subtracted.
    """
    rng = np.random.default_rng(seed)
    n = labels.size
    point = (tar_at_far(scores_b, labels, far) - tar_at_far(scores_a, labels, far)) * 100.0
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        lab = labels[idx]
        if lab.sum() == 0 or (~lab).sum() == 0:
            diffs[i] = np.nan
            continue
        diffs[i] = (tar_at_far(scores_b[idx], lab, far) - tar_at_far(scores_a[idx], lab, far)) * 100.0
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def verdict(delta: float, lo: float, hi: float) -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "INCONCLUSIVE"
    if delta >= DECISION["adopt_threshold_points"] and lo > 0:
        return "ADOPT"
    if delta <= DECISION["reject_threshold_points"] or lo <= 0 <= hi:
        return "REJECT"
    return "INCONCLUSIVE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--embedder", default="arcface", choices=["stub", "arcface"])
    ap.add_argument("--pairs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distance", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--chunk", type=int, default=500, help="pairs held in memory at once")
    ap.add_argument("--arms", default="", help="comma-separated arm names; default is every available arm")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--cache-dir", default=str(_ROOT / "runtime" / "enhancement_cache"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    cache = None if args.no_cache else EnhancementCache(Path(args.cache_dir))

    requested = [a.strip() for a in args.arms.split(",") if a.strip()] or list(ENHANCEMENT_ARMS)
    arms: dict[str, EnhancementArm] = {}
    unavailable: dict[str, str] = {}
    for name in requested:
        if name not in ENHANCEMENT_ARMS:
            raise SystemExit(f"unknown arm {name!r}; known: {', '.join(ENHANCEMENT_ARMS)}")
        arm = ENHANCEMENT_ARMS[name]
        ok, reason = arm.availability()
        if ok:
            arms[name] = arm
        else:
            unavailable[name] = reason

    if unavailable:
        print("arms skipped for missing dependencies:")
        for name, reason in unavailable.items():
            print(f"  {name}: {reason}")
    if not arms:
        raise SystemExit(
            "No enhancement arm is runnable on this host. The classical arms (E0, E1) need only "
            "opencv; if those are unavailable the install is broken rather than merely unseeded."
        )

    embed, embed_name = load_embedder(args.embedder)
    print(f"embedder: {embed_name}")
    print(f"arms:     {', '.join(arms)}")

    if args.dataset == "scface":
        args.pairs = len(_scface_pairs(args.distance, args.seed)[2])
        print(f"scface distance {args.distance}: {args.pairs:,} enumerated pairs")

    chunk = max(1, min(args.chunk, args.pairs))
    n_chunks = (args.pairs + chunk - 1) // chunk

    names = ["A", *arms]
    accumulated: dict[str, list[np.ndarray]] = {name: [] for name in names}
    all_labels: list[np.ndarray] = []
    arm_reports: dict[str, dict] = {}

    started = time.time()
    for index in range(n_chunks):
        take = min(chunk, args.pairs - index * chunk)
        if args.dataset == "scface":
            gallery, probes, labels = LOADERS["scface"](take, args.seed, args.distance, offset=index * chunk)
            if not gallery:
                break
        else:
            gallery, probes, labels = LOADERS[args.dataset](take, args.seed + index * 7919)
        all_labels.append(labels)

        # Arm A: the untransformed baseline, on exactly these pairs.
        gallery_embedded = embed(gallery)
        accumulated["A"].append(np.sum(gallery_embedded * embed(probes), axis=1))

        for name, arm in arms.items():
            # ENHANCE -> RELEASE VRAM -> EMBED. transform_batch drops every
            # model and empties the cache before returning, so the embedder
            # below finds the card free. On 6 GB this ordering is the difference
            # between a run and an OOM.
            arm_started = time.time()
            enhanced, report = arm.transform_batch(
                probes, device=args.device, cache=cache, progress=lambda m: print(m, flush=True)
            )
            enhance_s = time.time() - arm_started
            accumulated[name].append(np.sum(gallery_embedded * embed(enhanced), axis=1))
            arm_reports.setdefault(name, report)
            # One line per arm per chunk, flushed: this is the only way to see
            # WHERE a slow run is spending its time from outside the process.
            print(
                f"  {name}: enhance {enhance_s:.1f}s (hits {report.get('cache_hits', 0)}, "
                f"computed {report.get('computed', 0)}), embed {time.time() - arm_started - enhance_s:.1f}s",
                flush=True,
            )
            del enhanced

        del gallery, probes, gallery_embedded
        print(
            f"chunk {index + 1}/{n_chunks}  pairs={take}  elapsed={time.time() - started:.0f}s",
            flush=True,
        )

    labels = np.concatenate(all_labels)
    scores = {name: np.concatenate(values) for name, values in accumulated.items()}

    out: dict = {
        "provenance": {
            "experiment": "S0.3-E (enhancement arms)",
            "embedder": embed_name,
            "recogniser_modified": False,
            "dataset": args.dataset,
            "distance": args.distance if args.dataset == "scface" else None,
            "n_pairs": int(labels.size),
            "seed": args.seed,
            "device": device_report(),
            "cache": cache.stats() if cache else {"enabled": False},
            "binding": args.embedder == "arcface",
            "warning": (
                ""
                if args.embedder == "arcface"
                else "STUB EMBEDDER. These numbers measure plumbing, not recognition, and must never be quoted."
            ),
        },
        "decision_rule": DECISION,
        "arms_unavailable": unavailable,
        "arms": {},
        "enhancement_reports": arm_reports,
    }

    for name in names:
        out["arms"][name] = {
            "tar_at_far_1e3": round(tar_at_far(scores[name], labels), 6),
            "auc": round(auc(scores[name], labels), 6),
            "score_mean_genuine": round(float(scores[name][labels].mean()), 6),
            "score_mean_impostor": round(float(scores[name][~labels].mean()), 6),
            "description": ENHANCEMENT_ARMS[name].description if name in ENHANCEMENT_ARMS else "untransformed baseline",
        }

    out["decisions"] = {}
    for name in arms:
        delta, lo, hi = paired_bootstrap(scores["A"], scores[name], labels, seed=args.seed)
        entry = {
            "vs_A_points": round(delta, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "verdict": verdict(delta, lo, hi),
        }
        if "E0" in scores and name != "E0":
            control_delta, control_lo, control_hi = paired_bootstrap(scores["E0"], scores[name], labels, seed=args.seed)
            entry["vs_E0_control_points"] = round(control_delta, 4)
            entry["vs_E0_ci95"] = [round(control_lo, 4), round(control_hi, 4)]
            entry["beats_interpolation_control"] = bool(control_lo > 0)
        out["decisions"][name] = entry

    suffix = f"_d{args.distance}" if args.dataset == "scface" else ""
    path = Path(args.out) if args.out else RESULTS / f"s0_3E_{args.embedder}_{args.dataset}{suffix}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nS0.3-E  {args.dataset}  embedder={embed_name}  pairs={labels.size:,}")
    print(f"{'arm':<6}{'TAR@FAR=0.1%':>14}{'AUC':>10}   verdict vs A")
    print("-" * 56)
    for name in names:
        row = out["arms"][name]
        decision = out["decisions"].get(name, {})
        tail = ""
        if decision:
            tail = f"   {decision['vs_A_points']:+.2f} pts CI[{decision['ci95'][0]:+.2f},{decision['ci95'][1]:+.2f}] {decision['verdict']}"
        print(f"{name:<6}{row['tar_at_far_1e3'] * 100:>13.2f}%{row['auc']:>10.4f}{tail}")

    for name, report in arm_reports.items():
        for stage in report.get("stages", []):
            if stage.get("warning"):
                print(f"\nWARNING  {name}/{stage['stage']}: {stage['warning']}")
            if stage.get("skipped"):
                print(f"skipped  {name}/{stage['stage']}: {stage['skipped']}")

    if args.embedder != "arcface":
        print("\nNOT BINDING: stub embedder. Plumbing check only.")
    print(f"\nwrote {path.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
