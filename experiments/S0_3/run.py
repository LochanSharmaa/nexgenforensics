#!/usr/bin/env python
"""S0.3 runner. CPU-complete; only the embedder needs a GPU.

    # CPU smoke test -- proves the whole pipeline works, zero inference
    python experiments/S0_3/run.py --embedder stub --pairs 60

    # The real run (REQUIRES GPU APPROVAL)
    python experiments/S0_3/run.py --embedder arcface --pairs 3000

Decision rule, fixed BEFORE the run so it cannot be adjusted afterwards to suit
the outcome:

    PASS  if  B2 - B1 >= +2.0 points TAR@FAR=0.1%  on BOTH TinyFace and QMUL,
              with a bootstrap 95% CI on the paired difference excluding zero.
    FAIL  if  B2 - B1 <= +0.5 points, or the CI includes zero.
    Between those, INCONCLUSIVE: report and do not proceed on it.

A PASS licenses Stage 3. A FAIL cancels it and redirects the programme to the
evidence layer. Registering the rule in code is the point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "backend"))

from arms import ARMS, arm_C_score, stub_embedder  # noqa: E402

from nexgen_engine.forensics.evidence import bootstrap_ci  # noqa: E402

RESULTS = _HERE / "results"
DECISION = {"pass_threshold": 2.0, "fail_threshold": 0.5, "metric": "TAR@FAR=0.1% (points)"}


def tar_at_far(scores: np.ndarray, labels: np.ndarray, far: float = 1e-3) -> float:
    imp = scores[~labels]
    if imp.size == 0:
        return float("nan")
    return float((scores[labels] > np.quantile(imp, 1.0 - far)).mean())


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_embedder(kind: str):
    if kind == "stub":
        return stub_embedder, "stub (CPU, not a recogniser)"
    if kind == "arcface":
        raise SystemExit(
            "The arcface embedder requires GPU inference and has not been approved.\n"
            "See GPU_EXECUTION_REQUEST.md. Run with --embedder stub to validate the\n"
            "pipeline on CPU first."
        )
    raise SystemExit(f"unknown embedder: {kind}")


def synthetic_pairs(n: int, size: int = 96, seed: int = 0):
    """Deterministic synthetic HR/LR pairs for the CPU smoke test.

    NOT a benchmark. Its only purpose is to exercise every arm, metric and
    bootstrap so the GPU run cannot fail on plumbing.
    """
    rng = np.random.default_rng(seed)
    from nexgen_engine.degradation.psf import DegradationParams, apply_forward

    identities = [rng.normal(0.5, 0.15, (size, size)).clip(0, 1) for _ in range(max(n // 2, 2))]
    gal, prb, lab = [], [], []
    for k in range(n):
        i = k % len(identities)
        base = identities[i]
        gal.append(np.clip(base + rng.normal(0, 0.01, base.shape), 0, 1))
        same = k % 2 == 0
        src = base if same else identities[(i + 1) % len(identities)]
        prb.append(
            apply_forward(
                np.clip(src + rng.normal(0, 0.01, src.shape), 0, 1),
                DegradationParams(blur_sigma=1.5, downsample=4.0, noise_sigma=0.02, jpeg_quality=40),
                seed=k,
            )
        )
        lab.append(same)
    return gal, prb, np.array(lab, dtype=bool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="stub", choices=["stub", "arcface"])
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    embed, embed_name = load_embedder(args.embedder)
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.dataset != "synthetic":
        raise SystemExit(
            f"dataset '{args.dataset}' requires embedding real imagery (GPU). "
            "Only --dataset synthetic runs on CPU."
        )
    gal, prb, labels = synthetic_pairs(args.pairs)

    out: dict = {
        "provenance": {
            "embedder": embed_name,
            "dataset": args.dataset,
            "n_pairs": int(labels.size),
            "gpu_used": False,
            "warning": (
                "Synthetic data with a stub embedder. These numbers measure PLUMBING, "
                "not recognition, and must never be quoted as a result."
            ),
        },
        "decision_rule": DECISION,
        "arms": {},
    }

    for name, fn in ARMS.items():
        g_t, p_t, reports = [], [], []
        for g, p in zip(gal, prb):
            a, b, rep = fn(g, p)
            g_t.append(a)
            p_t.append(b)
            reports.append(rep)
        eg, ep = embed(g_t), embed(p_t)
        scores = np.sum(eg * ep, axis=1)
        out["arms"][name] = {
            "tar_at_far_1e3": round(tar_at_far(scores, labels), 6),
            "auc": round(auc(scores, labels), 6),
            "score_mean_genuine": round(float(scores[labels].mean()), 6),
            "score_mean_impostor": round(float(scores[~labels].mean()), 6),
            "example_report": reports[0],
        }

    c_scores = np.array([arm_C_score(g, p)[0] for g, p in zip(gal, prb)])
    out["arms"]["C"] = {
        "tar_at_far_1e3": round(tar_at_far(c_scores, labels), 6),
        "auc": round(auc(c_scores, labels), 6),
        "note": "pixel-space log-likelihood; rank metrics only, scale not comparable",
    }

    # The decision: paired bootstrap on B2 - B1.
    def paired(idx, arm_x, arm_y):
        return tar_at_far(arm_x[idx], labels[idx]) - tar_at_far(arm_y[idx], labels[idx])

    rebuilt = {}
    for name in ("B1", "B2"):
        fn = ARMS[name]
        gg, pp = [], []
        for g, p in zip(gal, prb):
            a, b, _ = fn(g, p)
            gg.append(a)
            pp.append(b)
        rebuilt[name] = np.sum(embed(gg) * embed(pp), axis=1)

    delta = (tar_at_far(rebuilt["B2"], labels) - tar_at_far(rebuilt["B1"], labels)) * 100
    lo, hi = bootstrap_ci(
        np.arange(labels.size),
        lambda idx: paired(idx, rebuilt["B2"], rebuilt["B1"]) * 100,
        n_boot=1000,
    )
    verdict = (
        "PASS" if delta >= DECISION["pass_threshold"] and lo > 0
        else "FAIL" if delta <= DECISION["fail_threshold"] or lo <= 0 <= hi
        else "INCONCLUSIVE"
    )
    out["decision"] = {
        "b2_minus_b1_points": round(delta, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "verdict": verdict,
        "binding": args.embedder == "arcface",
        "note": "Not binding: stub embedder." if args.embedder != "arcface" else "",
    }

    path = Path(args.out) if args.out else RESULTS / f"s0_3_{args.embedder}_{args.dataset}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nS0.3  embedder={embed_name}  pairs={labels.size}")
    print(f"{'arm':<6}{'TAR@FAR=0.1%':>14}{'AUC':>10}")
    print("-" * 30)
    for k, v in out["arms"].items():
        print(f"{k:<6}{v['tar_at_far_1e3']*100:>13.2f}%{v['auc']:>10.4f}")
    print(f"\nB2 - B1 = {delta:+.2f} points  CI95 [{lo:+.2f}, {hi:+.2f}]  -> {verdict}")
    if args.embedder != "arcface":
        print("NOT BINDING: stub embedder on synthetic data. Plumbing check only.")
    print(f"\nwrote {path.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
