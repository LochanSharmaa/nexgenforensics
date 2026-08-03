#!/usr/bin/env python
"""S0.4 -- how much capture-condition signal leaks into the identity embedding?

    python backend/scripts/measure_condition_leakage.py

THE QUANTITY
------------
    leakage = E[sim | same condition, different identity]
            - E[sim | different condition, different identity]

If the embedding carried identity only, both expectations would be equal and
leakage would be zero. A positive value means two DIFFERENT people photographed
under the same conditions look more alike to the model than two different people
under different conditions -- i.e. the embedding partly encodes the camera, not
the face. This is precisely the artefact this project's QMUL overlap audit
caught by hand in July (a QMUL face resembled an arbitrary *different* QMUL
person at 0.600, more than anything in TinyFace at 0.522), and it is the
quantity the training objective's adversarial term (lambda_5 in
IMPLEMENTATION-PLAN.md section 2) exists to drive to zero.

WHY QMUL CAN MEASURE IT CLEANLY
-------------------------------
QMUL filenames are `<id>_cam<n>_<m>.jpg`, so every embedding in the cached
identification set carries a camera label. Same-camera/different-identity and
different-camera/different-identity pools differ ONLY in the condition variable,
within one corpus, one collection campaign, one resolution regime. No
cross-corpus confound.

Also measured per camera pair, because "camera" bundles optics, mounting site,
lighting and compression -- if leakage concentrates in specific camera pairs
that is diagnostic in itself.

This number is the BASELINE for the current model. Its role is to be compared
against the same measurement on any future backbone or fine-tune: a training run
that improves TAR while inflating leakage is learning the camera.

Output: runtime/forensics/condition_leakage.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.benchmarks.verification import l2n  # noqa: E402
from nexgen_engine.forensics.evidence import bootstrap_ci  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
OUT = _ROOT / "runtime" / "forensics"
NAME_RE = re.compile(r"^(\d+)_cam(\d+)_", re.IGNORECASE)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="w600k_r50")
    cli = ap.parse_args()
    ap_path = CACHE / f"qmul_ident__{cli.model}.npz"
    if not ap_path.exists():
        raise SystemExit(f"missing {ap_path.name} -- run embed_qmul_ident.py first")
    OUT.mkdir(parents=True, exist_ok=True)

    d = np.load(ap_path)
    emb = l2n(d["emb"].astype(np.float64))
    files = d["files"]
    rng = np.random.default_rng(0)

    ids = np.empty(len(files), dtype=np.int64)
    cams = np.empty(len(files), dtype=np.int64)
    keep = np.zeros(len(files), dtype=bool)
    for k, f in enumerate(files):
        m = NAME_RE.match(str(f))
        if m:
            ids[k], cams[k], keep[k] = int(m.group(1)), int(m.group(2)), True
    emb, ids, cams = emb[keep], ids[keep], cams[keep]
    cam_values = sorted(set(cams.tolist()))
    print(f"{emb.shape[0]:,} embeddings, {len(set(ids.tolist())):,} identities, cameras {cam_values}")

    # -- sample cross-identity pairs, tagged same-cam / diff-cam --------------
    n = 5_000_000
    a = rng.integers(0, emb.shape[0], size=n)
    b = rng.integers(0, emb.shape[0], size=n)
    diff_id = ids[a] != ids[b]
    a, b = a[diff_id], b[diff_id]
    sims = np.empty(a.size, dtype=np.float32)
    for s in range(0, a.size, 500_000):
        e = min(s + 500_000, a.size)
        sims[s:e] = np.einsum("ij,ij->i", emb[a[s:e]], emb[b[s:e]]).astype(np.float32)
    same_cam = cams[a] == cams[b]

    mean_same = float(sims[same_cam].mean())
    mean_diff = float(sims[~same_cam].mean())
    leakage = mean_same - mean_diff

    # CI by bootstrap over a subsample (the pools are huge; 200k is plenty).
    sub_s = rng.choice(np.flatnonzero(same_cam), 200_000, replace=False)
    sub_d = rng.choice(np.flatnonzero(~same_cam), 200_000, replace=False)
    ss, sd = sims[sub_s], sims[sub_d]
    lo, hi = bootstrap_ci(
        np.arange(200_000),
        lambda idx: float(ss[idx].mean() - sd[idx].mean()),
        n_boot=1000,
    )

    # Leakage in decision units: how far up the impostor tail does the
    # same-camera distribution sit? Report the FAR that a same-camera impostor
    # pool produces at the threshold fixed for FAR=0.1% on the diff-camera pool.
    thr = float(np.quantile(sims[~same_cam], 1.0 - 1e-3))
    far_same_at_thr = float((sims[same_cam] > thr).mean())

    # -- per camera pair ------------------------------------------------------
    per_pair = {}
    for ci in cam_values:
        for cj in cam_values:
            if ci > cj:
                continue
            sel = ((cams[a] == ci) & (cams[b] == cj)) | ((cams[a] == cj) & (cams[b] == ci))
            if sel.sum() < 5_000:
                continue
            per_pair[f"cam{ci}-cam{cj}"] = {
                "n": int(sel.sum()),
                "mean_sim": round(float(sims[sel].mean()), 6),
            }

    print(f"\nCONDITION LEAKAGE (cross-identity similarity)")
    print(f"  same camera : {mean_same:+.4f}")
    print(f"  diff camera : {mean_diff:+.4f}")
    print(f"  leakage     : {leakage:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"\n  At the diff-camera FAR=0.1% threshold ({thr:.4f}), same-camera")
    print(f"  impostors alarm at {far_same_at_thr:.4%} -- {far_same_at_thr / 1e-3:.1f}x the nominal rate.")
    print(f"\n  per camera pair (cross-identity mean similarity):")
    for k, v in sorted(per_pair.items(), key=lambda kv: -kv[1]["mean_sim"]):
        print(f"    {k:<12} n={v['n']:>9,}  {v['mean_sim']:+.4f}")

    payload = {
        "model": cli.model,
        "corpus": "qmul_ident (gallery+mated+unmated)",
        "definition": "E[sim | same cam, diff id] - E[sim | diff cam, diff id]",
        "n_cross_identity_pairs": int(a.size),
        "mean_same_camera": mean_same,
        "mean_diff_camera": mean_diff,
        "leakage": leakage,
        "leakage_ci95": [lo, hi],
        "far_amplification": {
            "diff_camera_far_target": 1e-3,
            "threshold": thr,
            "same_camera_far_at_threshold": far_same_at_thr,
            "amplification_x": far_same_at_thr / 1e-3,
        },
        "per_camera_pair": per_pair,
        "role": (
            "Baseline for the lambda_5 leakage penalty. Compare any future "
            "backbone/fine-tune against this number; TAR gains that inflate it "
            "are learning the camera, not the face."
        ),
    }
    out = OUT / f"condition_leakage__{cli.model}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
