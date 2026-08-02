#!/usr/bin/env python
"""Regenerate, as measured artifacts, every diagnostic number quoted in the docs.

    python backend/scripts/generate_provenance_artifacts.py

WHY THIS EXISTS
---------------
An adversarial audit of docs/MEASUREMENT_RECORD.md found that several diagnostic
figures -- the ones used to justify withdrawing earlier results -- lived only in
code comments and prose. They were correct, but a number whose only source is a
sentence written by the same person who wrote the conclusion is not evidence.

The audit's exact wording: "the doc's own preamble promises 'every number names
the artifact that produced it'; this one names none."

So each figure below is recomputed here and written to
runtime/forensics/provenance_diagnostics.json. Anything that cannot be
regenerated is reported as such rather than quietly retained.

Covers:
  A  TinyFace pairing diagnosis    39.4% Gallery x Gallery, mean 0.4225 vs 0.3268
  B  protocol-pack censoring       99.7% censored at 3,000 impostors
  C  protocol-pack contamination   ~0.2% same-person in unlabelled pools
  D  JPEG estimator sweep          exact-recovery rate across q and size
  E  JPEG confidence separation    compressed vs never-compressed
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.benchmarks.verification import l2n  # noqa: E402
from nexgen_engine.degradation.jpeg import JpegModel, estimate_quality  # noqa: E402
from nexgen_engine.forensics.information import identity_bits  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
OUT = _ROOT / "runtime" / "forensics"
TF = _ROOT / "src_extracted/tinyface/tinyface/Testing_Set"
LFW = _ROOT / "src_extracted/lfw_deepfunneled/lfw-deepfunneled/lfw-deepfunneled"
TF_RE = re.compile(r"^(\d+)_\d+\.jpg$", re.IGNORECASE)
MODEL = "w600k_r50"


def diag_a_tinyface_pairing() -> dict:
    """The diagnosis that withdrew rank-1 = 37.43%."""
    p = CACHE / f"tinyface_labelled__{MODEL}_tta.npz"
    if not p.exists():
        return {"available": False, "reason": f"{p.name} missing"}
    d = np.load(p)
    emb, ids, files = l2n(d["emb"].astype(np.float64)), d["ids"], d["files"]
    gm = {q.name for q in (TF / "Gallery_Match").glob("*.jpg")}
    subset = np.array(["G" if f in gm else "P" for f in files])

    by = defaultdict(list)
    for k, i in enumerate(ids):
        by[int(i)].append(k)
    keys = [k for k, v in by.items() if len(v) >= 2]

    # The withdrawn construction: first two images of each identity by sort order.
    pair01 = [(by[k][0], by[k][1]) for k in keys]
    kinds = [subset[a] + subset[b] for a, b in pair01]
    n_gg = sum(1 for k in kinds if k == "GG")
    s01 = np.array([float(emb[a] @ emb[b]) for a, b in pair01])

    # The official construction: Gallery_Match x Probe only.
    ga, gb = [], []
    for m in by.values():
        G = [i for i in m if subset[i] == "G"]
        P = [i for i in m if subset[i] == "P"]
        for x in G:
            for y in P:
                ga.append(x)
                gb.append(y)
    sgp = np.einsum("ij,ij->i", emb[np.array(ga)], emb[np.array(gb)])

    return {
        "available": True,
        "withdrawn_pairing": {
            "n_pairs": len(pair01),
            "fraction_gallery_x_gallery": round(n_gg / len(kinds), 6),
            "mean_score": round(float(s01.mean()), 6),
        },
        "official_pairing": {
            "n_pairs": int(sgp.size),
            "mean_score": round(float(sgp.mean()), 6),
        },
        "score_inflation": round(float(s01.mean() - sgp.mean()), 6),
    }


def diag_bc_pack_pools() -> dict:
    """Censoring and contamination on unlabelled protocol-pack impostor pools."""
    out = {}
    ident = {"lfw": 5749, "cfp_ff": 500, "cfp_fp": 500, "agedb_30": 568, "calfw": 4025, "cplfw": 3884}
    for ds, n_ids in ident.items():
        p = CACHE / f"{ds}__{MODEL}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        if "issame" not in d.files:
            continue
        emb = l2n((d["orig"] + d["flip"]).astype(np.float64))
        lab = np.asarray(d["issame"], dtype=bool)
        scores = np.sum(emb[0::2] * emb[1::2], axis=1)

        # (B) censoring against the pack's own ~3,000 impostor pairs
        _, censored = identity_bits(scores[lab], scores[~lab])

        # (C) contamination in a large pool built WITHOUT identity labels,
        # i.e. assuming "different pair => different identity"
        rng = np.random.default_rng(0)
        n = 2_000_000
        a, b = rng.integers(0, emb.shape[0], n), rng.integers(0, emb.shape[0], n)
        keep = (a // 2) != (b // 2)
        a, b = a[keep], b[keep]
        pool = np.empty(a.size, dtype=np.float32)
        for s in range(0, a.size, 500_000):
            e = min(s + 500_000, a.size)
            pool[s:e] = np.einsum("ij,ij->i", emb[a[s:e]], emb[b[s:e]]).astype(np.float32)

        out[ds] = {
            "identities": n_ids,
            "n_impostor_pairs_in_pack": int((~lab).sum()),
            "censored_fraction_at_pack_size": round(float(censored.mean()), 6),
            "unlabelled_pool_contamination_above_0.5": round(float((pool > 0.5).mean()), 6),
        }
    return out


def diag_d_jpeg_sweep() -> dict:
    """Exact-recovery rate of the lattice-fit estimator, real faces."""
    if not LFW.is_dir():
        return {"available": False, "reason": "LFW not extracted"}
    faces = []
    for d in sorted(LFW.iterdir())[:8]:
        g = sorted(d.glob("*.jpg"))
        if g:
            faces.append(cv2.imread(str(g[0]), cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0)
    qualities = [20, 35, 50, 65, 80, 95]
    sizes = [31, 48, 112, 250]
    cells, exact = [], 0
    for q in qualities:
        for size in sizes:
            ests = [estimate_quality(JpegModel(q).apply(cv2.resize(f, (size, size))))[0] for f in faces]
            med = int(np.median(ests))
            ok = med == q
            exact += int(ok)
            cells.append({"true_q": q, "size_px": size, "median_estimate": med, "exact": ok})
    return {
        "available": True,
        "n_faces_per_cell": len(faces),
        "qualities": qualities,
        "sizes_px": sizes,
        "cells": cells,
        "exact_recovery": f"{exact}/{len(cells)}",
        "exact_rate": round(exact / len(cells), 6),
    }


def diag_e_jpeg_confidence() -> dict:
    """Confidence separation: genuinely compressed vs never compressed."""
    if not LFW.is_dir():
        return {"available": False, "reason": "LFW not extracted"}
    faces = []
    for d in sorted(LFW.iterdir())[:24]:
        g = sorted(d.glob("*.jpg"))
        if g:
            faces.append(cv2.imread(str(g[0]), cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0)
    comp, raw = [], []
    for f in faces:
        x = cv2.resize(f, (112, 112))
        for q in (25, 35, 50, 75):
            comp.append(estimate_quality(JpegModel(q).apply(x))[1])
        raw.append(estimate_quality(x)[1])
    return {
        "available": True,
        "n_compressed": len(comp),
        "n_never_compressed": len(raw),
        "median_confidence_compressed": round(float(np.median(comp)), 6),
        "median_confidence_never_compressed": round(float(np.median(raw)), 6),
        "gate_threshold": 0.15,
        "accept_rate_compressed": round(float(np.mean(np.array(comp) > 0.15)), 6),
        "false_accept_rate_never_compressed": round(float(np.mean(np.array(raw) > 0.15)), 6),
    }


def diag_f_prefix_estimator_and_channels() -> dict:
    """The PRE-FIX failure and the channel effect -- both previously unsourced.

    An audit noted that section 4's claims "returned q=95 for true q=35 on every
    real face size tested" (defect 1) and "76% -> 100% on green channel"
    (defect 3) had no artifact: only the post-fix sweep was recorded. Both are
    reconstructed here.

    The broken criterion is re-implemented locally rather than restored into the
    shipping module, so the module keeps only the correct method.
    """
    if not LFW.is_dir():
        return {"available": False, "reason": "LFW not extracted"}

    faces_gray, faces_colour = [], []
    for d in sorted(LFW.iterdir())[:25]:
        g = sorted(d.glob("*.jpg"))
        if not g:
            continue
        im = cv2.imread(str(g[0]))
        faces_colour.append(im.astype(np.float64) / 255.0)
        faces_gray.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0)

    def broken_estimate(img: np.ndarray) -> int:
        """The withdrawn criterion: minimum re-compression MSE."""
        cands = list(range(30, 100, 5))
        errs = [float(np.mean((JpegModel(q).apply(img) - img) ** 2)) for q in cands]
        return cands[int(np.argmin(errs))]

    prefix = []
    for size in (31, 48, 112, 250):
        ests = [broken_estimate(cv2.resize(f, (size, size))) for f in faces_gray[:6]]
        med = int(np.median([JpegModel(35).apply(cv2.resize(f, (size, size))) is not None or 0 for f in []] or ests))
        prefix.append({"size_px": size, "true_q": 35, "median_estimate_broken": med})
    # recompute properly: broken estimator applied to genuinely q=35 images
    prefix = []
    for size in (31, 48, 112, 250):
        ests = [broken_estimate(JpegModel(35).apply(cv2.resize(f, (size, size)))) for f in faces_gray[:6]]
        prefix.append({
            "size_px": size, "true_q": 35,
            "median_estimate_broken": int(np.median(ests)),
            "correct": bool(abs(int(np.median(ests)) - 35) <= 5),
        })

    from nexgen_engine.degradation.psf import DegradationParams, apply_forward

    op = lambda q: DegradationParams(blur_sigma=1.6, downsample=8.0, noise_sigma=0.02, jpeg_quality=q)
    mean_ok, green_ok = [], []
    for i, f in enumerate(faces_colour):
        for tq in (25, 35, 50, 75):
            deg = apply_forward(cv2.resize(f, (250, 250)), op(tq), seed=i)
            mean_ok.append(abs(estimate_quality(deg.mean(axis=2))[0] - tq) <= 5)
            green_ok.append(abs(estimate_quality(deg[:, :, 1])[0] - tq) <= 5)

    return {
        "available": True,
        "prefix_estimator_on_true_q35": prefix,
        "channel_handling": {
            "n_cases": len(mean_ok),
            "accuracy_channel_mean": round(float(np.mean(mean_ok)), 6),
            "accuracy_green_channel": round(float(np.mean(green_ok)), 6),
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": (
            "Measured sources for diagnostic figures previously quoted only in "
            "prose and code comments. Raised by adversarial audit of "
            "docs/MEASUREMENT_RECORD.md, 2026-08-02."
        ),
        "model": MODEL,
        "A_tinyface_pairing_diagnosis": diag_a_tinyface_pairing(),
        "BC_protocol_pack_pools": diag_bc_pack_pools(),
        "D_jpeg_estimator_sweep": diag_d_jpeg_sweep(),
        "E_jpeg_confidence_separation": diag_e_jpeg_confidence(),
        "F_prefix_estimator_and_channels": diag_f_prefix_estimator_and_channels(),
    }
    out = OUT / "provenance_diagnostics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    a = payload["A_tinyface_pairing_diagnosis"]
    if a.get("available"):
        print(f"A  withdrawn pairing: {a['withdrawn_pairing']['fraction_gallery_x_gallery']:.1%} GxG, "
              f"mean {a['withdrawn_pairing']['mean_score']:.4f}")
        print(f"   official pairing : mean {a['official_pairing']['mean_score']:.4f}  "
              f"(inflation {a['score_inflation']:+.4f})")
    print("B/C protocol-pack pools:")
    for ds, v in payload["BC_protocol_pack_pools"].items():
        print(f"   {ds:<10} ids={v['identities']:>5}  censored={v['censored_fraction_at_pack_size']:.1%}  "
              f"unlabelled-pool contamination={v['unlabelled_pool_contamination_above_0.5']:.3%}")
    d = payload["D_jpeg_estimator_sweep"]
    if d.get("available"):
        print(f"D  JPEG estimator exact recovery: {d['exact_recovery']} ({d['exact_rate']:.1%})")
    e = payload["E_jpeg_confidence_separation"]
    if e.get("available"):
        print(f"E  confidence: compressed {e['median_confidence_compressed']:.3f} / "
              f"raw {e['median_confidence_never_compressed']:.3f}  "
              f"accept {e['accept_rate_compressed']:.0%} / false-accept {e['false_accept_rate_never_compressed']:.0%}")
    print(f"\nwrote {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
