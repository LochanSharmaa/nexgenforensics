#!/usr/bin/env python
"""Full forensic evaluation of the CURRENT system, from cached embeddings only.

    python backend/scripts/evaluate_baseline.py

No model load, no GPU, no new inference. Establishes the baseline that every
future architecture must beat, measured the way a forensic system should be
measured rather than the way a benchmark leaderboard is.

Produces per dataset:
    ROC / DET curve points        classical discrimination
    TAR at four FAR operating points, with bootstrap 95% CIs
    Cllr / Cllr_min / Cllr_cal    calibration, on HELD-OUT folds
    ECE                            calibration error of the reported LR
    Tippett rates                  misleading-evidence rates, both directions
    failure mining                 worst genuine and worst impostor comparisons
    CMC + open-set                 TinyFace only, where identity labels exist

Datasets covered are exactly those with cached embeddings. QMUL, IJB-B and IJB-C
have images but no cached embeddings, so they are reported as BLOCKED-ON-GPU
rather than silently omitted.
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
from nexgen_engine.forensics import cllr_report, cross_validated_log10_lr, tippett  # noqa: E402
from nexgen_engine.forensics.evidence import bootstrap_ci  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
TINYFACE = _ROOT / "src_extracted/tinyface/tinyface/Testing_Set"
OUT = _ROOT / "runtime" / "forensics"
DOCS = _ROOT / "docs"
MODEL = "w600k_r50"
PAIRED = ["lfw", "cfp_ff", "agedb_30", "cfp_fp", "calfw", "cplfw"]
FARS = [1e-1, 1e-2, 1e-3, 1e-4]
ID_RE = re.compile(r"^(\d+)_\d+\.jpg$", re.IGNORECASE)

BLOCKED = {
    "qmul_survface": "images present, embeddings not cached",
    "ijb_b": "archive present, not extracted, embeddings not cached",
    "ijb_c": "archive present, not extracted, embeddings not cached",
    "tinyface_distractors": "153,428 images present, embeddings not cached",
}


def load_pairs(dataset: str):
    p = CACHE / f"{dataset}__{MODEL}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    if "issame" not in d.files:
        return None
    emb = l2n((d["orig"] + d["flip"]).astype(np.float64))
    return np.sum(emb[0::2] * emb[1::2], axis=1), np.asarray(d["issame"], dtype=bool)


def load_tinyface():
    p = CACHE / f"tinyface_labelled__{MODEL}.npz"
    if not p.exists() or not TINYFACE.is_dir():
        return None
    files = sorted(
        f for sub in ("Gallery_Match", "Probe")
        if (TINYFACE / sub).is_dir()
        for f in (TINYFACE / sub).glob("*.jpg") if ID_RE.match(f.name)
    )
    emb = l2n(np.load(p)["emb"].astype(np.float64))
    if emb.shape[0] != len(files):
        return None
    ids = np.array([int(ID_RE.match(f.name).group(1)) for f in files])
    return emb, ids


def roc_det(scores, labels, n_points: int = 200):
    """ROC and DET points. DET plots FNMR vs FMR, which is the forensic view."""
    g, i = np.sort(scores[labels]), np.sort(scores[~labels])
    lo, hi = float(scores.min()), float(scores.max())
    thr = np.linspace(lo, hi, n_points)
    fmr = np.array([(i >= t).mean() for t in thr])
    tar = np.array([(g >= t).mean() for t in thr])
    return {
        "threshold": [round(float(t), 6) for t in thr],
        "fmr": [round(float(v), 8) for v in fmr],
        "tar": [round(float(v), 8) for v in tar],
        "fnmr": [round(float(1 - v), 8) for v in tar],
    }


def tar_at(scores, labels, far):
    i = np.sort(scores[~labels])
    if i.size == 0:
        return float("nan")
    thr = float(np.quantile(i, 1.0 - far))
    return float((scores[labels] > thr).mean())


def ece(log10_lr, labels, n_bins: int = 15):
    """Expected calibration error of the posterior implied by the LR at a
    neutral prior. Complements Cllr, which is a proper score but not an
    interpretable probability gap."""
    p = 1.0 / (1.0 + np.power(10.0, -np.clip(log10_lr, -300, 300)))
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        total += sel.mean() * abs(p[sel].mean() - labels[sel].mean())
    return float(total)


def cmc_openset(emb, ids, n_probes: int = 1500, seed: int = 0):
    """Closed-set CMC and open-set TPIR from a labelled gallery.

    Gallery = one capture per identity; probes = another capture of identities
    that ARE enrolled, plus probes from identities deliberately held out so the
    unenrolled case is measured rather than assumed.
    """
    rng = np.random.default_rng(seed)
    by_id = defaultdict(list)
    for k, i in enumerate(ids):
        by_id[int(i)].append(k)
    multi = {k: v for k, v in by_id.items() if len(v) >= 2}
    keys = sorted(multi)
    if len(keys) < 50:
        return None

    # Hold out 30% of identities entirely: these probes have no true mate.
    rng.shuffle(keys)
    cut = int(0.7 * len(keys))
    enrolled, held_out = keys[:cut], keys[cut:]

    gal_idx, gal_id = [], []
    for k in enrolled:
        gal_idx.append(multi[k][0])
        gal_id.append(k)
    gal = emb[np.array(gal_idx)]
    gal_id = np.array(gal_id)

    probes, truth = [], []
    for k in enrolled:
        probes.append(multi[k][1])
        truth.append(k)
    for k in held_out:
        probes.append(multi[k][1])
        truth.append(-1)  # not enrolled
    probes, truth = np.array(probes), np.array(truth)
    if probes.size > n_probes:
        sel = rng.choice(probes.size, n_probes, replace=False)
        probes, truth = probes[sel], truth[sel]

    sims = emb[probes] @ gal.T
    order = np.argsort(-sims, axis=1)
    mated = truth >= 0

    ranks = []
    for r in range(sims.shape[0]):
        if not mated[r]:
            continue
        pos = np.flatnonzero(gal_id[order[r]] == truth[r])
        ranks.append(int(pos[0]) + 1 if pos.size else 10**9)
    ranks = np.array(ranks)
    cmc = {f"rank_{k}": float((ranks <= k).mean()) for k in (1, 5, 10, 20, 50)}

    # Open set: threshold on the top score; report TPIR at fixed FPIR.
    top = sims.max(axis=1)
    top_id = gal_id[order[:, 0]]
    openset = {}
    if (~mated).sum() > 0:
        for fpir in (0.01, 0.1):
            thr = float(np.quantile(top[~mated], 1.0 - fpir))
            correct = mated & (top > thr) & (top_id == truth)
            openset[f"tpir_at_fpir_{fpir}"] = float(correct.sum() / max(mated.sum(), 1))
    return {
        "gallery_size": int(gal.shape[0]),
        "n_probes": int(probes.size),
        "n_mated": int(mated.sum()),
        "n_nonmated": int((~mated).sum()),
        "cmc": cmc,
        "open_set": openset,
    }


def failure_cases(scores, labels, k: int = 10):
    g_idx = np.flatnonzero(labels)
    i_idx = np.flatnonzero(~labels)
    worst_g = g_idx[np.argsort(scores[g_idx])[:k]]
    worst_i = i_idx[np.argsort(-scores[i_idx])[:k]]
    return {
        "hardest_genuine": [{"pair": int(j), "score": round(float(scores[j]), 6)} for j in worst_g],
        "hardest_impostor": [{"pair": int(j), "score": round(float(scores[j]), 6)} for j in worst_i],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    results, table = {}, []

    work = [(d, load_pairs(d)) for d in PAIRED]
    tf = load_tinyface()
    if tf is not None:
        emb, ids = tf
        by_id = defaultdict(list)
        for k, i in enumerate(ids):
            by_id[int(i)].append(k)
        rng = np.random.default_rng(0)
        ga, gb, ia, ib = [], [], [], []
        keys = [k for k, v in by_id.items() if len(v) >= 2]
        for _ in range(3000):
            k = keys[rng.integers(0, len(keys))]
            a, b = rng.choice(by_id[k], 2, replace=False)
            ga.append(a); gb.append(b)
            x, y = keys[rng.integers(0, len(keys))], keys[rng.integers(0, len(keys))]
            while x == y:
                y = keys[rng.integers(0, len(keys))]
            ia.append(by_id[x][0]); ib.append(by_id[y][0])
        s = np.empty(6000); lab = np.zeros(6000, dtype=bool)
        s[0::2] = np.sum(emb[ga] * emb[gb], axis=1); lab[0::2] = True
        s[1::2] = np.sum(emb[ia] * emb[ib], axis=1)
        work.append(("tinyface", (s, lab)))

    for name, got in work:
        if got is None:
            print(f"  {name}: no cached embeddings -- skipped")
            continue
        scores, labels = got
        held = cross_validated_log10_lr(scores, labels, n_folds=10)
        cv = cllr_report(held, labels, scores=scores)
        tip = tippett(held, labels)

        tars = {}
        for far in FARS:
            point = tar_at(scores, labels, far)
            idx = np.arange(scores.size)
            lo, hi = bootstrap_ci(
                idx, lambda b, f=far: tar_at(scores[b], labels[b], f), n_boot=400
            )
            tars[f"tar_at_far_{far:g}"] = {
                "value": round(point, 6),
                "ci95": [round(lo, 6), round(hi, 6)],
            }

        results[name] = {
            "n_pairs": int(scores.size),
            "roc_det": roc_det(scores, labels),
            "tar": tars,
            "cllr": cv.as_dict(),
            "ece": round(ece(held, labels), 6),
            "tippett": {
                "rate_misleading_same_source": tip.rate_misleading_same_source,
                "rate_misleading_different_source": tip.rate_misleading_different_source,
            },
            "failures": failure_cases(scores, labels),
        }
        table.append(
            {
                "dataset": name,
                "cllr": cv.cllr,
                "cllr_min": cv.cllr_min,
                "cllr_cal": cv.cllr_cal,
                "ece": results[name]["ece"],
                "tar3": tars["tar_at_far_0.001"]["value"],
                "tar3_ci": tars["tar_at_far_0.001"]["ci95"],
                "mis_ss": tip.rate_misleading_same_source,
            }
        )

    if tf is not None:
        results["tinyface"]["identification"] = cmc_openset(*tf)

    results["_blocked_on_gpu"] = BLOCKED
    results["_provenance"] = {
        "model": MODEL,
        "flip_tta": "yes for protocol packs; NO for tinyface (cache lacks orig/flip)",
        "calibration": "logistic, 10-fold, fitted on 9 and applied to the held-out fold",
        "inference_run": False,
    }
    (OUT / "baseline_evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    hdr = f"{'dataset':<12}{'Cllr':>8}{'Cllr_min':>10}{'Cllr_cal':>10}{'ECE':>8}{'TAR@FAR=0.1%':>15}{'mislead SS':>12}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in table:
        ci = f"[{r['tar3_ci'][0]*100:.1f},{r['tar3_ci'][1]*100:.1f}]"
        print(
            f"{r['dataset']:<12}{r['cllr']:>8.4f}{r['cllr_min']:>10.4f}{r['cllr_cal']:>10.4f}"
            f"{r['ece']:>8.4f}{r['tar3']*100:>8.2f}% {ci:>6}{r['mis_ss']*100:>11.2f}%"
        )

    if tf is not None and results["tinyface"].get("identification"):
        idn = results["tinyface"]["identification"]
        print(f"\nTinyFace identification  gallery={idn['gallery_size']:,}  "
              f"mated={idn['n_mated']}  non-mated={idn['n_nonmated']}")
        print("  CMC     " + "  ".join(f"{k}={v*100:.2f}%" for k, v in idn["cmc"].items()))
        if idn["open_set"]:
            print("  OpenSet " + "  ".join(f"{k}={v*100:.2f}%" for k, v in idn["open_set"].items()))

    print("\nBLOCKED ON GPU (images present, embeddings absent):")
    for k, v in BLOCKED.items():
        print(f"  {k:<24} {v}")
    print(f"\nwrote {(OUT / 'baseline_evaluation.json').relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
