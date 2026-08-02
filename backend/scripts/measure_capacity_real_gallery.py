#!/usr/bin/env python
"""G1 follow-up -- capacity and open-set 1:N against the REAL TinyFace gallery.

    python backend/scripts/measure_capacity_real_gallery.py

This is the measurement the whole capacity framework was built for, and until now
it could not be run. Prior numbers used a 1,794-entry proxy gallery assembled
from the labelled set; every one of them was a placeholder.

WHY THE DISTRACTOR SET IS THE RIGHT REFERENCE POPULATION
--------------------------------------------------------
TinyFace's Gallery_Distractor holds 153,428 images of people who do NOT appear in
the labelled probe/gallery split. A probe-vs-distractor comparison is therefore a
genuine impostor comparison **by construction of the protocol**, not by an
assumption about identity labels.

That distinction is the entire lesson of this project's two failed capacity
attempts (see nexgen_engine/forensics/population.py). The first saturated at
11.55 bits because 3,000 impostor pairs cannot resolve a deeper tail. The second
used a 20M pool built by randomly pairing images from different pairs, which
silently contained same-person pairs at a rate set by corpus identity count --
0.2% for AgeDB, twice the FAR being measured.

Here neither failure applies: the pool is large AND disjoint by protocol.

Outputs:
    runtime/forensics/capacity_real_gallery.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.benchmarks.verification import l2n  # noqa: E402
from nexgen_engine.forensics import capacity_from_pools, cllr_report, cross_validated_log10_lr  # noqa: E402
from nexgen_engine.forensics.evidence import bootstrap_ci  # noqa: E402
from nexgen_engine.forensics.population import ReferencePopulation, audit_population  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
OUT = _ROOT / "runtime" / "forensics"


def load(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = CACHE / f"tinyface_labelled__{model}_tta.npz"
    dis = CACHE / f"tinyface_distractors__{model}.npz"
    for p in (lab, dis):
        if not p.exists():
            raise SystemExit(f"missing {p.name} -- run embed_tinyface_gallery.py first")
    L, D = np.load(lab), np.load(dis)
    return l2n(L["emb"].astype(np.float64)), L["ids"], l2n(D["emb"].astype(np.float64))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--impostor-samples", type=int, default=50_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    emb, ids, dis = load(args.model)
    rng = np.random.default_rng(args.seed)
    print(f"labelled {emb.shape[0]:,} images / {len(set(ids.tolist())):,} identities")
    print(f"distractors {dis.shape[0]:,} images")

    # -- genuine pool: every within-identity pair ----------------------------
    by_id: dict[int, list[int]] = defaultdict(list)
    for k, i in enumerate(ids):
        by_id[int(i)].append(k)
    ga, gb = [], []
    for members in by_id.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                ga.append(members[x])
                gb.append(members[y])
    ga, gb = np.array(ga), np.array(gb)
    genuine = np.einsum("ij,ij->i", emb[ga], emb[gb])
    print(f"genuine comparisons: {genuine.size:,}")

    # -- impostor pool: probe x distractor, disjoint BY PROTOCOL --------------
    n = args.impostor_samples
    a = rng.integers(0, emb.shape[0], size=n)
    b = rng.integers(0, dis.shape[0], size=n)
    impostor = np.empty(n, dtype=np.float32)
    for s in range(0, n, 1_000_000):
        e = min(s + 1_000_000, n)
        impostor[s:e] = np.einsum("ij,ij->i", emb[a[s:e]], dis[b[s:e]]).astype(np.float32)
    print(f"impostor comparisons: {impostor.size:,}")

    audit = audit_population(impostor, identity_labels_present=True, target_far=1e-5)
    print(f"population audit: {audit.verdict} -- {audit.detail}")

    cap = capacity_from_pools("tinyface_real_gallery", genuine, impostor)
    print("\nCAPACITY (real 153k-image reference population)")
    print(f"  censoring ceiling      {cap.censoring_ceiling_bits:.2f} bits")
    print(f"  censored fraction      {cap.censored_fraction:.4%}")
    print(f"  bits  p10/p20/med/mean {cap.bits_p10:.2f} / {cap.bits_p20:.2f} / "
          f"{cap.bits_median:.2f} / {cap.bits_mean:.2f}")
    print(f"  supportable gallery @50%/80%/90% rank-1: "
          f"{cap.gallery_at_rank1_50:,.0f} / {cap.gallery_at_rank1_80:,.0f} / "
          f"{cap.gallery_at_rank1_90:,.0f}")

    # bootstrap CI on the headline gallery figure
    from nexgen_engine.forensics.information import gallery_for_rank1, identity_bits

    bits, _ = identity_bits(genuine, impostor)
    lo, hi = bootstrap_ci(bits, lambda b: gallery_for_rank1(b, 0.80), n_boot=400)
    print(f"  gallery@80% 95% CI: [{lo:,.0f}, {hi:,.0f}]")

    # -- TAR@FAR against the real population ---------------------------------
    tar = {}
    for far in (1e-2, 1e-3, 1e-4, 1e-5):
        thr = float(np.quantile(impostor, 1.0 - far))
        tar[f"tar_at_far_{far:g}"] = float((genuine > thr).mean())
    print("\nTAR against the real reference population")
    for k, v in tar.items():
        print(f"  {k:<22} {v * 100:6.2f}%")

    # -- open-set 1:N identification, real distractor gallery ----------------
    keys = [k for k, v in by_id.items() if len(v) >= 2]
    rng2 = np.random.default_rng(args.seed + 1)
    rng2.shuffle(keys)
    cut = int(0.7 * len(keys))
    enrolled, held = keys[:cut], keys[cut:]

    gal_emb = np.concatenate([emb[[by_id[k][0] for k in enrolled]], dis])
    gal_id = np.concatenate([np.array(enrolled), np.full(dis.shape[0], -1)])
    probes = np.array([by_id[k][1] for k in enrolled] + [by_id[k][1] for k in held])
    truth = np.array(enrolled + [-1] * len(held))
    sel = rng2.choice(probes.size, min(1500, probes.size), replace=False)
    probes, truth = probes[sel], truth[sel]

    print(f"\nOPEN-SET 1:N  gallery={gal_emb.shape[0]:,} "
          f"({len(enrolled):,} enrolled + {dis.shape[0]:,} distractors)")
    sims = emb[probes] @ gal_emb.T
    mated = truth >= 0
    top_idx = np.argmax(sims, axis=1)
    top = sims[np.arange(sims.shape[0]), top_idx]
    top_id = gal_id[top_idx]

    ranks = []
    for r in np.flatnonzero(mated):
        row = sims[r]
        mate = np.flatnonzero(gal_id == truth[r])
        ranks.append(int((row > row[mate[0]]).sum()) + 1)
    ranks = np.array(ranks)
    cmc = {f"rank_{k}": float((ranks <= k).mean()) for k in (1, 5, 10, 20, 50, 100)}
    openset = {}
    for fpir in (0.01, 0.1):
        thr = float(np.quantile(top[~mated], 1.0 - fpir))
        openset[f"tpir_at_fpir_{fpir}"] = float(
            ((mated) & (top > thr) & (top_id == truth)).sum() / max(mated.sum(), 1)
        )
    print("  CMC     " + "  ".join(f"{k}={v*100:.2f}%" for k, v in cmc.items()))
    print("  OpenSet " + "  ".join(f"{k}={v*100:.2f}%" for k, v in openset.items()))

    # -- calibration on the real population ----------------------------------
    m = min(genuine.size, 200_000)
    gs = rng2.choice(genuine.size, m, replace=False)
    isamp = rng2.choice(impostor.size, m, replace=False)
    scores = np.empty(2 * m)
    labels = np.zeros(2 * m, dtype=bool)
    scores[0::2] = genuine[gs]
    labels[0::2] = True
    scores[1::2] = impostor[isamp]
    held_lr = cross_validated_log10_lr(scores, labels, n_folds=10)
    cv = cllr_report(held_lr, labels, scores=scores)
    print(f"\nCALIBRATION (real population)  Cllr={cv.cllr:.4f}  "
          f"Cllr_min={cv.cllr_min:.4f}  Cllr_cal={cv.cllr_cal:.4f}  "
          f"({cv.cllr_cal / max(cv.cllr, 1e-9) * 100:.1f}% recoverable)")

    payload = {
        "model": args.model,
        "population": {
            "name": "tinyface_gallery_distractor",
            "n_distractor_images": int(dis.shape[0]),
            "n_impostor_comparisons": int(impostor.size),
            "disjointness": "by protocol -- distractors are identities absent from the labelled split",
            "audit": audit.as_dict(),
        },
        "capacity": cap.as_dict(),
        "gallery_at_rank1_80_ci95": [lo, hi],
        "tar_vs_real_population": tar,
        "identification": {
            "gallery_size": int(gal_emb.shape[0]),
            "n_enrolled": len(enrolled),
            "n_probes": int(probes.size),
            "n_mated": int(mated.sum()),
            "n_nonmated": int((~mated).sum()),
            "cmc": cmc,
            "open_set": openset,
        },
        "calibration": cv.as_dict(),
        "flip_tta": True,
    }
    out = OUT / "capacity_real_gallery.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
