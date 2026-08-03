#!/usr/bin/env python
"""Capacity and open-set 1:N on OFFICIAL protocol splits. Supersedes prior numbers.

    python backend/scripts/measure_capacity_official.py --dataset tinyface
    python backend/scripts/measure_capacity_official.py --dataset qmul

WHY THIS SCRIPT REPLACES measure_capacity_real_gallery.py
---------------------------------------------------------
That script built genuine pairs as "the first two images of each identity, by
sort order". 39.4% of those turned out to be Gallery_Match x Gallery_Match --
near-duplicate frames from the same surveillance track. Measured effect:

    ([0],[1]) pairs            mean score 0.4225   <- what it used
    Gallery_Match x Probe      mean score 0.3268   <- the official pairing

It reported rank-1 = 37.43%. That number is inflated and is withdrawn. Both
corpora ship an official split precisely to prevent this, and this script uses
nothing else.

PROTOCOLS
---------
TinyFace   gallery  = Gallery_Match
           probes   = Probe
           non-mates= Gallery_Distractor (153,428, identities absent from the
                      labelled split -- disjoint BY PROTOCOL, not by assumption)

QMUL       gallery       = Face_Identification_Test_Set/gallery
           mated probes  = .../mated_probe
           unmated probes= .../unmated_probe   (true open-set non-mates)

Both give an identity-disjoint reference population by construction, which is
the property two earlier attempts failed to achieve (saturation, then 0.2%
contamination against a 0.1% FAR). See forensics/population.py.
"""

from __future__ import annotations

import argparse
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
from nexgen_engine.forensics import capacity_from_pools, cllr_report, cross_validated_log10_lr  # noqa: E402
from nexgen_engine.forensics.evidence import bootstrap_ci  # noqa: E402
from nexgen_engine.forensics.information import gallery_for_rank1, identity_bits  # noqa: E402
from nexgen_engine.forensics.population import audit_population  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
OUT = _ROOT / "runtime" / "forensics"
TF = _ROOT / "src_extracted/tinyface/tinyface/Testing_Set"
QMUL_ID = Path("C:/Users/hello/Downloads/QMUL-SurvFace-v1/QMUL-SurvFace/Face_Identification_Test_Set")
TF_RE = re.compile(r"^(\d+)_\d+\.jpg$", re.IGNORECASE)
QM_RE = re.compile(r"^(\d+)_cam", re.IGNORECASE)


def load_tinyface(model: str):
    lab = np.load(CACHE / f"tinyface_labelled__{model}_tta.npz")
    dis = np.load(CACHE / f"tinyface_distractors__{model}.npz")
    emb = l2n(lab["emb"].astype(np.float64))
    ids = lab["ids"]
    files = lab["files"]
    gm = {p.name for p in (TF / "Gallery_Match").glob("*.jpg")}
    is_gallery = np.array([f in gm for f in files])
    return {
        "gal_emb": emb[is_gallery], "gal_ids": ids[is_gallery],
        "prb_emb": emb[~is_gallery], "prb_ids": ids[~is_gallery],
        "non_emb": l2n(dis["emb"].astype(np.float64)),
        # TinyFace distractors belong in the GALLERY -- that is what the protocol
        # means by Gallery_Distractor. TinyFace ships no unmated probes, so it
        # cannot measure open-set rejection at all; it measures rank-1 against a
        # large gallery instead.
        "non_role": "gallery_filler",
    }


def load_qmul(model: str):
    d = np.load(CACHE / f"qmul_ident__{model}.npz")
    emb = l2n(d["emb"].astype(np.float64))
    split = d["split"]
    ids = d["ids"]
    return {
        "gal_emb": emb[split == "gallery"], "gal_ids": ids[split == "gallery"],
        "prb_emb": emb[split == "mated"], "prb_ids": ids[split == "mated"],
        "non_emb": emb[split == "unmated"],
        "non_ids": ids[split == "unmated"],
        # QMUL unmated_probe are PROBES of non-enrolled people, not gallery
        # entries. Adding them to the gallery (an earlier bug here) inflated it
        # from 2,965 to 95,837 and drove rank-1 to 0.53%. They are the only
        # true open-set probes either corpus provides.
        "non_role": "unmated_probe",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tinyface", "qmul"])
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--impostor-samples", type=int, default=50_000_000)
    ap.add_argument("--max-probes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    D = load_tinyface(args.model) if args.dataset == "tinyface" else load_qmul(args.model)
    rng = np.random.default_rng(args.seed)
    gal_emb, gal_ids = D["gal_emb"], D["gal_ids"]
    prb_emb, prb_ids = D["prb_emb"], D["prb_ids"]
    non_emb = D["non_emb"]
    print(f"{args.dataset}: gallery {gal_emb.shape[0]:,} / probes {prb_emb.shape[0]:,} "
          f"/ non-mates {non_emb.shape[0]:,}")

    # ---- genuine: probe x its mate in the gallery (official pairing) --------
    gal_by = defaultdict(list)
    for k, i in enumerate(gal_ids):
        gal_by[int(i)].append(k)
    ga, pa = [], []
    for k, i in enumerate(prb_ids):
        for g in gal_by.get(int(i), []):
            ga.append(g); pa.append(k)
    ga, pa = np.array(ga), np.array(pa)
    if ga.size > 500_000:
        sel = rng.choice(ga.size, 500_000, replace=False)
        ga, pa = ga[sel], pa[sel]
    genuine = np.einsum("ij,ij->i", gal_emb[ga], prb_emb[pa])
    print(f"genuine (probe x mated gallery): {genuine.size:,}")

    # ---- impostor: probe x non-mate, disjoint BY PROTOCOL -------------------
    n = args.impostor_samples
    a = rng.integers(0, prb_emb.shape[0], size=n)
    b = rng.integers(0, non_emb.shape[0], size=n)
    impostor = np.empty(n, dtype=np.float32)
    for s in range(0, n, 1_000_000):
        e = min(s + 1_000_000, n)
        impostor[s:e] = np.einsum("ij,ij->i", prb_emb[a[s:e]], non_emb[b[s:e]]).astype(np.float32)
    audit = audit_population(impostor, identity_labels_present=True, target_far=1e-5)
    print(f"impostor: {impostor.size:,}   audit={audit.verdict}")

    cap = capacity_from_pools(f"{args.dataset}_official", genuine, impostor)
    bits, _ = identity_bits(genuine, impostor)
    lo80, hi80 = bootstrap_ci(bits, lambda x: gallery_for_rank1(x, 0.80), n_boot=300)
    print(f"\nCAPACITY  censored {cap.censored_fraction:.3%}  ceiling {cap.censoring_ceiling_bits:.2f} bits")
    print(f"  bits p10/p20/med/mean: {cap.bits_p10:.2f} / {cap.bits_p20:.2f} / "
          f"{cap.bits_median:.2f} / {cap.bits_mean:.2f}")
    print(f"  gallery @50/80/90% rank-1: {cap.gallery_at_rank1_50:,.0f} / "
          f"{cap.gallery_at_rank1_80:,.0f} [{lo80:,.0f}, {hi80:,.0f}] / {cap.gallery_at_rank1_90:,.0f}")

    tar = {}
    for far in (1e-2, 1e-3, 1e-4, 1e-5):
        thr = float(np.quantile(impostor, 1.0 - far))
        tar[f"tar_at_far_{far:g}"] = float((genuine > thr).mean())
    print("\nTAR vs official non-mate population")
    for k, v in tar.items():
        print(f"  {k:<22} {v*100:6.2f}%")

    # ---- open-set 1:N on the official gallery ------------------------------
    one_per_id = np.array([v[0] for v in gal_by.values()])
    ids_one = np.array(list(gal_by.keys()))
    role = D["non_role"]

    if role == "gallery_filler":
        full_gal = np.concatenate([gal_emb[one_per_id], non_emb])
        full_ids = np.concatenate([ids_one, np.full(non_emb.shape[0], -1)])
        all_prb, all_ids = prb_emb, prb_ids
    else:
        # Unmated probes join the PROBE set; the gallery stays as enrolled only.
        full_gal, full_ids = gal_emb[one_per_id], ids_one
        all_prb = np.concatenate([prb_emb, non_emb])
        all_ids = np.concatenate([prb_ids, np.full(non_emb.shape[0], -10**9)])

    mated_mask = np.array([int(i) in gal_by for i in all_ids])
    idx = rng.choice(all_prb.shape[0], min(args.max_probes, all_prb.shape[0]), replace=False)
    P, T, M = all_prb[idx], all_ids[idx], mated_mask[idx]
    print(f"\nOPEN-SET 1:N  gallery={full_gal.shape[0]:,} "
          f"({ids_one.size:,} enrolled + {non_emb.shape[0]:,} non-mates)")
    print(f"  probes={P.shape[0]:,}  mated={int(M.sum()):,}  unmated={int((~M).sum()):,}")

    sims = P @ full_gal.T
    top_i = np.argmax(sims, axis=1)
    top = sims[np.arange(sims.shape[0]), top_i]
    top_id = full_ids[top_i]

    ranks = []
    for r in np.flatnonzero(M):
        mate = np.flatnonzero(full_ids == T[r])
        if mate.size:
            ranks.append(int((sims[r] > sims[r][mate[0]]).sum()) + 1)
    ranks = np.array(ranks)
    cmc = {f"rank_{k}": float((ranks <= k).mean()) for k in (1, 5, 10, 20, 50, 100)}
    openset = {}
    if (~M).sum() > 0:
        for fpir in (0.01, 0.1):
            thr = float(np.quantile(top[~M], 1.0 - fpir))
            openset[f"tpir_at_fpir_{fpir}"] = float(
                ((M) & (top > thr) & (top_id == T)).sum() / max(M.sum(), 1)
            )
    print("  CMC     " + "  ".join(f"{k}={v*100:.2f}%" for k, v in cmc.items()))
    if openset:
        print("  OpenSet " + "  ".join(f"{k}={v*100:.2f}%" for k, v in openset.items()))

    # ---- calibration -------------------------------------------------------
    m = min(genuine.size, 100_000)
    gs = rng.choice(genuine.size, m, replace=False)
    isamp = rng.choice(impostor.size, m, replace=False)
    sc = np.empty(2 * m); lb = np.zeros(2 * m, dtype=bool)
    sc[0::2] = genuine[gs]; lb[0::2] = True
    sc[1::2] = impostor[isamp]
    cv = cllr_report(cross_validated_log10_lr(sc, lb, 10), lb, scores=sc)
    print(f"\nCALIBRATION  Cllr={cv.cllr:.4f}  Cllr_min={cv.cllr_min:.4f}  "
          f"Cllr_cal={cv.cllr_cal:.4f}  ({cv.cllr_cal/max(cv.cllr,1e-9)*100:.1f}% recoverable)")

    payload = {
        "dataset": args.dataset, "model": args.model, "protocol": "official split",
        "counts": {
            "gallery": int(gal_emb.shape[0]), "probes": int(prb_emb.shape[0]),
            "non_mates": int(non_emb.shape[0]), "genuine_pairs": int(genuine.size),
            "impostor_pairs": int(impostor.size),
        },
        "population_audit": audit.as_dict(),
        "capacity": cap.as_dict(),
        "gallery_at_rank1_80_ci95": [lo80, hi80],
        "tar": tar,
        "identification": {
            "gallery_size": int(full_gal.shape[0]), "n_probes": int(P.shape[0]),
            "n_mated": int(M.sum()), "n_unmated": int((~M).sum()),
            "cmc": cmc, "open_set": openset,
        },
        "calibration": cv.as_dict(),
        "supersedes": "capacity_real_gallery.json (non-official pairing, rank-1 inflated)",
    }
    # Model key is IN THE FILENAME. An earlier version wrote
    # capacity_official_<dataset>.json regardless of backbone, so each run
    # silently destroyed the previous model's result -- the w600k_r50 TinyFace
    # artifact was overwritten twice before this was noticed, and its numbers
    # survived only because they had already been transcribed into
    # docs/MEASUREMENT_RECORD.md. A comparison harness that cannot hold two
    # results at once is not a comparison harness.
    out = OUT / f"capacity_official_{args.dataset}__{args.model}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
