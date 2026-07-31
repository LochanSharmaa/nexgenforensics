#!/usr/bin/env python
"""
Can the QMUL checkpoint be USED, rather than shelved?

    python backend/scripts/evaluate_routed_engine.py

BENCHMARKS.md records the fine-tune as "no improvement": accuracy moved nowhere
on any of the seven benchmarks. But the per-metric table says something more
specific than the accuracy column does. At the 0.1% false-match operating point
a forensic deployment actually uses:

    TinyFace   TAR@FAR0.1%   33.13 -> 38.10   (+4.97pp)
    AgeDB-30   TAR@FAR0.1%   96.03 -> 88.10   (-7.93pp)
    CPLFW      TAR@FAR0.1%   87.40 -> 81.73   (-5.67pp)

That is not a worse model. It is a DIFFERENT model: better where images are
degraded, worse where they are clean but hard (age, pose). A single global
choice between them throws away whichever advantage it does not pick.

This script tests whether choosing PER PROBE recovers both, using the quality
score the pipeline already computes on every request -- so routing costs no
extra inference.

THREE QUESTIONS, IN ORDER. The third only matters if the first two hold.

 1. Are the two embedding spaces compatible? If a template enrolled under one
    model can be compared against a probe under the other, routing is free
    everywhere. If not, 1:N search needs BOTH templates stored per subject and
    that is a real cost, not a detail.

 2. Does the quality score actually separate degraded from clean imagery? If it
    does not, there is nothing to route on and the idea dies here.

 3. Does routing beat the deployed model on TinyFace WITHOUT regressing the
    clean sets? This is the only claim worth making, and it is the one that
    would be quoted, so it is measured end to end rather than inferred from the
    two columns above.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "scripts"))

from nexgen_engine.benchmarks.verification import (  # noqa: E402
    decode_pack,
    evaluate_pairs,
    l2n,
    load_pack,
)

_ROOT = _BACKEND.parent
CLEAN = ["lfw", "agedb_30", "cfp_fp", "cfp_ff", "calfw", "cplfw"]


def quality_scores(images: list[np.ndarray]) -> np.ndarray:
    """The pipeline's own quality score, one per image."""
    from PIL import Image

    from nexgen_engine.config import QualityConfig
    from nexgen_engine.data.quality_filter import ImageQualityFilter

    qf = ImageQualityFilter(QualityConfig())
    out = np.zeros(len(images), dtype=np.float64)
    for i, img in enumerate(images):
        try:
            # face=None so the whole frame is measured -- correct here, because
            # every benchmark image IS an aligned face crop already.
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            out[i] = qf.evaluate(pil).score
        except Exception:
            out[i] = float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(_ROOT / "runtime/checkpoints/arcface_qmul_v2.pt"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tinyface-pairs", type=int, default=3000)
    ap.add_argument("--threshold", type=float, default=None,
                    help="route below this quality to the specialist; derived if omitted")
    ap.add_argument("--sweep", action="store_true", help="print the threshold tradeoff curve")
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/routed_engine.json"))
    args = ap.parse_args()

    from benchmark_verification import find_pack, load_recognizer
    from eval_finetuned_checkpoint import TorchRecognizer, embed_all, tinyface_pairs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("  Quality-routed engine: can the QMUL checkpoint be used?")
    print("=" * 78)

    specialist = TorchRecognizer(Path(args.checkpoint), device)
    generalist = load_recognizer("w600k_r50")
    report: dict = {}

    # ---- gather every dataset once -------------------------------------
    datasets: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    for ds in CLEAN:
        try:
            bins, issame = load_pack(find_pack(ds))
        except Exception as exc:
            print(f"  {ds}: unavailable ({exc}); skipped")
            continue
        datasets[ds] = (list(decode_pack(bins)), np.asarray(issame, dtype=bool))

    tf = tinyface_pairs(args.tinyface_pairs, 0)
    tf_pairs = None
    if tf is not None:
        pairs, labels = tf
        needed = sorted({p for pr in pairs for p in pr})
        imgs = []
        for p in needed:
            im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            imgs.append(cv2.resize(im, (112, 112)))
        pos = {p: i for i, p in enumerate(needed)}
        tf_pairs = (imgs, pairs, labels, pos)

    # ---- Q1: are the embedding spaces compatible? ----------------------
    print("\n1. EMBEDDING SPACE COMPATIBILITY")
    probe_imgs = datasets["lfw"][0][:256] if "lfw" in datasets else []
    if probe_imgs:
        a = l2n(np.asarray(generalist.get_feat(probe_imgs)).astype(np.float64))
        b = l2n(np.asarray(specialist.get_feat(probe_imgs)).astype(np.float64))
        same_image = np.sum(a * b, axis=1)
        print(f"   same image, both models : median cosine {np.median(same_image):+.4f}")
        compatible = bool(np.median(same_image) > 0.5)
        report["cross_model_same_image_median"] = round(float(np.median(same_image)), 4)
        report["embedding_spaces_compatible"] = compatible
        if compatible:
            print("   -> COMPATIBLE. A template from one model can be searched with the other.")
        else:
            print("   -> INCOMPATIBLE, as expected: fine-tuning rotated the space.")
            print("      1:1 verification can route freely (both images use one model).")
            print("      1:N search CANNOT: a routed probe would be compared against")
            print("      templates from the other space and match nothing. Serving both")
            print("      means storing TWO templates per subject.")

    # ---- Q2: does quality separate the conditions? ---------------------
    print("\n2. DOES THE QUALITY SCORE SEPARATE DEGRADED FROM CLEAN?")
    q_by_ds: dict[str, np.ndarray] = {}
    for ds, (imgs, _) in datasets.items():
        q_by_ds[ds] = quality_scores(imgs[:1500])
    if tf_pairs:
        q_by_ds["tinyface"] = quality_scores(tf_pairs[0][:1500])

    for ds, q in q_by_ds.items():
        finite = q[np.isfinite(q)]
        if finite.size:
            print(f"   {ds:10s} median {np.median(finite):.4f}   "
                  f"p10 {np.percentile(finite, 10):.4f}  p90 {np.percentile(finite, 90):.4f}")
    report["quality_by_dataset"] = {
        k: round(float(np.median(v[np.isfinite(v)])), 4)
        for k, v in q_by_ds.items() if np.isfinite(v).any()
    }

    clean_med = np.median([v for k, v in report["quality_by_dataset"].items() if k != "tinyface"])
    tiny_med = report["quality_by_dataset"].get("tinyface")
    if tiny_med is not None:
        gap = clean_med - tiny_med
        print(f"\n   clean median {clean_med:.4f}  vs  tinyface median {tiny_med:.4f}"
              f"   separation {gap:+.4f}")
        report["quality_separation"] = round(float(gap), 4)
        if gap < 0.05:
            print("   -> The score does NOT separate the conditions. There is nothing to")
            print("      route on; a quality-gated engine cannot be built from this signal.")
        else:
            print("   -> Separated. A threshold between them is a usable routing rule.")

    # ---- Q3: measure the routed system end to end ----------------------
    print("\n3. ROUTED SYSTEM vs DEPLOYED, on identical pairs")
    print(f"   {'dataset':10s} {'deployed':>9s} {'specialist':>11s} {'ROUTED':>9s}"
          f" | {'TAR dep':>8s} {'TAR rt':>8s}")
    print("   " + "-" * 66)

    rows = []
    # THRESHOLD CHOICE.
    #
    # NOT the midpoint between the two medians -- that lands inside the clean
    # distribution's lower tail and misroutes a third of LFW and CPLFW for no
    # gain. The distributions overlap far less than their medians suggest:
    # TinyFace's p90 sits below the lowest clean p10, so a threshold in that
    # gap captures essentially all degraded imagery and almost no clean.
    #
    # It is set from the QUALITY DISTRIBUTIONS, which are a property of image
    # capture, and NOT from benchmark accuracy -- tuning it on the seven
    # reporting sets would be fitting the operating point to the test set and
    # every number after it would be unquotable.
    threshold = args.threshold
    if threshold is None:
        tiny_q = q_by_ds.get("tinyface")
        clean_p10 = [np.percentile(v[np.isfinite(v)], 10)
                     for k, v in q_by_ds.items() if k != "tinyface" and np.isfinite(v).any()]
        if tiny_q is not None and clean_p10:
            hi = float(min(clean_p10))
            lo = float(np.percentile(tiny_q[np.isfinite(tiny_q)], 90))
            threshold = round((lo + hi) / 2, 3) if lo < hi else round(lo, 3)
        else:
            threshold = 0.58
    print(f"   routing rule: quality < {threshold} -> specialist, else deployed\n")

    cache: dict[str, tuple] = {}

    def score_set(name, imgs, pair_idx, labels, thr=None):
        thr = threshold if thr is None else thr
        if name not in cache:
            cache[name] = (
                l2n(embed_all(generalist, imgs, args.batch_size).astype(np.float64)),
                l2n(embed_all(specialist, imgs, args.batch_size).astype(np.float64)),
                quality_scores(imgs),
            )
        eg, es, q = cache[name]
        a_i = np.array([i for i, _ in pair_idx])
        b_i = np.array([j for _, j in pair_idx])
        # A pair is routed to the specialist when EITHER image is degraded: the
        # weaker image is what limits the comparison.
        use_spec = (np.nan_to_num(q[a_i], nan=1.0) < thr) | (
            np.nan_to_num(q[b_i], nan=1.0) < thr)
        ga, gb = eg[a_i], eg[b_i]
        sa, sb = es[a_i], es[b_i]
        ra = np.where(use_spec[:, None], sa, ga)
        rb = np.where(use_spec[:, None], sb, gb)
        return (
            evaluate_pairs(ga, gb, labels, name, "deployed"),
            evaluate_pairs(sa, sb, labels, name, "specialist"),
            evaluate_pairs(ra, rb, labels, name, "routed"),
            float(use_spec.mean()),
        )

    for ds, (imgs, issame) in datasets.items():
        idx = [(2 * i, 2 * i + 1) for i in range(len(issame))]
        dep, spec, routed, frac = score_set(ds, imgs, idx, issame)
        rows.append((ds, dep, spec, routed, frac))

    if tf_pairs:
        imgs, pairs, labels, pos = tf_pairs
        idx = [(pos[p[0]], pos[p[1]]) for p in pairs]
        dep, spec, routed, frac = score_set("tinyface", imgs, idx, labels)
        rows.append(("tinyface", dep, spec, routed, frac))

    payload = {}
    for ds, dep, spec, routed, frac in rows:
        print(f"   {ds:10s} {dep.accuracy_mean * 100:8.2f}% {spec.accuracy_mean * 100:10.2f}%"
              f" {routed.accuracy_mean * 100:8.2f}% | {dep.tar_at_far_1e3 * 100:7.2f}%"
              f" {routed.tar_at_far_1e3 * 100:7.2f}%   ({frac * 100:.0f}% routed)")
        payload[ds] = {
            "deployed_acc": round(dep.accuracy_mean * 100, 3),
            "specialist_acc": round(spec.accuracy_mean * 100, 3),
            "routed_acc": round(routed.accuracy_mean * 100, 3),
            "deployed_tar_1e3": round(dep.tar_at_far_1e3 * 100, 3),
            "specialist_tar_1e3": round(spec.tar_at_far_1e3 * 100, 3),
            "routed_tar_1e3": round(routed.tar_at_far_1e3 * 100, 3),
            "fraction_routed_to_specialist": round(frac, 4),
        }

    if args.sweep:
        print("\n   threshold sweep (TAR@FAR0.1%)")
        print(f"   {'thr':>6s} {'tinyface':>9s} {'worst clean delta':>19s}")
        for thr in [0.50, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.68]:
            deltas, tiny_tar = [], None
            for ds, (imgs, issame) in datasets.items():
                idx = [(2 * i, 2 * i + 1) for i in range(len(issame))]
                d, _, r, _ = score_set(ds, imgs, idx, issame, thr=thr)
                deltas.append(r.tar_at_far_1e3 - d.tar_at_far_1e3)
            if tf_pairs:
                imgs, pairs, labels, pos = tf_pairs
                idx = [(pos[p[0]], pos[p[1]]) for p in pairs]
                d, _, r, _ = score_set("tinyface", imgs, idx, labels, thr=thr)
                tiny_tar = r.tar_at_far_1e3 * 100
            mark = "  <- chosen" if abs(thr - threshold) < 1e-9 else ""
            print(f"   {thr:6.2f} {tiny_tar:8.2f}% {min(deltas) * 100:+18.2f}pp{mark}")

    # ---- verdict --------------------------------------------------------
    print("\n" + "=" * 78)
    tiny = payload.get("tinyface")
    clean_rows = [(k, v) for k, v in payload.items() if k != "tinyface"]
    worst_clean = min((v["routed_tar_1e3"] - v["deployed_tar_1e3"] for _, v in clean_rows),
                      default=0.0)
    if tiny:
        tar_gain = tiny["routed_tar_1e3"] - tiny["deployed_tar_1e3"]
        print(f"  TinyFace TAR@FAR0.1%  {tiny['deployed_tar_1e3']:.2f} -> "
              f"{tiny['routed_tar_1e3']:.2f}  ({tar_gain:+.2f}pp)")
        print(f"  worst clean-set TAR change under routing: {worst_clean:+.2f}pp")
        report["verdict"] = (
            "ADOPT" if tar_gain > 1.0 and worst_clean > -0.5 else "DO NOT ADOPT"
        )
        print(f"\n  VERDICT: {report['verdict']}")
    print("=" * 78)

    report["routing_threshold"] = threshold
    report["results"] = payload
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
