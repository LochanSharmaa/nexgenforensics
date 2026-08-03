#!/usr/bin/env python
"""Embed the InsightFace protocol packs with the ViT backbone, and recalibrate.

    python backend/scripts/embed_packs_with_vit.py

WHY THIS IS REQUIRED BEFORE ANY BACKBONE SWAP
---------------------------------------------
The deployed threshold 0.2871 is the FMR=0.1% operating point *for w600k_r50*,
measured on AgeDB. It is a property of that model's score distribution, not a
universal constant. Measured distributions on TinyFace:

    w600k_r50          genuine mean +0.3268   impostor mean +0.0912
    vit_kprpe_wf12m    genuine mean +0.3757   impostor mean +0.0097

Carrying 0.2871 across would put the operating point in a completely different
place on the ROC. Every threshold must be re-derived on the new backbone, by the
same rule that derived the old one, or the decision layer silently changes
meaning while continuing to look calibrated.

ALIGNMENT: pack crops are already ArcFace-aligned (that is what the .bin format
is), so canonical keypoints are correct here and the DFA aligner is skipped --
the same reasoning, inverted, that makes the aligner mandatory for TinyFace and
QMUL detector crops. See test_vit_backbone_contract.py defect 10.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
for p in (str(_BACKEND), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from nexgen_engine.benchmarks.verification import decode_pack, evaluate_pairs, l2n, load_pack  # noqa: E402
from nexgen_engine.models.cvlface_backbone import ARCFACE_5PTS, CvlfaceViTKprpe  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
OUT = _ROOT / "runtime" / "forensics"
MODEL_KEY = "vit_kprpe_wf12m"  # overridden to *_lora_<tag> when --lora is given
PACK_DIRS = [
    _ROOT / "src_extracted/faces_webface_112x112/faces_webface_112x112",
    _ROOT / "src_extracted/faces_megafacetrain_112x112/faces_megafacetrain_112x112",
    _ROOT / "src_extracted/faces_umd/faces_umd",
]
DATASETS = ["lfw", "cfp_ff", "cfp_fp", "agedb_30", "calfw", "cplfw"]

MIRROR_5PTS = np.array(
    [
        [1.0 - ARCFACE_5PTS[1][0], ARCFACE_5PTS[1][1]],
        [1.0 - ARCFACE_5PTS[0][0], ARCFACE_5PTS[0][1]],
        [1.0 - ARCFACE_5PTS[2][0], ARCFACE_5PTS[2][1]],
        [1.0 - ARCFACE_5PTS[4][0], ARCFACE_5PTS[4][1]],
        [1.0 - ARCFACE_5PTS[3][0], ARCFACE_5PTS[3][1]],
    ],
    dtype=np.float32,
)


def find_pack(ds: str) -> Path | None:
    for d in PACK_DIRS:
        p = d / f"{ds}.bin"
        if p.exists():
            return p
    return None


def embed_pack(model, images, batch: int) -> np.ndarray:
    out = np.empty((len(images), 512), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(images), batch):
        chunk = list(images[i : i + batch])
        n = len(chunk)
        out[i : i + n] = model.get_feat(chunk, np.repeat(ARCFACE_5PTS[None], n, axis=0)) + model.get_feat(
            [im[:, ::-1] for im in chunk], np.repeat(MIRROR_5PTS[None], n, axis=0)
        )
        if (i // batch) % 20 == 0:
            r = (i + n) / max(time.time() - t0, 1e-6)
            print(f"    {i + n:,}/{len(images):,}  {r:.0f} img/s", end="\r", flush=True)
    print()
    return out


def fmr_threshold(scores: np.ndarray, labels: np.ndarray, fmr: float) -> float:
    """Score at which the false-match rate equals `fmr`. Same rule as 0.2871."""
    return float(np.quantile(scores[~labels], 1.0 - fmr))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lora", default=None,
                    help="path to a LoRA adapter; results are cached under a distinct model key")
    args = ap.parse_args()
    global MODEL_KEY
    if args.lora:
        # Distinct key so fine-tuned results can never overwrite the base
        # backbone's artifacts -- the defect-11 lesson.
        MODEL_KEY = f"vit_kprpe_{Path(args.lora).stem}"
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    model = CvlfaceViTKprpe(batch_size=args.batch, lora_path=args.lora)
    print(f"backbone: {model.provider_label}  (canonical keypoints; packs are pre-aligned)")

    results, thresholds = {}, {}
    for ds in DATASETS:
        pack = find_pack(ds)
        if pack is None:
            print(f"  {ds}: pack not found -- skipped")
            continue
        cache = CACHE / f"{ds}__{MODEL_KEY}.npz"
        bins, issame = load_pack(pack)
        labels = np.asarray(issame, dtype=bool)
        if cache.exists() and not args.force:
            emb = np.load(cache)["emb"]
            print(f"  {ds}: cache hit")
        else:
            print(f"  {ds}: {len(bins):,} images")
            emb = embed_pack(model, decode_pack(bins), args.batch)
            np.savez_compressed(cache, emb=emb, issame=labels, flip_tta=True)

        e = l2n(emb.astype(np.float64))
        r = evaluate_pairs(e[0::2], e[1::2], labels, ds, f"single:{MODEL_KEY}")
        scores = np.sum(e[0::2] * e[1::2], axis=1)
        results[ds] = {
            "accuracy_mean": r.accuracy_mean,
            "accuracy_std": r.accuracy_std,
            "tar_at_far_1e3": r.tar_at_far_1e3,
            "tar_at_far_1e4": r.tar_at_far_1e4,
            "auc": r.auc,
            "eer": r.eer,
            "fmr_1e3_threshold": fmr_threshold(scores, labels, 1e-3),
            "fmr_1e4_threshold": fmr_threshold(scores, labels, 1e-4),
            "genuine_mean": float(scores[labels].mean()),
            "impostor_mean": float(scores[~labels].mean()),
        }
        print(f"    acc {r.accuracy_mean*100:.2f}+/-{r.accuracy_std*100:.2f}  "
              f"TAR@FAR0.1% {r.tar_at_far_1e3*100:.2f}  "
              f"FMR=0.1% thr {results[ds]['fmr_1e3_threshold']:.4f}")

    if "agedb_30" in results:
        a = results["agedb_30"]
        thresholds = {
            "match": round(a["fmr_1e3_threshold"], 4),
            "review": round(a["fmr_1e3_threshold"] * 0.75, 4),
            "derivation": (
                "match = FMR=0.1% operating point on AgeDB-30, the same rule that "
                "produced 0.2871 for w600k_r50. review = 0.75 x match, preserving "
                "the incumbent's 0.2153/0.2871 = 0.75 ratio."
            ),
            "incumbent_w600k_r50": {"match": 0.2871, "review": 0.2153},
        }
        print(f"\nRECALIBRATED THRESHOLDS ({MODEL_KEY})")
        print(f"  match  {thresholds['match']:.4f}   (w600k_r50: 0.2871)")
        print(f"  review {thresholds['review']:.4f}   (w600k_r50: 0.2153)")

    payload = {"model": MODEL_KEY, "flip_tta": True, "alignment": "canonical (packs pre-aligned)",
               "datasets": results, "thresholds": thresholds}
    out = OUT / f"pack_benchmarks__{MODEL_KEY}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
