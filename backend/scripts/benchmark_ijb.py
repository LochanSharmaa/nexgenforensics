#!/usr/bin/env python
"""G4 -- IJB-B / IJB-C 1:1 verification, against the official protocol.

    python backend/scripts/benchmark_ijb.py --dataset IJBB
    python backend/scripts/benchmark_ijb.py --dataset IJBC
    python backend/scripts/benchmark_ijb.py --dataset IJBC --model vit_kprpe_wf12m

WHY THIS RUNS AT ALL
--------------------
PROJECT_OVERVIEW and SCORECARD both describe IJB as "a 1.57 GB partial download".
It is the complete suite: IJB-B 227,630 crops, IJB-C 469,375 crops, full meta,
the official IJB_11.py, and reference ArcFace result vectors. It has been on disk
unrun for the entire project.

WHAT IT IS FOR
--------------
Harness validation, not a new capability claim. IJB-B/C are high-quality imagery,
so they say nothing about the degraded case that decides this programme. What
they do is check our pipeline against published numbers -- if we cannot
reproduce a known ArcFace result on a standard protocol, every other number we
report is suspect.

PROTOCOL, following IJB_11.py exactly:
  1. Align each loose_crop with its 5 landmarks (similarity transform -> 112x112)
  2. Embed, with horizontal-flip TTA
  3. Pool to templates: average within each MEDIA, then SUM across medias,
     then L2-normalise. Media-aware pooling matters -- a template with 40 frames
     from one video and 1 still should not be dominated by the video.
  4. Score the official template pairs by cosine
  5. TAR at FAR 1e-6..1e-1

Expected reference (published ArcFace, R100/MS1MV2): IJB-B ~94-95% and
IJB-C ~96-97% TAR@FAR=1e-4. We run R50/WebFace600K, so a few points lower is
correct; a large gap means our harness is wrong.

ALIGNMENT, AND WHY THE DFA ALIGNER IS NOT USED HERE
---------------------------------------------------
Step 1 above runs the official 5-point `norm_crop`, so what reaches the model is
an ArcFace-aligned 112x112 crop -- structurally identical to a `.bin` pack crop.
Canonical KP-RPE keypoints are therefore correct by construction, and the DFA
aligner is skipped, for exactly the reason it is mandatory on TinyFace/QMUL
detector crops. Same reasoning as embed_packs_with_vit.py; see
test_vit_backbone_contract.py defect 10.

The flip-TTA pass gets MIRRORED keypoints. Feeding unmirrored canonical points
alongside a mirrored image is the same class of error as defect 10 -- false
landmark information handed straight to the attention mechanism, on half of
every embedding.

Artifacts carry the model key in the filename (defect 11): a model-agnostic
`ijb_<ds>.json` let each run destroy the previous one's result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
for p in (str(_BACKEND), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

_ROOT = _BACKEND.parent
IJB = _ROOT / "src_extracted" / "ijb" / "ijb"
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
OUT = _ROOT / "runtime" / "forensics"


def read_meta(ds: str):
    low = ds.lower()
    meta = IJB / ds / "meta"
    names, lmks, scores = [], [], []
    with open(meta / f"{low}_name_5pts_score.txt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            names.append(parts[0])
            lmks.append([float(x) for x in parts[1:11]])
            scores.append(float(parts[11]))
    tid, mid = [], []
    with open(meta / f"{low}_face_tid_mid.txt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            tid.append(int(parts[1]))
            mid.append(int(parts[2]))
    pair_file = meta / f"{low}_template_pair_label.txt"
    try:
        # np.loadtxt takes minutes on IJB-C's 15.6M-row / 222 MB list.
        import pandas as pd  # noqa: PLC0415

        pairs = pd.read_csv(pair_file, sep=r"\s+", header=None).to_numpy(dtype=np.int64)
    except ImportError:
        pairs = np.loadtxt(pair_file, dtype=np.int64)
    return (
        names,
        np.array(lmks, dtype=np.float32).reshape(-1, 5, 2),
        np.array(scores, dtype=np.float32),
        np.array(tid, dtype=np.int64),
        np.array(mid, dtype=np.int64),
        pairs,
    )


def load_backbone(model_key: str):
    """Return (model, embed_fn). `embed_fn(imgs)` does the flip-TTA sum."""
    if model_key.startswith("vit_kprpe"):
        from nexgen_engine.models.cvlface_backbone import (  # noqa: PLC0415
            ARCFACE_5PTS,
            CvlfaceViTKprpe,
        )

        # Mirror of the canonical template: reflect x, and swap the L/R pairs
        # (eyes, mouth corners) so landmark identity survives the flip.
        mirror = np.array(
            [
                [1.0 - ARCFACE_5PTS[1][0], ARCFACE_5PTS[1][1]],
                [1.0 - ARCFACE_5PTS[0][0], ARCFACE_5PTS[0][1]],
                [1.0 - ARCFACE_5PTS[2][0], ARCFACE_5PTS[2][1]],
                [1.0 - ARCFACE_5PTS[4][0], ARCFACE_5PTS[4][1]],
                [1.0 - ARCFACE_5PTS[3][0], ARCFACE_5PTS[3][1]],
            ],
            dtype=np.float32,
        )
        model = CvlfaceViTKprpe(batch_size=64)
        print(f"    [{model_key}] {model.provider_label}", flush=True)

        def embed(imgs):
            n = len(imgs)
            return np.asarray(
                model.get_feat(imgs, np.repeat(ARCFACE_5PTS[None], n, axis=0)), dtype=np.float32
            ) + np.asarray(
                model.get_feat([x[:, ::-1] for x in imgs], np.repeat(mirror[None], n, axis=0)),
                dtype=np.float32,
            )

        return model, embed

    from benchmark_verification import load_recognizer  # noqa: PLC0415

    model = load_recognizer(model_key)

    def embed(imgs):
        return np.asarray(model.get_feat(imgs), dtype=np.float32) + np.asarray(
            model.get_feat([x[:, ::-1] for x in imgs]), dtype=np.float32
        )

    return model, embed


def embed_all(ds: str, names, lmks, embed, model_key: str, batch_size: int, force: bool) -> np.ndarray:
    cache = CACHE / f"{ds.lower()}__{model_key}.npz"
    if cache.exists() and not force:
        print(f"  cache hit: {cache.name}")
        return np.load(cache)["emb"]

    from insightface.utils import face_align  # noqa: PLC0415

    root = IJB / ds / "loose_crop"
    out = np.empty((len(names), 512), dtype=np.float32)

    def prep(i: int) -> np.ndarray:
        im = cv2.imread(str(root / names[i]))
        if im is None:
            raise ValueError(f"decode failed: {names[i]}")
        # Same alignment as the reference implementation: 5-point similarity
        # transform onto the ArcFace canonical layout.
        return face_align.norm_crop(im, landmark=lmks[i], image_size=112)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(0, len(names), batch_size):
            idx = list(range(i, min(i + batch_size, len(names))))
            imgs = list(pool.map(prep, idx))
            out[i : i + len(idx)] = embed(imgs)
            done = i + len(idx)
            if (i // batch_size) % 20 == 0 or done == len(names):
                r = done / max(time.time() - t0, 1e-6)
                print(f"    {done:,}/{len(names):,}  {r:.0f} img/s  "
                      f"eta {(len(names)-done)/max(r,1e-6)/60:.1f} min", end="\r", flush=True)
    print(f"\n  embedded {len(names):,} in {time.time()-t0:.0f}s")
    np.savez(cache, emb=out)
    return out


def pool_templates(feats: np.ndarray, tid: np.ndarray, mid: np.ndarray):
    """Media-aware template pooling, per IJB_11.py image2template_feature."""
    feats = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)
    uniq = np.unique(tid)
    out = np.zeros((uniq.size, feats.shape[1]), dtype=np.float32)
    for k, t in enumerate(uniq):
        sel = np.flatnonzero(tid == t)
        media = mid[sel]
        acc = []
        for m in np.unique(media):
            f = feats[sel[media == m]]
            acc.append(f[0] if f.shape[0] == 1 else f.mean(axis=0))
        out[k] = np.sum(np.stack(acc), axis=0)
        if (k + 1) % 2000 == 0:
            print(f"    pooled {k+1:,}/{uniq.size:,}", end="\r", flush=True)
    out /= np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
    print(f"    pooled {uniq.size:,} templates" + " " * 20)
    return out, uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="IJBB", choices=["IJBB", "IJBC"])
    ap.add_argument("--model", default="w600k_r50",
                    choices=["w600k_r50", "glintr100", "vit_kprpe_wf12m"])
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not (IJB / args.dataset).is_dir():
        raise SystemExit(f"{IJB / args.dataset} not found")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    print(f"{args.dataset}: reading meta")
    names, lmks, det, tid, mid, pairs = read_meta(args.dataset)
    print(f"  {len(names):,} faces, {np.unique(tid).size:,} templates, {pairs.shape[0]:,} pairs")

    _, embed = load_backbone(args.model)
    feats = embed_all(args.dataset, names, lmks, embed, args.model, args.batch_size, args.force)

    print("  pooling templates")
    tfeat, uniq = pool_templates(feats, tid, mid)
    pos = {int(t): i for i, t in enumerate(uniq)}

    print("  scoring pairs")
    p1 = np.array([pos[int(x)] for x in pairs[:, 0]])
    p2 = np.array([pos[int(x)] for x in pairs[:, 1]])
    labels = pairs[:, 2].astype(bool)
    scores = np.empty(p1.size, dtype=np.float32)
    for s in range(0, p1.size, 2_000_000):
        e = min(s + 2_000_000, p1.size)
        scores[s:e] = np.einsum("ij,ij->i", tfeat[p1[s:e]], tfeat[p2[s:e]])

    imp = np.sort(scores[~labels])
    gen = scores[labels]
    res, support = {}, {}
    print(f"\n{args.dataset}  {int(labels.sum()):,} genuine / {imp.size:,} impostor pairs")
    print(f"  {'FAR':>8} {'TAR':>9} {'impostor pairs above thr':>26}")
    for far in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        # Exact order statistic, not an interpolated quantile: at FAR=1e-6 the
        # threshold rests on a handful of pairs and interpolation invents
        # precision the data does not have.
        k = int(np.ceil(far * imp.size))
        thr = float(imp[-k])
        tar = float((gen > thr).mean())
        res[f"tar_at_far_{far:g}"] = tar
        support[f"impostors_above_far_{far:g}"] = k
        print(f"  {far:>8.0e} {tar*100:>8.2f}% {k:>26,}")
    print("\n  NOTE: any operating point resting on <100 impostor pairs carries a\n"
          "  confidence interval several points wide. See MEASUREMENT_RECORD.md.")

    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "flip_tta": True,
        "alignment": "insightface face_align.norm_crop, official 5pts",
        "pooling": "media-average then media-sum, per IJB_11.py",
        "n_faces": len(names),
        "n_templates": int(uniq.size),
        "n_pairs": int(pairs.shape[0]),
        "n_genuine": int(labels.sum()),
        "n_impostor": int(imp.size),
        "tar": res,
        "impostor_support": support,
        "note": (
            "Harness validation. Published ArcFace R100/MS1MV2 reaches ~94-95% (IJB-B) "
            "and ~96-97% (IJB-C) at FAR=1e-4. This is R50/WebFace600K, so a few points "
            "lower is expected; a large gap indicates a harness fault."
        ),
    }
    out = OUT / f"ijb_{args.dataset.lower()}__{args.model}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
