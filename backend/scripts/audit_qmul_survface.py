#!/usr/bin/env python
"""
QMUL-SurvFace identity-overlap audit + exclusion list.

    python backend/scripts/audit_qmul_survface.py

Same methodology as build_exclusion_list.py (CASIA), with two additions that
this dataset specifically requires.

WHY THIS ONE NEEDS EXTRA CARE
-----------------------------
1. TinyFace and QMUL-SurvFace come from THE SAME LAB (Cheng, Zhu & Gong at
   QMUL). TinyFace is the degraded-condition benchmark this whole exercise is
   trying to improve. If the two share source imagery, training on SurvFace and
   reporting a TinyFace gain would be measuring memorisation. So the nearest
   neighbour is attributed back to WHICH eval set it came from, not just scored.

2. SurvFace images are native low-resolution surveillance crops; the five clean
   eval sets are high-quality portraits. Degraded probes produce systematically
   WEAKER embeddings, which compresses cosine similarity downward for everything
   including true matches. A fixed 0.40 threshold carried over from the CASIA
   audit is therefore a LOOSER filter here in real terms, not a stricter one.
   Counts are reported at several thresholds so that sensitivity is visible
   rather than hidden behind one number.

The gallery covers all seven evaluation sets: LFW, AgeDB-30, CFP-FP, CFP-FF,
CALFW, CPLFW and TinyFace.
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(_BACKEND / "scripts"))

from nexgen_engine.benchmarks.verification import decode_pack, load_pack  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
QMUL = Path("C:/Users/hello/Downloads/QMUL-SurvFace-v1/QMUL-SurvFace/training_set")
TINYFACE = _ROOT / "src_extracted/tinyface/tinyface/Testing_Set"
CACHED_SETS = ["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"]
ID_RE = re.compile(r"^(\d+)_")


def embed_images(model, images: list[np.ndarray], batch: int = 64) -> np.ndarray:
    """Original + flip, summed then L2-normalised -- as every other benchmark."""
    n = len(images)
    out = np.zeros((n, 512), dtype=np.float32)
    for i in range(0, n, batch):
        chunk = images[i : i + batch]
        out[i : i + len(chunk)] = (
            np.asarray(model.get_feat(list(chunk)))
            + np.asarray(model.get_feat([c[:, ::-1] for c in chunk]))
        )
        if (i // batch) % 40 == 0:
            print(f"      {min(i + batch, n)}/{n}", end="\r", flush=True)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out


def build_gallery(model_key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (embeddings, source_id_per_row, source_names)."""
    from benchmark_verification import find_pack, load_recognizer

    parts, src, names = [], [], []
    model = None

    for ds in CACHED_SETS:
        p = CACHE / f"{ds}__{model_key}.npz"
        if not p.exists():
            print(f"    {ds}: no cache; SKIPPED")
            continue
        d = np.load(p)
        e = (d["orig"] + d["flip"]).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
        parts.append(e)
        src.append(np.full(len(e), len(names), dtype=np.int32))
        names.append(ds)
        print(f"    {ds}: {len(e):,} (cached)")

    # CFP-FF -- a published pack, never previously embedded here.
    ff_cache = CACHE / f"cfp_ff__{model_key}.npz"
    try:
        if ff_cache.exists():
            d = np.load(ff_cache)
            e = (d["orig"] + d["flip"]).astype(np.float32)
        else:
            bins, issame = load_pack(find_pack("cfp_ff"))
            imgs = decode_pack(bins)
            model = model or load_recognizer(model_key)
            print(f"    cfp_ff: embedding {len(imgs):,} ...")
            raw = np.zeros((len(imgs), 512), dtype=np.float32)
            for i in range(0, len(imgs), 64):
                c = imgs[i : i + 64]
                raw[i : i + len(c)] = (np.asarray(model.get_feat([x for x in c]))
                                       + np.asarray(model.get_feat([x[:, ::-1] for x in c])))
            np.savez_compressed(ff_cache, orig=raw, flip=np.zeros((1, 1)), issame=issame)
            e = raw
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        parts.append(e)
        src.append(np.full(len(e), len(names), dtype=np.int32))
        names.append("cfp_ff")
        print(f"    cfp_ff: {len(e):,}")
    except Exception as exc:
        print(f"    cfp_ff: unavailable ({exc}); SKIPPED")

    # TinyFace -- the labelled benchmark surface (Gallery_Match + Probe).
    tf_cache = CACHE / f"tinyface_labelled__{model_key}.npz"
    if tf_cache.exists():
        e = np.load(tf_cache)["emb"]
    else:
        files = []
        for sub in ("Gallery_Match", "Probe"):
            d = TINYFACE / sub
            if d.is_dir():
                files += sorted(d.glob("*.jpg"))
        if files:
            imgs = []
            for f in files:
                im = cv2.imdecode(np.frombuffer(f.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if im is not None:
                    imgs.append(cv2.resize(im, (112, 112)))
            model = model or load_recognizer(model_key)
            print(f"    tinyface: embedding {len(imgs):,} ...")
            e = embed_images(model, imgs)
            np.savez_compressed(tf_cache, emb=e)
        else:
            e = None
    if e is not None and len(e):
        parts.append(e.astype(np.float32))
        src.append(np.full(len(e), len(names), dtype=np.int32))
        names.append("tinyface")
        print(f"    tinyface: {len(e):,}")

    return np.concatenate(parts), np.concatenate(src), names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--per-identity", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/qmul_exclusion_list.json"))
    args = ap.parse_args()

    import faiss
    from benchmark_verification import load_recognizer

    print("=" * 78)
    print("  QMUL-SurvFace identity-overlap audit")
    print("=" * 78)

    if not QMUL.is_dir():
        print(f"  training set not found at {QMUL}")
        return 1

    print("\n  Building evaluation gallery (all 7 sets):")
    gallery, gsrc, names = build_gallery(args.model)
    print(f"\n  gallery total : {gallery.shape[0]:,} embeddings from {names}")

    # ---- sample QMUL training identities ----
    rng = np.random.default_rng(args.seed)
    id_dirs = sorted([d for d in QMUL.iterdir() if d.is_dir()], key=lambda p: p.name)
    print(f"  QMUL identities: {len(id_dirs):,}")

    paths, labels, dims = [], [], []
    for d in id_dirs:
        fs = sorted(d.glob("*.jpg"))
        if len(fs) > args.per_identity:
            fs = [fs[i] for i in rng.choice(len(fs), args.per_identity, replace=False)]
        for f in fs:
            paths.append(f)
            labels.append(d.name)
    print(f"  sampled {len(paths):,} images at <= {args.per_identity}/identity")

    model = load_recognizer(args.model)
    print("  embedding QMUL images ...")
    embs = np.zeros((len(paths), 512), dtype=np.float32)
    B = 64
    for i in range(0, len(paths), B):
        chunk = paths[i : i + B]
        imgs = []
        for f in chunk:
            im = cv2.imdecode(np.frombuffer(f.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                im = np.zeros((112, 112, 3), np.uint8)
            else:
                dims.append(im.shape[:2])
            imgs.append(cv2.resize(im, (112, 112)))
        embs[i : i + len(chunk)] = (
            np.asarray(model.get_feat(imgs))
            + np.asarray(model.get_feat([x[:, ::-1] for x in imgs]))
        )
        if (i // B) % 100 == 0:
            print(f"    {min(i + B, len(paths)):,}/{len(paths):,}", end="\r", flush=True)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    print(f"  embedded {len(paths):,} images                    ")

    # ---- exact nearest neighbour ----
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(np.ascontiguousarray(gallery))
    sims, idx = index.search(np.ascontiguousarray(embs), 1)
    best, who = sims[:, 0], gsrc[idx[:, 0]]

    per_id: dict[str, float] = defaultdict(lambda: -2.0)
    per_id_src: dict[str, int] = {}
    for lab, s, w in zip(labels, best.tolist(), who.tolist()):
        if s > per_id[lab]:
            per_id[lab] = s
            per_id_src[lab] = w

    n_ids = len(per_id)
    print(f"\n  identity max-similarity distribution ({n_ids:,} identities)")
    for lo, hi in [(0.9, 2.0), (0.7, 0.9), (0.5, 0.7), (0.4, 0.5),
                   (0.35, 0.4), (0.3, 0.35), (-2.0, 0.3)]:
        n = sum(1 for s in per_id.values() if lo <= s < hi)
        band = f">={lo:.2f}" if hi > 1 else f"{lo:.2f}-{hi:.2f}"
        print(f"    {band:>12s}  {n:>6,}  {n / n_ids * 100:5.1f}%")

    print("\n  threshold sensitivity")
    for t in (0.30, 0.35, 0.40, 0.45, 0.50):
        n = sum(1 for s in per_id.values() if s >= t)
        mark = "  <- primary (matches CASIA audit)" if abs(t - args.threshold) < 1e-9 else ""
        print(f"    >= {t:.2f} : exclude {n:>6,} ({n / n_ids * 100:5.1f}%){mark}")

    excluded = sorted(i for i, s in per_id.items() if s >= args.threshold)
    kept = sorted(i for i, s in per_id.items() if s < args.threshold)

    print("\n  which eval set the nearest neighbour came from (ALL identities)")
    tally = defaultdict(int)
    for lab in per_id:
        tally[names[per_id_src[lab]]] += 1
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"    {k:12s} {v:>6,}  {v / n_ids * 100:5.1f}%")

    if excluded:
        print("\n  ... and among the EXCLUDED identities only")
        tally_x = defaultdict(int)
        for lab in excluded:
            tally_x[names[per_id_src[lab]]] += 1
        for k, v in sorted(tally_x.items(), key=lambda kv: -kv[1]):
            print(f"    {k:12s} {v:>6,}  {v / len(excluded) * 100:5.1f}%")

    print(f"\n  EXCLUDE : {len(excluded):,} identities "
          f"({len(excluded) / n_ids * 100:.1f}%) at threshold {args.threshold}")
    print(f"  KEEP    : {len(kept):,} identities ({len(kept) / n_ids * 100:.1f}%)")
    print(f"  peak similarity anywhere : {float(best.max()):.4f}")

    if dims:
        a = np.array(dims)
        print(f"\n  native resolution of sampled QMUL images (pre-resize)")
        print(f"    median {int(np.median(a[:, 0]))}x{int(np.median(a[:, 1]))} px   "
              f"min {a[:, 0].min()}x{a[:, 1].min()}   max {a[:, 0].max()}x{a[:, 1].max()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": "QMUL-SurvFace training_set",
        "license": "research purposes only; images sourced from person re-identification "
                   "datasets, copyright with original owners (qmul-survface.github.io)",
        "model": args.model,
        "threshold": args.threshold,
        "per_identity_sampled": args.per_identity,
        "images_sampled": len(paths),
        "eval_sets": names,
        "gallery_embeddings": int(gallery.shape[0]),
        "identities_total": n_ids,
        "identities_excluded": len(excluded),
        "identities_kept": len(kept),
        "peak_similarity": round(float(best.max()), 4),
        "nearest_eval_set_tally": dict(tally),
        "threshold_sensitivity": {
            f"{t:.2f}": sum(1 for s in per_id.values() if s >= t)
            for t in (0.30, 0.35, 0.40, 0.45, 0.50)
        },
        "caveat": (
            "Degraded probes yield weaker embeddings, compressing cosine similarity "
            "downward for true matches too. A 0.40 threshold carried over from the "
            "clean-vs-clean CASIA audit is a LOOSER filter here, not a stricter one."
        ),
        "excluded_labels": excluded,
        "kept_labels": kept,
    }, indent=2))
    print(f"\n  Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
