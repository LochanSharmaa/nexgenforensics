#!/usr/bin/env python
"""
Item 36 — audit identity overlap between TRAINING and EVALUATION data.

    python backend/scripts/audit_train_eval_overlap.py --per-identity 2

WHY THIS IS EMBEDDING-BASED AND NOT NAME-BASED
-----------------------------------------------
Name matching is impossible here, established by inspection rather than
assumption:

  * The .bin evaluation packs carry NO identity metadata at all. Verified by
    scanning their pickle opcodes without executing them: the only strings in
    the byte stream are numpy dtype descriptors.
  * The training sets carry no names either. train.lst records original paths
    like /raid5data/dplearn/CASIA-WebFace/0000045/001.jpg — numeric folder IDs.
    MegaFace uses Flickr handles (100001044@N04_identity_0). Neither maps to a
    person's name without an external table that is not on disk.

So overlap can only be detected in embedding space: embed training images,
embed evaluation images, and look for pairs that score high enough to be the
same photograph or the same person.

WHAT THE THRESHOLDS MEAN
------------------------
Calibrated against this project's own measured distributions (BENCHMARKS.md):
genuine AgeDB pairs average ~0.49, impostors ~0.00, and the deployed decision
threshold is 0.2871.

  >= 0.90  near-duplicate  — almost certainly the SAME photograph, or a crop
                             of it. This is the unambiguous leak.
  >= 0.70  probable same identity — well above any genuine-pair average, so a
                             different photo of the same person.
  >= 0.2871 same-person by the system's own deployed rule. Reported for
                             context; at this level false positives are
                             expected and it is NOT evidence of leakage on its
                             own.

HONEST LIMITATION — READ BEFORE QUOTING THE RESULT
--------------------------------------------------
This samples K images per training identity. It can prove overlap EXISTS. It
cannot prove overlap is ABSENT: an identity whose sampled images happen not to
resemble the evaluation shots will be missed. The conclusion wording must say
"no overlap detected at this sampling depth", never "no overlap exists".
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "scripts"))

from nexgen_engine.benchmarks.verification import l2n  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"

TRAIN_SETS = {
    "faces_webface_112x112": _ROOT / "src_extracted/faces_webface_112x112/faces_webface_112x112",
    "faces_umd": _ROOT / "src_extracted/faces_umd/faces_umd",
    "faces_megafacetrain_112x112": _ROOT / "src_extracted/faces_megafacetrain_112x112/faces_megafacetrain_112x112",
}

_MAGIC = 0xCED7230A


def read_idx(path: Path) -> list[int]:
    """train.idx is plain text: '<record_id>\\t<byte_offset>' per line."""
    offsets = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    offsets.append(int(parts[1]))
                except ValueError:
                    continue
    return offsets


def read_record(fh, offset: int) -> tuple[int, bytes] | None:
    """Read one MXNet RecordIO record -> (label, image_bytes).

    Layout: magic(u32) | lrecord(u32) | payload | 4-byte padding.
    lrecord packs a 3-bit continuation flag in the top bits and the payload
    length in the low 29. The payload for image records begins with an
    IRHeader: flag(i32) label(f32) id(u64) id2(u64) = 24 bytes.
    """
    fh.seek(offset)
    head = fh.read(8)
    if len(head) < 8:
        return None
    magic, lrecord = struct.unpack("<II", head)
    if magic != _MAGIC:
        return None
    length = lrecord & 0x1FFFFFFF
    payload = fh.read(length)
    if len(payload) < 24:
        return None
    _flag, label = struct.unpack("<if", payload[:8])
    return int(label), payload[24:]


def sample_training(name: str, directory: Path, per_identity: int, max_images: int, seed: int):
    """Return (images_bgr, labels) sampled across identities."""
    rec, idx = directory / "train.rec", directory / "train.idx"
    if not rec.exists() or not idx.exists():
        return None, None
    offsets = read_idx(idx)
    if not offsets:
        return None, None

    rng = np.random.default_rng(seed)
    # Walk a shuffled subset of records, keeping at most per_identity per label.
    order = rng.permutation(len(offsets))
    kept: dict[int, int] = defaultdict(int)
    imgs, labels = [], []
    with open(rec, "rb") as fh:
        for i in order:
            if len(imgs) >= max_images:
                break
            r = read_record(fh, offsets[int(i)])
            if r is None:
                continue
            label, blob = r
            if kept[label] >= per_identity:
                continue
            im = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            if im.shape[:2] != (112, 112):
                im = cv2.resize(im, (112, 112))
            imgs.append(im)
            labels.append(label)
            kept[label] += 1
    return imgs, np.asarray(labels)


def embed(model, images, batch: int = 128) -> np.ndarray:
    out = np.empty((len(images), 512), dtype=np.float32)
    for i in range(0, len(images), batch):
        c = images[i : i + batch]
        out[i : i + len(c)] = np.asarray(model.get_feat([x for x in c])) + np.asarray(
            model.get_feat([x[:, ::-1] for x in c])
        )
        if (i // batch) % 20 == 0:
            print(f"      {min(i + batch, len(images))}/{len(images)}", end="\r", flush=True)
    return l2n(out.astype(np.float64))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--per-identity", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=25000,
                    help="cap per training set; sampling limit is reported in the output")
    ap.add_argument("--eval-datasets", nargs="+",
                    default=["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"])
    ap.add_argument("--train-sets", nargs="+", default=["faces_webface_112x112", "faces_umd"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/train_eval_overlap.json"))
    args = ap.parse_args()

    print("=" * 78)
    print("  Item 36 - train/eval identity overlap audit (embedding-based)")
    print("=" * 78)

    # --- evaluation side: reuse the cached benchmark embeddings ---
    ev = {}
    for ds in args.eval_datasets:
        p = CACHE / f"{ds}__{args.model}.npz"
        if not p.exists():
            print(f"  !! no cached embeddings for {ds}; EXCLUDED")
            continue
        d = np.load(p)
        ev[ds] = l2n((d["orig"] + d["flip"]).astype(np.float64))
        print(f"  eval  {ds:10s} {ev[ds].shape[0]:>7,} images")
    if not ev:
        print("no evaluation embeddings; run benchmark_verification.py first")
        return 1

    from benchmark_verification import load_recognizer  # noqa: PLC0415

    model = load_recognizer(args.model)

    results = {}
    overlap_found = False

    for ts in args.train_sets:
        d = TRAIN_SETS.get(ts)
        if d is None or not d.exists():
            print(f"\n  !! training set {ts} not found; skipped")
            continue
        print(f"\n  train {ts}: sampling <= {args.per_identity}/identity, "
              f"cap {args.max_images:,} ...")
        imgs, labels = sample_training(ts, d, args.per_identity, args.max_images, args.seed)
        if not imgs:
            print(f"     could not read records from {ts}; skipped")
            continue
        n_ids = len(set(labels.tolist()))
        print(f"     sampled {len(imgs):,} images across {n_ids:,} identities")
        temb = embed(model, imgs)

        per_eval = {}
        for ds, e in ev.items():
            # max similarity of each EVAL image to any sampled TRAINING image
            best = np.full(e.shape[0], -1.0)
            step = 2048
            for i in range(0, temb.shape[0], step):
                sims = e @ temb[i : i + step].T
                np.maximum(best, sims.max(axis=1), out=best)
            counts = {
                "near_duplicate_ge_0.90": int((best >= 0.90).sum()),
                "probable_same_id_ge_0.70": int((best >= 0.70).sum()),
                "above_deployed_thr_0.2871": int((best >= 0.2871).sum()),
                "eval_images": int(e.shape[0]),
                "max_similarity": float(best.max()),
                "mean_max_similarity": float(best.mean()),
            }
            per_eval[ds] = counts
            if counts["near_duplicate_ge_0.90"] > 0 or counts["probable_same_id_ge_0.70"] > 0:
                overlap_found = True
            print(f"     vs {ds:10s} max={counts['max_similarity']:.4f} "
                  f"| >=0.90: {counts['near_duplicate_ge_0.90']:>5} "
                  f"| >=0.70: {counts['probable_same_id_ge_0.70']:>5} "
                  f"| >=0.2871: {counts['above_deployed_thr_0.2871']:>6}/{counts['eval_images']}")

        results[ts] = {
            "sampled_images": len(imgs),
            "sampled_identities": n_ids,
            "per_identity_cap": args.per_identity,
            "per_eval": per_eval,
        }

    verdict = (
        "OVERLAP DETECTED - training must not proceed until resolved"
        if overlap_found
        else "No overlap detected AT THIS SAMPLING DEPTH (absence not proven)"
    )
    print("\n" + "=" * 78)
    print(f"  VERDICT: {verdict}")
    print("=" * 78)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "method": "embedding near-duplicate detection; name matching impossible "
                  "(training sets carry numeric IDs, .bin packs carry no labels)",
        "thresholds": {"near_duplicate": 0.90, "probable_same_identity": 0.70,
                       "deployed_decision": 0.2871},
        "overlap_found": overlap_found,
        "verdict": verdict,
        "limitation": "Sampling proves overlap EXISTS but cannot prove it is ABSENT.",
        "results": results,
    }, indent=2))
    print(f"\nWrote {out}")
    return 2 if overlap_found else 0


if __name__ == "__main__":
    raise SystemExit(main())
