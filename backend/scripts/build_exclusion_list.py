#!/usr/bin/env python
"""
Phase 6 step 3 — build the training-identity EXCLUSION LIST.

    python backend/scripts/build_exclusion_list.py --per-identity 10

audit_train_eval_overlap.py answers "is there overlap?" from the EVAL side: for
each evaluation image, how similar is the nearest training image. That proves
contamination exists but cannot be acted on, because it never names the
training identities responsible.

This answers the other direction: for each TRAINING identity, how similar is its
nearest evaluation image. Identities above the threshold are written to an
exclusion list so a fine-tune can drop them.

THRESHOLD CHOICE — DELIBERATELY CONSERVATIVE
--------------------------------------------
Default 0.40, which is BELOW the ~0.49 mean of genuine same-person pairs in this
system. That is intentional and asymmetric:

  * Excluding a clean identity costs a little training data. Cheap.
  * Keeping a contaminated one means the model memorises a face it will later
    be tested on, and every downstream accuracy number becomes unfalsifiable.

So the threshold errs toward over-exclusion. A tighter value (0.70, the
"probable same identity" line used in the audit) would leave same-person pairs
scoring 0.40-0.70 in the training set, and those are exactly the cross-age and
cross-pose pairs the benchmarks are built from.

Uses faiss IndexFlatIP for the nearest-neighbour search -- exact, no recall
loss, and the one place BENCHMARKS.md §7d found it genuinely worth using.
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
sys.path.insert(0, str(_BACKEND / "scripts"))

from audit_train_eval_overlap import TRAIN_SETS, embed, sample_training  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"


def eval_gallery(model: str) -> tuple[np.ndarray, list[str]]:
    """Every cached evaluation embedding, pooled into one matrix."""
    parts, names = [], []
    for p in sorted(CACHE.glob(f"*__{model}.npz")):
        d = np.load(p)
        e = (d["orig"] + d["flip"]).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        parts.append(e)
        names.append(p.name.split("__")[0])
    if not parts:
        raise SystemExit("no cached eval embeddings; run benchmark_verification.py first")
    return np.concatenate(parts, axis=0), names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-set", default="faces_webface_112x112", choices=list(TRAIN_SETS))
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--per-identity", type=int, default=10)
    ap.add_argument("--max-images", type=int, default=120000)
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/exclusion_list.json"))
    args = ap.parse_args()

    import faiss
    from benchmark_verification import load_recognizer

    print("=" * 78)
    print("  Phase 6 step 3 - training-identity exclusion list")
    print(f"  threshold {args.threshold} (below the ~0.49 genuine mean: errs toward")
    print("  over-exclusion, because keeping a contaminated identity is worse)")
    print("=" * 78)

    gallery, ev_names = eval_gallery(args.model)
    print(f"\n  eval gallery : {gallery.shape[0]:,} embeddings from {ev_names}")

    d = TRAIN_SETS[args.train_set]
    print(f"  sampling {args.train_set} at <= {args.per_identity}/identity ...")
    imgs, labels = sample_training(args.train_set, d, args.per_identity, args.max_images, args.seed)
    if not imgs:
        print("  could not read training records")
        return 1
    ids = sorted(set(labels.tolist()))
    print(f"  sampled {len(imgs):,} images across {len(ids):,} identities")

    model = load_recognizer(args.model)
    temb = embed(model, imgs).astype(np.float32)
    print(f"  embedded {temb.shape[0]:,} training images")

    # Exact nearest-neighbour: recall 1.000 by construction. No approximation
    # here -- a missed match would silently leave a contaminated identity in.
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)
    sims, _ = index.search(np.ascontiguousarray(temb), 1)
    best = sims[:, 0]

    per_identity: dict[int, float] = defaultdict(lambda: -1.0)
    for lab, s in zip(labels.tolist(), best.tolist()):
        if s > per_identity[lab]:
            per_identity[lab] = s

    excluded = sorted(i for i, s in per_identity.items() if s >= args.threshold)
    kept = sorted(i for i, s in per_identity.items() if s < args.threshold)

    print(f"\n  {'identity max-similarity distribution':38s}")
    for lo, hi in [(0.9, 2.0), (0.7, 0.9), (0.5, 0.7), (0.4, 0.5), (0.3, 0.4), (-2.0, 0.3)]:
        n = sum(1 for s in per_identity.values() if lo <= s < hi)
        band = f">={lo:.1f}" if hi > 1 else f"{lo:.1f}-{hi:.1f}"
        print(f"    {band:>10s}  {n:>6,} identities  {n / len(per_identity) * 100:5.1f}%")

    print(f"\n  EXCLUDE : {len(excluded):,} identities ({len(excluded) / len(ids) * 100:.1f}%)")
    print(f"  KEEP    : {len(kept):,} identities ({len(kept) / len(ids) * 100:.1f}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "train_set": args.train_set,
        "model": args.model,
        "threshold": args.threshold,
        "per_identity_sampled": args.per_identity,
        "images_sampled": len(imgs),
        "identities_total": len(ids),
        "identities_excluded": len(excluded),
        "identities_kept": len(kept),
        "eval_sets": ev_names,
        "limitation": (
            "Sampling. An identity whose sampled images happen not to resemble "
            "the eval shots is NOT excluded. This list is a floor: the true "
            "contaminated set is at least this large."
        ),
        "excluded_labels": excluded,
        "kept_labels": kept,
    }, indent=2))
    print(f"\n  Wrote {out}")
    print("\n  NOTE: this list is a FLOOR. Sampling cannot prove an identity clean,")
    print("  only that its sampled images did not match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
