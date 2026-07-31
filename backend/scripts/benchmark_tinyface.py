#!/usr/bin/env python
"""
TinyFace 1:1 verification — degraded / surveillance-distance conditions.

This is the honest secondary number. TinyFace crops are native low-resolution
surveillance captures (median well under 40x40 px), which is what real
investigative footage looks like. The gap between this and the clean-benchmark
result in BENCHMARKS.md §2 is the operational risk, and the two must never be
averaged into a single headline figure.

Pairs are built from the TinyFace testing set, where the filename prefix is the
identity id (`<identity>_<n>.jpg`). Genuine pairs are two captures of one
identity; impostor pairs are two different identities drawn from the same pool,
so the difficulty comes from resolution rather than from an easy negative pool.

The same 10-fold protocol as the clean benchmark is used: the threshold is
fitted on 9 folds and applied to the held-out fold.
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

from nexgen_engine.benchmarks.verification import evaluate_pairs, l2n  # noqa: E402

_ROOT = _BACKEND.parent
TINYFACE = _ROOT / "src_extracted/tinyface/tinyface/Testing_Set"
ID_RE = re.compile(r"^(\d+)_\d+\.jpg$", re.IGNORECASE)


def collect() -> dict[str, list[Path]]:
    by_id: dict[str, list[Path]] = defaultdict(list)
    for sub in ("Gallery_Match", "Probe"):
        d = TINYFACE / sub
        if not d.is_dir():
            continue
        for p in d.glob("*.jpg"):
            m = ID_RE.match(p.name)
            if m:
                by_id[m.group(1)].append(p)
    return by_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glintr100")
    ap.add_argument("--pairs", type=int, default=3000, help="genuine (and impostor) pair count")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/tinyface.json"))
    args = ap.parse_args()

    from benchmark_verification import load_recognizer  # noqa: PLC0415

    by_id = collect()
    multi = {k: v for k, v in by_id.items() if len(v) >= 2}
    print(f"identities: {len(by_id)}  with >=2 captures: {len(multi)}")
    if len(multi) < 2:
        print("not enough identities for pairing")
        return 1

    rng = np.random.default_rng(args.seed)
    ids = sorted(multi)

    genuine, impostor = [], []
    while len(genuine) < args.pairs:
        i = ids[rng.integers(0, len(ids))]
        imgs = multi[i]
        a, b = rng.choice(len(imgs), 2, replace=False)
        genuine.append((imgs[a], imgs[b]))
    while len(impostor) < args.pairs:
        i, j = ids[rng.integers(0, len(ids))], ids[rng.integers(0, len(ids))]
        if i == j:
            continue
        impostor.append((multi[i][rng.integers(0, len(multi[i]))],
                         multi[j][rng.integers(0, len(multi[j]))]))

    # interleave so contiguous 10-fold slices each hold both classes
    pairs, labels = [], []
    for g, im in zip(genuine, impostor):
        pairs.append(g); labels.append(True)
        pairs.append(im); labels.append(False)
    labels = np.array(labels, dtype=bool)
    print(f"pairs: {len(pairs)} (genuine={int(labels.sum())}, impostor={int((~labels).sum())})")

    needed = sorted({p for pr in pairs for p in pr})
    print(f"unique images: {len(needed)}")

    # report the true resolution distribution -- this is the whole point
    dims = []
    for p in needed[:2000]:
        im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            dims.append(im.shape[:2])
    dims = np.array(dims)
    print(f"native resolution: median {int(np.median(dims[:, 0]))}x{int(np.median(dims[:, 1]))} "
          f"px, min {dims.min(0)[0]}x{dims.min(0)[1]}, max {dims.max(0)[0]}x{dims.max(0)[1]}")

    model = load_recognizer(args.model)
    embs = np.empty((len(needed), 512), dtype=np.float32)
    for i in range(0, len(needed), args.batch_size):
        chunk = needed[i : i + args.batch_size]
        imgs = []
        for p in chunk:
            im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError(f"failed to decode {p}")
            imgs.append(cv2.resize(im, (112, 112)))
        embs[i : i + len(chunk)] = (
            np.asarray(model.get_feat(imgs))
            + np.asarray(model.get_feat([x[:, ::-1] for x in imgs]))
        )
        if (i // args.batch_size) % 20 == 0:
            print(f"  {min(i + args.batch_size, len(needed))}/{len(needed)}", end="\r", flush=True)

    embs = l2n(embs.astype(np.float64))
    pos = {p: i for i, p in enumerate(needed)}
    a = np.stack([embs[pos[p[0]]] for p in pairs])
    b = np.stack([embs[pos[p[1]]] for p in pairs])

    r = evaluate_pairs(a, b, labels, "tinyface", f"single:{args.model}")
    print(f"\nTinyFace  acc={r.accuracy_mean * 100:.2f} +/- {r.accuracy_std * 100:.2f}  "
          f"thr={r.threshold_mean:.4f}  TAR@FAR1%={r.tar_at_far_1e2 * 100:.2f}  "
          f"TAR@FAR0.1%={r.tar_at_far_1e3 * 100:.2f}  AUC={r.auc:.5f}  EER={r.eer * 100:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in r.__dict__.items() if k != "folds"}
    payload["folds"] = [{"accuracy": f.accuracy, "threshold": f.threshold} for f in r.folds]
    payload["resolution_median_hw"] = [int(np.median(dims[:, 0])), int(np.median(dims[:, 1]))]
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
