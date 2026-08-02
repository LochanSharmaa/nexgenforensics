#!/usr/bin/env python
"""Embed QMUL-SurvFace's OFFICIAL open-set identification test set.

    python backend/scripts/embed_qmul_ident.py

WHY THIS IS NEEDED AND TINYFACE IS NOT ENOUGH
---------------------------------------------
The corrected TinyFace run reports `unmated = 0`. Every identity in TinyFace's
Probe set is also in Gallery_Match, so the corpus cannot measure the one thing
open-set recognition exists to measure: what the system does when the person is
NOT enrolled. Any TPIR@FPIR computed on it would be measuring nothing.

QMUL ships the missing piece:

    gallery         60,294 images   enrolled identities
    mated_probe     60,423 images   probes whose identity IS enrolled
    unmated_probe  121,736 images   probes whose identity is NOT enrolled

That third split is a genuine non-mate population defined by the protocol
authors, which makes it both the open-set probe set AND an identity-disjoint
reference population for capacity -- the property two earlier attempts failed to
obtain by construction.

Identity is parsed from the filename (`<id>_cam<n>_<m>.jpg`). Unmated identities
are namespaced negatively so they can never collide with gallery identities.
"""

from __future__ import annotations

import argparse
import re
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
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
ROOT = Path("C:/Users/hello/Downloads/QMUL-SurvFace-v1/QMUL-SurvFace/Face_Identification_Test_Set")
ID_RE = re.compile(r"^(\d+)_cam", re.IGNORECASE)


def decode(p: Path) -> np.ndarray:
    im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"decode failed: {p}")
    return cv2.resize(im, (112, 112))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cache = CACHE / f"qmul_ident__{args.model}.npz"
    if cache.exists() and not args.force:
        print(f"cache hit: {cache.name}")
        return 0
    if not ROOT.is_dir():
        raise SystemExit(f"not found: {ROOT}")
    CACHE.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    ids: list[int] = []
    split: list[str] = []
    for sub, tag in (("gallery", "gallery"), ("mated_probe", "mated"), ("unmated_probe", "unmated")):
        d = ROOT / sub
        n0 = len(files)
        for f in sorted(d.glob("*.jpg")):
            m = ID_RE.match(f.name)
            if not m:
                continue
            ident = int(m.group(1))
            files.append(f)
            # Namespace unmated identities negatively: they are non-mates by
            # protocol, and a numeric collision with a gallery id would silently
            # turn a true non-mate into a false genuine pair.
            ids.append(-ident if tag == "unmated" else ident)
            split.append(tag)
        print(f"  {tag}: {len(files)-n0:,} images")

    print(f"total {len(files):,}")
    from benchmark_verification import load_recognizer  # noqa: PLC0415

    model = load_recognizer(args.model)
    out = np.empty((len(files), 512), dtype=np.float32)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(0, len(files), args.batch_size):
            chunk = files[i : i + args.batch_size]
            imgs = list(pool.map(decode, chunk))
            out[i : i + len(chunk)] = np.asarray(model.get_feat(imgs), dtype=np.float32) + np.asarray(
                model.get_feat([x[:, ::-1] for x in imgs]), dtype=np.float32
            )
            done = i + len(chunk)
            if (i // args.batch_size) % 20 == 0 or done == len(files):
                r = done / max(time.time() - t0, 1e-6)
                print(f"    {done:,}/{len(files):,}  {r:.0f} img/s  "
                      f"eta {(len(files)-done)/max(r,1e-6)/60:.1f} min", end="\r", flush=True)

    np.savez(
        cache, emb=out, ids=np.array(ids, dtype=np.int64),
        split=np.array(split), files=np.array([f.name for f in files]), flip_tta=True,
    )
    print(f"\nwrote {cache.name} ({cache.stat().st_size/1024**2:.0f} MB) in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
