#!/usr/bin/env python
"""G1 -- embed the TinyFace gallery so capacity can be measured against a real population.

    python backend/scripts/embed_tinyface_gallery.py

Two outputs, both cached to runtime/benchmarks/embeddings/:

  tinyface_distractors__<model>.npz   153,428 Gallery_Distractor images
  tinyface_labelled__<model>_tta.npz  8,171 labelled images, WITH flip-TTA

WHY THIS IS THE FIRST GPU TASK
------------------------------
Every capacity number this project holds rests on a 1,794-entry proxy gallery
built from the labelled set. That is not a reference population, it is a
placeholder, and docs/CAPACITY_VALIDATION.md names it as the top blocker. The
distractor set is 85x larger and is what the TinyFace protocol intends as the
open-set gallery, so embedding it converts the capacity framework from
"implemented" to "measured".

The second output exists because the current labelled cache stores a single
`emb` array with no flip-TTA, which is why our TinyFace figures are not
comparable to published ones. Re-embedding with TTA costs ~20 seconds and
removes a caveat from every table.

Resumable: if a cache exists it is reused unless --force. The distractor pass is
the long one; losing it to an interrupted run would be annoying rather than
catastrophic, but resumption is cheap so it is supported.
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
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_ROOT = _BACKEND.parent
TESTING = _ROOT / "src_extracted/tinyface/tinyface/Testing_Set"
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
ID_RE = re.compile(r"^(\d+)_\d+\.jpg$", re.IGNORECASE)


def decode(path: Path) -> np.ndarray:
    im = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"failed to decode {path}")
    # Same convention as benchmark_tinyface.py: plain resize to the recogniser's
    # 112x112 input. No detector -- TinyFace crops are already face-centred, and
    # running detection on a 20px crop mostly fails.
    return cv2.resize(im, (112, 112))


def embed_files(model, files: list[Path], batch_size: int, label: str) -> np.ndarray:
    """Flip-TTA embeddings, summed (not normalised) to match the cache convention."""
    out = np.empty((len(files), 512), dtype=np.float32)
    t0 = time.time()
    # Decode on worker threads while the GPU is busy: at ~800 img/s the JPEG
    # reads become the bottleneck otherwise.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(0, len(files), batch_size):
            chunk = files[i : i + batch_size]
            imgs = list(pool.map(decode, chunk))
            feats = np.asarray(model.get_feat(imgs), dtype=np.float32)
            flipped = np.asarray(model.get_feat([x[:, ::-1] for x in imgs]), dtype=np.float32)
            out[i : i + len(chunk)] = feats + flipped
            done = i + len(chunk)
            if (i // batch_size) % 10 == 0 or done == len(files):
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(files) - done) / max(rate, 1e-6)
                print(
                    f"    {label}: {done:,}/{len(files):,}  {rate:.0f} img/s  eta {eta / 60:.1f} min",
                    end="\r",
                    flush=True,
                )
    print(f"    {label}: {len(files):,} images in {time.time() - t0:.1f}s"
          f" ({2 * len(files) / max(time.time() - t0, 1e-6):.0f} fwd/s)" + " " * 20, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-distractors", action="store_true")
    args = ap.parse_args()

    if not TESTING.is_dir():
        print(f"TinyFace Testing_Set not found at {TESTING}")
        return 1
    CACHE.mkdir(parents=True, exist_ok=True)

    from benchmark_verification import load_recognizer  # noqa: PLC0415

    model = load_recognizer(args.model)  # asserts CUDA binding or raises

    # -- labelled set, with flip-TTA -----------------------------------------
    labelled = sorted(
        f
        for sub in ("Gallery_Match", "Probe")
        if (TESTING / sub).is_dir()
        for f in (TESTING / sub).glob("*.jpg")
        if ID_RE.match(f.name)
    )
    lab_cache = CACHE / f"tinyface_labelled__{args.model}_tta.npz"
    if lab_cache.exists() and not args.force:
        print(f"  labelled: cache hit ({lab_cache.name})")
    else:
        print(f"  labelled: {len(labelled):,} images")
        emb = embed_files(model, labelled, args.batch_size, "labelled")
        ids = np.array([int(ID_RE.match(f.name).group(1)) for f in labelled], dtype=np.int64)
        np.savez_compressed(
            lab_cache,
            emb=emb,
            ids=ids,
            files=np.array([f.name for f in labelled]),
            flip_tta=True,
        )
        print(f"  wrote {lab_cache.name}")

    # -- distractor gallery ---------------------------------------------------
    if not args.skip_distractors:
        dis_dir = TESTING / "Gallery_Distractor"
        dis_cache = CACHE / f"tinyface_distractors__{args.model}.npz"
        if dis_cache.exists() and not args.force:
            print(f"  distractors: cache hit ({dis_cache.name})")
        else:
            files = sorted(dis_dir.glob("*.jpg"))
            print(f"  distractors: {len(files):,} images")
            emb = embed_files(model, files, args.batch_size, "distractors")
            np.savez(  # uncompressed: 300 MB of float32 embeddings barely compress
                dis_cache,
                emb=emb,
                files=np.array([f.name for f in files]),
                flip_tta=True,
            )
            print(f"  wrote {dis_cache.name} ({dis_cache.stat().st_size / 1024**2:.0f} MB)")

    print("\nG1 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
