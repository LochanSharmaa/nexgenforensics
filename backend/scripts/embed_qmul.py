#!/usr/bin/env python
"""G3 -- embed QMUL-SurvFace so the capacity measurement has a second corpus.

    python backend/scripts/embed_qmul.py

TinyFace is currently the ONLY dataset on which a valid capacity measurement
exists, and a programme-level conclusion should not rest on one corpus. QMUL is
the natural replication: independently collected, also natively low-resolution,
also identity-labelled (folder per identity), and drawn from real surveillance
rather than from web imagery.

Licence: QMUL-SurvFace is distributed for RESEARCH PURPOSES ONLY and its images
derive from person re-identification datasets whose upstream copyright holders
are not enumerated. Any result produced here inherits that restriction.

Outputs:
    runtime/benchmarks/embeddings/qmul_train__<model>.npz   emb, ids, files
"""

from __future__ import annotations

import argparse
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
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
QMUL_ROOTS = [
    Path("C:/Users/hello/Downloads/QMUL-SurvFace-v1/QMUL-SurvFace"),
    _ROOT / "src_extracted" / "QMUL-SurvFace",
]


def find_root() -> Path:
    for r in QMUL_ROOTS:
        if (r / "training_set").is_dir():
            return r
    raise SystemExit("QMUL-SurvFace training_set not found")


def decode(p: Path) -> np.ndarray:
    im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"failed to decode {p}")
    return cv2.resize(im, (112, 112))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = find_root()
    cache = CACHE / f"qmul_train__{args.model}.npz"
    if cache.exists() and not args.force:
        print(f"cache hit: {cache.name}")
        return 0
    CACHE.mkdir(parents=True, exist_ok=True)

    train = root / "training_set"
    files: list[Path] = []
    ids: list[int] = []
    for d in sorted(train.iterdir()):
        if not d.is_dir():
            continue
        try:
            ident = int(d.name)
        except ValueError:
            continue
        for f in sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")):
            files.append(f)
            ids.append(ident)
    print(f"QMUL training_set: {len(files):,} images / {len(set(ids)):,} identities")
    if not files:
        raise SystemExit("no images found")

    from benchmark_verification import load_recognizer  # noqa: PLC0415

    model = load_recognizer(args.model)  # raises unless CUDA binds

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
            if (i // args.batch_size) % 10 == 0 or done == len(files):
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  {done:,}/{len(files):,}  {rate:.0f} img/s  "
                      f"eta {(len(files) - done) / max(rate, 1e-6) / 60:.1f} min",
                      end="\r", flush=True)

    np.savez(
        cache,
        emb=out,
        ids=np.array(ids, dtype=np.int64),
        files=np.array([f.name for f in files]),
        flip_tta=True,
    )
    print(f"\nwrote {cache.name} ({cache.stat().st_size / 1024**2:.0f} MB) "
          f"in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
