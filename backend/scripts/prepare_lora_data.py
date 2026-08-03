#!/usr/bin/env python
"""Pre-align and cache the real low-resolution training corpora for LoRA.

    python backend/scripts/prepare_lora_data.py

WHY PRE-ALIGN RATHER THAN ALIGN IN THE TRAINING LOOP
-----------------------------------------------------
Alignment must match inference exactly or the fine-tune optimises a distribution
the deployed system never sees. Inference runs DFA aligner -> aligned 112 crop +
that image's own landmarks -> KP-RPE (defect 10 in MEASUREMENT_RECORD; skipping
it cost 51 points). So training must do the same.

Running the aligner inside the training loop would repeat that work every epoch
for no benefit -- alignment is deterministic and input-independent of the LoRA
weights. One pass, cached to a memmap, then epochs are pure ViT forward/backward.

CORPORA, both audited label-disjoint from their own test splits:
    TinyFace Training_Set    7,804 images / 2,570 identities
    QMUL     training_set  220,888 images / 5,319 identities

QMUL is capped per identity. It averages ~41 images per identity against
TinyFace's ~3, and most QMUL images within an identity are near-duplicate track
frames -- keeping all of them would let one corpus dominate the gradient while
adding little information. The cap is recorded in the manifest.

Outputs (runtime/lora/):
    crops.npy      uint8 memmap (N, 112, 112, 3), BGR, aligned
    ldmks.npy      float32 (N, 5, 2) in [0,1], the aligner's own predictions
    labels.npy     int64 (N,) contiguous class ids
    manifest.json  provenance, per-corpus counts, the cap, class mapping
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.models.cvlface_aligner import DfaAligner  # noqa: E402

_ROOT = _BACKEND.parent
OUT = _ROOT / "runtime" / "lora"
TF_TRAIN = _ROOT / "src_extracted/tinyface/tinyface/Training_Set"
QMUL_TRAIN = Path("C:/Users/hello/Downloads/QMUL-SurvFace-v1/QMUL-SurvFace/training_set")


def collect(cap_qmul: int) -> tuple[list[Path], list[str], dict]:
    files: list[Path] = []
    ids: list[str] = []
    stats = {}

    n0 = len(files)
    if TF_TRAIN.is_dir():
        for d in sorted(TF_TRAIN.iterdir()):
            if not d.is_dir():
                continue
            imgs = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
            for f in imgs:
                files.append(f)
                ids.append(f"tf:{d.name}")
    stats["tinyface"] = {"images": len(files) - n0,
                         "identities": len({i for i in ids if i.startswith("tf:")})}

    n0 = len(files)
    if QMUL_TRAIN.is_dir():
        rng = np.random.default_rng(0)
        for d in sorted(QMUL_TRAIN.iterdir()):
            if not d.is_dir():
                continue
            imgs = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
            if len(imgs) > cap_qmul:
                # Deterministic subsample: near-duplicate track frames add
                # little and would let QMUL dominate the gradient.
                imgs = [imgs[i] for i in sorted(rng.choice(len(imgs), cap_qmul, replace=False))]
            for f in imgs:
                files.append(f)
                ids.append(f"qm:{d.name}")
    stats["qmul"] = {"images": len(files) - n0,
                     "identities": len({i for i in ids if i.startswith("qm:")}),
                     "cap_per_identity": cap_qmul}
    return files, ids, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-qmul", type=int, default=24)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists() and not args.force:
        print("cache present; use --force to rebuild")
        return 0

    files, raw_ids, stats = collect(args.cap_qmul)
    if not files:
        raise SystemExit("no training images found")
    classes = sorted(set(raw_ids))
    cls_of = {c: i for i, c in enumerate(classes)}
    labels = np.array([cls_of[i] for i in raw_ids], dtype=np.int64)
    n = len(files)
    print(f"corpora: {stats}")
    print(f"total {n:,} images / {len(classes):,} identities")

    crops = np.lib.format.open_memmap(OUT / "crops.npy", mode="w+", dtype=np.uint8, shape=(n, 112, 112, 3))
    ldmks = np.empty((n, 5, 2), dtype=np.float32)

    aligner = DfaAligner(batch_size=args.batch)
    t0 = time.time()
    for i in range(0, n, args.batch):
        chunk = files[i : i + args.batch]
        imgs = []
        for f in chunk:
            im = cv2.imdecode(np.frombuffer(f.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            imgs.append(np.zeros((32, 32, 3), np.uint8) if im is None else im)
        al, ld = aligner.align(imgs)
        crops[i : i + len(chunk)] = np.stack(al)
        ldmks[i : i + len(chunk)] = ld
        done = i + len(chunk)
        if (i // args.batch) % 20 == 0 or done == n:
            r = done / max(time.time() - t0, 1e-6)
            print(f"  {done:,}/{n:,}  {r:.0f} img/s  eta {(n - done) / max(r, 1e-6) / 60:.1f} min",
                  end="\r", flush=True)
    crops.flush()
    np.save(OUT / "ldmks.npy", ldmks)
    np.save(OUT / "labels.npy", labels)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "n_images": int(n),
                "n_classes": len(classes),
                "corpora": stats,
                "alignment": "DFA mobilenet, same path as inference",
                "landmarks": "aligner predictions, [0,1] normalised, KP-RPE-ready",
                "contamination_audit": {
                    "tinyface_train_vs_test_label_overlap": 0,
                    "qmul_train_vs_test_label_overlap": 0,
                    "note": "raw label disjointness; identity disjointness rests on the protocol authors' splits",
                },
                "classes": classes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT.relative_to(_ROOT)} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
