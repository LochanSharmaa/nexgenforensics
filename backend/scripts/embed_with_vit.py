#!/usr/bin/env python
"""Embed the evaluation corpora with the CVLface ViT-B KP-RPE backbone.

    python backend/scripts/embed_with_vit.py --stage lfw       # preprocessing validation
    python backend/scripts/embed_with_vit.py --stage tinyface  # labelled + distractors
    python backend/scripts/embed_with_vit.py --stage qmul      # official ident set

MODEL KEY: `vit_kprpe_wf12m`. Caches use the exact naming scheme of the
existing pipeline (`tinyface_labelled__<key>_tta.npz`, ...), so
measure_capacity_official.py and friends run against the new backbone by
passing --model vit_kprpe_wf12m. Nothing else changes -- which is the point:
same protocols, same reference populations, same metrics, one variable.

THE LFW STAGE IS A GATE, NOT A RESULT. CVLface's model board reports ~99.8% on
LFW for this checkpoint. If our 10-fold accuracy lands there, the preprocessing
(BGR->RGB, [-1,1] normalisation, canonical keypoints for aligned crops) is
right. If it lands at 95%, something in the tensor path is wrong and every
downstream number would be garbage -- so tinyface/qmul refuse to run until an
LFW validation artifact exists and passes 99.5%.

FLIP-TTA: applied as embed(img) + embed(mirror), matching the incumbent's
convention so cross-backbone comparisons hold the TTA choice constant. The
mirror's keypoints are the mirrored canonical points (eye order swapped), not
the originals -- KP-RPE conditions on landmark POSITIONS, and a mirrored image
with unmirrored landmarks is a misalignment we would be injecting ourselves.
"""

from __future__ import annotations

import argparse
import json
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

from nexgen_engine.models.cvlface_backbone import ARCFACE_5PTS, CvlfaceViTKprpe  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
OUT = _ROOT / "runtime" / "forensics"
MODEL_KEY = "vit_kprpe_wf12m"
TF = _ROOT / "src_extracted/tinyface/tinyface/Testing_Set"
QMUL_ID = Path("C:/Users/hello/Downloads/QMUL-SurvFace-v1/QMUL-SurvFace/Face_Identification_Test_Set")
TF_RE = re.compile(r"^(\d+)_\d+\.jpg$", re.IGNORECASE)
QM_RE = re.compile(r"^(\d+)_cam", re.IGNORECASE)

#: Canonical keypoints for a horizontally mirrored crop, in the same [0, 1]
#: normalised units as ARCFACE_5PTS: x -> 1 - x, and the left/right eye and
#: mouth-corner labels swap.
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


def embed_tta(model: CvlfaceViTKprpe, imgs: list[np.ndarray]) -> np.ndarray:
    n = len(imgs)
    kp = np.repeat(ARCFACE_5PTS[None], n, axis=0)
    kpm = np.repeat(MIRROR_5PTS[None], n, axis=0)
    return model.get_feat(imgs, kp) + model.get_feat([im[:, ::-1] for im in imgs], kpm)


def decode_resize(p: Path) -> np.ndarray:
    im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"decode failed: {p}")
    return cv2.resize(im, (112, 112))


def run_files(model, files: list[Path], label: str, batch: int) -> np.ndarray:
    out = np.empty((len(files), 512), dtype=np.float32)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(0, len(files), batch):
            chunk = files[i : i + batch]
            imgs = list(pool.map(decode_resize, chunk))
            out[i : i + len(chunk)] = embed_tta(model, imgs)
            done = i + len(chunk)
            if (i // batch) % 10 == 0 or done == len(files):
                r = done / max(time.time() - t0, 1e-6)
                print(f"    {label}: {done:,}/{len(files):,}  {r:.0f} img/s  "
                      f"eta {(len(files) - done) / max(r, 1e-6) / 60:.1f} min",
                      end="\r", flush=True)
    print()
    return out


def lfw_gate_passed() -> bool:
    p = OUT / f"lfw_validation_{MODEL_KEY}.json"
    if not p.exists():
        return False
    return bool(json.loads(p.read_text(encoding="utf-8")).get("gate_passed"))


def stage_lfw(model, batch: int) -> int:
    from nexgen_engine.benchmarks.verification import decode_pack, evaluate_pairs, l2n, load_pack

    pack = _ROOT / "src_extracted/faces_webface_112x112/faces_webface_112x112/lfw.bin"
    bins, issame = load_pack(pack)
    images = decode_pack(bins)
    print(f"LFW: {len(images):,} images")

    out = np.empty((len(images), 512), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(images), batch):
        chunk = images[i : i + batch]
        out[i : i + len(chunk)] = embed_tta(model, list(chunk))
        if (i // batch) % 10 == 0:
            r = (i + len(chunk)) / max(time.time() - t0, 1e-6)
            print(f"    {i + len(chunk):,}/{len(images):,}  {r:.0f} img/s", end="\r", flush=True)
    print()

    emb = l2n(out.astype(np.float64))
    r = evaluate_pairs(emb[0::2], emb[1::2], np.asarray(issame, dtype=bool), "lfw", f"single:{MODEL_KEY}")
    gate = r.accuracy_mean >= 0.995
    print(f"LFW 10-fold accuracy: {r.accuracy_mean * 100:.2f}% +/- {r.accuracy_std * 100:.2f} "
          f"(TAR@FAR=0.1% {r.tar_at_far_1e3 * 100:.2f}%)")
    print(f"GATE ({'>=99.5% required'}): {'PASSED' if gate else 'FAILED -- preprocessing is wrong, do not proceed'}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"lfw_validation_{MODEL_KEY}.json").write_text(
        json.dumps(
            {
                "model": MODEL_KEY,
                "accuracy_mean": r.accuracy_mean,
                "accuracy_std": r.accuracy_std,
                "tar_at_far_1e3": r.tar_at_far_1e3,
                "reference": "CVLface model board reports ~99.8 LFW for this checkpoint (uncited caveat applies)",
                "gate_threshold": 0.995,
                "gate_passed": gate,
                "flip_tta": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if gate else 1


def stage_tinyface(model, batch: int) -> int:
    if not lfw_gate_passed():
        raise SystemExit("LFW validation gate not passed -- run --stage lfw first")
    labelled = sorted(
        f for sub in ("Gallery_Match", "Probe") if (TF / sub).is_dir()
        for f in (TF / sub).glob("*.jpg") if TF_RE.match(f.name)
    )
    lab_cache = CACHE / f"tinyface_labelled__{MODEL_KEY}_tta.npz"
    if not lab_cache.exists():
        emb = run_files(model, labelled, "labelled", batch)
        np.savez_compressed(
            lab_cache, emb=emb,
            ids=np.array([int(TF_RE.match(f.name).group(1)) for f in labelled], dtype=np.int64),
            files=np.array([f.name for f in labelled]), flip_tta=True,
        )
        print(f"  wrote {lab_cache.name}")
    dis_cache = CACHE / f"tinyface_distractors__{MODEL_KEY}.npz"
    if not dis_cache.exists():
        files = sorted((TF / "Gallery_Distractor").glob("*.jpg"))
        emb = run_files(model, files, "distractors", batch)
        np.savez(dis_cache, emb=emb, files=np.array([f.name for f in files]), flip_tta=True)
        print(f"  wrote {dis_cache.name}")
    return 0


def stage_qmul(model, batch: int) -> int:
    if not lfw_gate_passed():
        raise SystemExit("LFW validation gate not passed -- run --stage lfw first")
    cache = CACHE / f"qmul_ident__{MODEL_KEY}.npz"
    if cache.exists():
        print(f"cache hit: {cache.name}")
        return 0
    files, ids, split = [], [], []
    for sub, tag in (("gallery", "gallery"), ("mated_probe", "mated"), ("unmated_probe", "unmated")):
        for f in sorted((QMUL_ID / sub).glob("*.jpg")):
            m = QM_RE.match(f.name)
            if m:
                files.append(f)
                ids.append(-int(m.group(1)) if tag == "unmated" else int(m.group(1)))
                split.append(tag)
    print(f"QMUL ident: {len(files):,} images")
    emb = run_files(model, files, "qmul_ident", batch)
    np.savez(cache, emb=emb, ids=np.array(ids, dtype=np.int64), split=np.array(split),
             files=np.array([f.name for f in files]), flip_tta=True)
    print(f"  wrote {cache.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["lfw", "tinyface", "qmul"])
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    model = CvlfaceViTKprpe(batch_size=args.batch)
    print(f"backbone: {model.provider_label}")
    return {"lfw": stage_lfw, "tinyface": stage_tinyface, "qmul": stage_qmul}[args.stage](model, args.batch)


if __name__ == "__main__":
    raise SystemExit(main())
