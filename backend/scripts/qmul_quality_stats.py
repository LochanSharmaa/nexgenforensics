#!/usr/bin/env python
"""
QMUL-SurvFace image quality / resolution characterisation.

    python backend/scripts/qmul_quality_stats.py

Step 3 of the QMUL evaluation: confirm the dataset is genuinely NATIVE
low-resolution capture, and not something that needs further processing before
it is usable as degraded-condition training data.

The failed synthetic attempt (BENCHMARKS.md §6d) came from an untested
assumption about what degraded imagery looks like. So this does not assume:
it measures QMUL against TinyFace (the benchmark it is meant to help) and
against CASIA (the clean training data it is meant to complement), on the same
three axes, and reports whether QMUL actually sits where it is supposed to.

  resolution   native pixel dimensions before any resize
  sharpness    variance of the Laplacian -- high-frequency energy. Real
               low-resolution capture is genuinely detail-poor; an upsampled
               clean image is smooth but not equivalently detail-poor.
  detectability whether SCRFD can find a face at NATIVE resolution. This is the
               practical question: if the production detector cannot locate a
               face, the image cannot enter the pipeline without pre-processing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "scripts"))

_ROOT = _BACKEND.parent
QMUL = Path("C:/Users/hello/Downloads/QMUL-SurvFace-v1/QMUL-SurvFace/training_set")
TINYFACE = _ROOT / "src_extracted/tinyface/tinyface/Testing_Set"


def sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def describe(name: str, dims: np.ndarray, sharp: list[float]) -> dict:
    h, w = dims[:, 0], dims[:, 1]
    px = h.astype(np.float64) * w
    d = {
        "images": int(len(dims)),
        "height_px": {q: int(np.percentile(h, q)) for q in (5, 25, 50, 75, 95)},
        "width_px": {q: int(np.percentile(w, q)) for q in (5, 25, 50, 75, 95)},
        "min_hw": [int(h.min()), int(w.min())],
        "max_hw": [int(h.max()), int(w.max())],
        "median_area_px": int(np.median(px)),
        "sharpness_median": round(float(np.median(sharp)), 2) if sharp else None,
        "pct_under_32px_high": round(float((h < 32).mean() * 100), 1),
        "pct_under_64px_high": round(float((h < 64).mean() * 100), 1),
    }
    print(f"\n  {name}")
    print(f"    images            {d['images']:,}")
    print(f"    height px  p5/p25/p50/p75/p95  "
          f"{d['height_px'][5]}/{d['height_px'][25]}/{d['height_px'][50]}/"
          f"{d['height_px'][75]}/{d['height_px'][95]}")
    print(f"    width  px  p5/p25/p50/p75/p95  "
          f"{d['width_px'][5]}/{d['width_px'][25]}/{d['width_px'][50]}/"
          f"{d['width_px'][75]}/{d['width_px'][95]}")
    print(f"    range             {d['min_hw'][0]}x{d['min_hw'][1]} .. "
          f"{d['max_hw'][0]}x{d['max_hw'][1]}")
    print(f"    under 32px high   {d['pct_under_32px_high']}%")
    print(f"    under 64px high   {d['pct_under_64px_high']}%")
    print(f"    sharpness (median var-of-Laplacian)  {d['sharpness_median']}")
    return d


def scan(files: list[Path], sharp_n: int, rng) -> tuple[np.ndarray, list[float]]:
    dims, sharp = [], []
    pick = set(rng.choice(len(files), min(sharp_n, len(files)), replace=False).tolist())
    for i, f in enumerate(files):
        im = cv2.imdecode(np.frombuffer(f.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        dims.append(im.shape[:2])
        if i in pick:
            sharp.append(sharpness(im))
        if i % 5000 == 0:
            print(f"      {i:,}/{len(files):,}", end="\r", flush=True)
    return np.array(dims), sharp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmul-sample", type=int, default=40000)
    ap.add_argument("--sharp-sample", type=int, default=4000)
    ap.add_argument("--detect-sample", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/qmul_quality.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print("=" * 78)
    print("  QMUL-SurvFace quality characterisation (step 3)")
    print("=" * 78)

    report = {}

    qfiles = sorted(QMUL.rglob("*.jpg"))
    print(f"\n  QMUL training images on disk: {len(qfiles):,}")
    if len(qfiles) > args.qmul_sample:
        sel = rng.choice(len(qfiles), args.qmul_sample, replace=False)
        qsample = [qfiles[i] for i in sel]
    else:
        qsample = qfiles
    print(f"  scanning {len(qsample):,} ...")
    qd, qs = scan(qsample, args.sharp_sample, rng)
    report["qmul_survface"] = describe("QMUL-SurvFace (training_set)", qd, qs)

    tfiles = []
    for sub in ("Gallery_Match", "Probe"):
        d = TINYFACE / sub
        if d.is_dir():
            tfiles += sorted(d.glob("*.jpg"))
    if tfiles:
        print(f"\n  scanning TinyFace labelled set ({len(tfiles):,}) ...")
        td, ts = scan(tfiles, args.sharp_sample, rng)
        report["tinyface"] = describe("TinyFace (Gallery_Match + Probe) - THE TARGET", td, ts)

    # CASIA for a clean-side reference point. Records are 112x112 by
    # construction, so only sharpness is informative there.
    try:
        from audit_train_eval_overlap import TRAIN_SETS, read_idx, read_record

        d = TRAIN_SETS["faces_webface_112x112"]
        offs = read_idx(d / "train.idx")
        sel = rng.choice(len(offs), min(args.sharp_sample, len(offs)), replace=False)
        cs, cd = [], []
        with open(d / "train.rec", "rb") as fh:
            for i in sel:
                r = read_record(fh, offs[int(i)])
                if r is None:
                    continue
                im = cv2.imdecode(np.frombuffer(r[1], np.uint8), cv2.IMREAD_COLOR)
                if im is not None:
                    cd.append(im.shape[:2])
                    cs.append(sharpness(im))
        if cd:
            report["casia_clean_reference"] = describe(
                "CASIA-WebFace (clean reference, pre-cropped to 112x112)", np.array(cd), cs)
    except Exception as exc:
        print(f"  CASIA reference unavailable: {exc}")

    # ---- detectability at NATIVE resolution ----
    print("\n  SCRFD detectability at NATIVE resolution")
    print("  (can the production detector find a face without pre-processing?)")
    try:
        from nexgen_engine.config import EngineConfig
        from nexgen_engine.inference.pipeline import FacialRecognitionPipeline

        pipe = FacialRecognitionPipeline(EngineConfig())
        app = pipe.runtime._analysis_app
        det = {}
        for label, files in (("qmul", qsample), ("tinyface", tfiles)):
            if not files:
                continue
            sel = rng.choice(len(files), min(args.detect_sample, len(files)), replace=False)
            hit = 0
            n = 0
            for i in sel:
                im = cv2.imdecode(np.frombuffer(files[int(i)].read_bytes(), np.uint8),
                                  cv2.IMREAD_COLOR)
                if im is None:
                    continue
                n += 1
                try:
                    if app.get(im):
                        hit += 1
                except Exception:
                    pass
            det[label] = {"sampled": n, "detected": hit,
                          "detect_rate_pct": round(hit / max(n, 1) * 100, 1)}
            print(f"    {label:10s} {hit}/{n} = {det[label]['detect_rate_pct']}%")
        report["native_detectability"] = det
    except Exception as exc:
        print(f"    unavailable: {exc}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
