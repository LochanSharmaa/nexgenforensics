#!/usr/bin/env python
"""
Evaluate a fine-tuned checkpoint on the SAME protocol as the stock models.

Phase 2 of the brief forbids claiming any accuracy for the fine-tuned model
until it has been measured on this protocol. This script produces that number
so the checkpoint can be accepted or rejected on evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.benchmarks.verification import (  # noqa: E402
    decode_pack,
    evaluate_pairs,
    l2n,
    load_pack,
)
from nexgen_engine.training.train_pipeline import ResNet50ArcFaceBackbone  # noqa: E402

_ROOT = _BACKEND.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--datasets", nargs="+", default=["lfw", "agedb_30"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/finetuned.json"))
    args = ap.parse_args()

    from benchmark_verification import find_pack  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ResNet50ArcFaceBackbone(embedding_dim=ckpt.get("embedding_dim", 512), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"loaded {args.checkpoint}")
    print(f"  trained on {ckpt.get('train_samples')} samples / "
          f"{ckpt.get('num_classes')} identities, {ckpt.get('epochs')} epoch(s)")

    # Match the training-time preprocessing: ImageNet normalization on RGB.
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(images: np.ndarray) -> np.ndarray:
        out = np.empty((len(images), 512), dtype=np.float32)
        for i in range(0, len(images), args.batch_size):
            chunk = images[i : i + args.batch_size]
            rgb = chunk[..., ::-1].copy()  # BGR -> RGB
            t = torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float() / 255.0
            t = (t - mean) / std
            e = model(t) + model(torch.flip(t, dims=[3]))  # flip-TTA
            out[i : i + len(chunk)] = e.cpu().numpy()
        return out

    results = []
    for ds in args.datasets:
        bins, issame = load_pack(find_pack(ds))
        images = decode_pack(bins)
        emb = l2n(embed(images).astype(np.float64))
        r = evaluate_pairs(emb[0::2], emb[1::2], np.asarray(issame, bool), ds,
                           f"finetuned:{Path(args.checkpoint).name}")
        results.append(r)
        print(f"  {ds:10s} acc={r.accuracy_mean * 100:6.2f} +/- {r.accuracy_std * 100:4.2f}  "
              f"thr={r.threshold_mean:.4f}  AUC={r.auc:.5f}  EER={r.eer * 100:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        [{k: v for k, v in r.__dict__.items() if k != "folds"} for r in results], indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
