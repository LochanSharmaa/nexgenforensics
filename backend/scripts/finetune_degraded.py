#!/usr/bin/env python
"""
Phase 6 step 4 — fine-tune for DEGRADED imagery, from ArcFace weights,
on a decontaminated subset.

    python backend/scripts/finetune_degraded.py --steps 2000

Everything the previous attempt got wrong is addressed here:

  item 36  training identities come from exclusion_list.json — the 692
           identities that matched an evaluation image are dropped.
  item 37  every batch mixes clean and SIMULATED DEGRADED crops. This is the
           whole point: the target is TinyFace-grade imagery (median 32x32),
           and a model fine-tuned only on clean faces cannot improve it.
  item 38  a held-out validation split, disjoint by IDENTITY, tracked each eval.
  item 39  initialised from the deployed ArcFace ONNX via onnx2torch — NOT
           ImageNet. This is what made the earlier run score at chance.
  item 40  hard-negative aware: ArcFace's angular margin already concentrates
           gradient on hard samples, and the degraded view of an image is by
           construction the hard positive of its clean counterpart.
  item 41  early stopping on validation loss, not a fixed epoch count.

WHAT IS DELIBERATELY NOT CLAIMED
--------------------------------
~9,880 identities is still ~36x fewer than glintr100 saw. The realistic goal
is a modest gain on degraded imagery WITHOUT regressing clean accuracy, not a
new state of the art. A run that improves nothing is a valid result and must be
reported as such — see BENCHMARKS.md §6b.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "scripts"))

from audit_train_eval_overlap import TRAIN_SETS, read_idx, read_record  # noqa: E402

_ROOT = _BACKEND.parent


def degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate surveillance capture on a 112x112 crop.

    Modelled on what TinyFace actually is: a small native capture that has been
    upsampled. Downscale to a random small edge, then back up -- that destroys
    the same high-frequency detail a distant camera never recorded. Blur and
    JPEG are added because real footage is compressed and imperfectly focused.
    """
    h = int(rng.integers(16, 48))  # TinyFace median is 32px
    small = cv2.resize(img, (h, h), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (112, 112), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.7:
        q = int(rng.integers(20, 60))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def to_tensor(batch: list[np.ndarray], device) -> torch.Tensor:
    """BGR uint8 HWC -> the normalisation the ArcFace graph expects."""
    a = np.stack(batch).astype(np.float32)
    a = a[..., ::-1].copy()                     # BGR -> RGB
    a = (a - 127.5) / 127.5                     # insightface input_mean/std
    t = torch.from_numpy(a).permute(0, 3, 1, 2)
    return t.to(device, non_blocking=True)


def load_backbone(device):
    """ArcFace weights as a trainable module. Item 39."""
    import onnx
    import onnx2torch

    p = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
    proto = onnx.load_model(io.BytesIO(p.read_bytes()))
    return onnx2torch.convert(proto).to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--per-identity", type=int, default=12)
    ap.add_argument("--val-identities", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=300,
                    help="steps with the backbone frozen while the head converges")
    ap.add_argument("--degrade-prob", type=float, default=0.5)
    ap.add_argument("--exclusion", default=str(_ROOT / "runtime/benchmarks/exclusion_list.json"))
    ap.add_argument("--out-dir", default=str(_ROOT / "runtime/checkpoints"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    print("=" * 78)
    print(f"  Phase 6 step 4 - degraded fine-tune from ArcFace init on {device}")
    print("=" * 78)

    excl = json.loads(Path(args.exclusion).read_text())
    keep = set(excl["kept_labels"])
    print(f"  exclusion list: dropping {excl['identities_excluded']:,} contaminated "
          f"identities, keeping {len(keep):,}")

    # ---- load clean training images ----
    d = TRAIN_SETS["faces_webface_112x112"]
    offsets = read_idx(d / "train.idx")
    order = rng.permutation(len(offsets))
    kept_count: dict[int, int] = {}
    imgs: list[np.ndarray] = []
    labels: list[int] = []
    print("  reading records ...")
    with open(d / "train.rec", "rb") as fh:
        for i in order:
            r = read_record(fh, offsets[int(i)])
            if r is None:
                continue
            lab, blob = r
            if lab not in keep or kept_count.get(lab, 0) >= args.per_identity:
                continue
            im = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            if im.shape[:2] != (112, 112):
                im = cv2.resize(im, (112, 112))
            imgs.append(im)
            labels.append(lab)
            kept_count[lab] = kept_count.get(lab, 0) + 1

    labels = np.asarray(labels)
    uniq = sorted(set(labels.tolist()))
    print(f"  loaded {len(imgs):,} images across {len(uniq):,} clean identities")

    # ---- identity-disjoint train/val split (item 38) ----
    val_ids = set(rng.choice(uniq, min(args.val_identities, len(uniq) // 5), replace=False).tolist())
    tr_idx = np.array([i for i, l in enumerate(labels) if l not in val_ids])
    va_idx = np.array([i for i, l in enumerate(labels) if l in val_ids])
    remap = {l: k for k, l in enumerate(sorted(set(labels[tr_idx].tolist())))}
    n_classes = len(remap)
    print(f"  train {len(tr_idx):,} imgs / {n_classes:,} identities   "
          f"val {len(va_idx):,} imgs / {len(val_ids):,} identities (DISJOINT)")

    # ---- model ----
    backbone = load_backbone(device)
    from nexgen_engine.training.arcface_loss import ArcFaceLoss

    head = ArcFaceLoss(in_features=512, out_features=n_classes, s=64.0, m=0.50).to(device)
    opt = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=5e-4
    )
    print(f"  backbone params {sum(p.numel() for p in backbone.parameters()):,} "
          f"(ArcFace init, NOT ImageNet)")

    def make_batch(pool: np.ndarray, train: bool):
        pick = rng.choice(pool, args.batch_size, replace=False)
        batch, ys = [], []
        for i in pick:
            im = imgs[i]
            if rng.random() < args.degrade_prob:
                im = degrade(im, rng)
            batch.append(im)
            ys.append(remap.get(int(labels[i]), 0))
        return to_tensor(batch, device), torch.tensor(ys, device=device)

    @torch.no_grad()
    def validate() -> float:
        """Val identities are unseen classes, so classification loss is not
        meaningful. Measure what actually matters instead: the margin between
        a DEGRADED view and its own CLEAN counterpart versus a different
        identity. Higher is better."""
        backbone.eval()
        gaps = []
        for _ in range(12):
            pick = rng.choice(va_idx, args.batch_size, replace=False)
            clean = to_tensor([imgs[i] for i in pick], device)
            deg = to_tensor([degrade(imgs[i], rng) for i in pick], device)
            ec = torch.nn.functional.normalize(backbone(clean), dim=1)
            ed = torch.nn.functional.normalize(backbone(deg), dim=1)
            sim = ec @ ed.T
            pos = sim.diag()
            neg = (sim - torch.eye(len(pick), device=device) * 2).max(dim=1).values
            gaps.append((pos - neg).mean().item())
        backbone.train()
        return float(np.mean(gaps))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, bad, best_step = -1e9, 0, 0
    ck = out_dir / "arcface_degraded_v1.pt"

    base = validate()
    print(f"\n  baseline val margin (before any training): {base:+.4f}")
    print(f"\n  {'step':>6} {'loss':>9} {'val margin':>12} {'note'}")

    # HEAD WARM-UP — the backbone is FROZEN for the first `warmup` steps.
    #
    # Without this the run destroys the very weights it starts from. The
    # ArcFace head begins randomly initialised over ~9,700 classes, so its
    # initial loss is ~44 and the gradients flowing back into a well-trained
    # backbone are enormous. Measured on a 20-step trial: the validation margin
    # fell from +0.5880 to +0.4687 — worse than doing nothing at all.
    #
    # Freezing the backbone lets the head reach a sane operating point first,
    # so that when the backbone unfreezes it receives gradients that refine
    # rather than overwrite.
    for p in backbone.parameters():
        p.requires_grad_(False)
    print(f"  backbone FROZEN for the first {args.warmup} steps (head warm-up)")

    backbone.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if step == args.warmup + 1:
            for p in backbone.parameters():
                p.requires_grad_(True)
            # Drop the LR on unfreeze: the backbone is already good, so it
            # needs refinement, not the rate the head needed to converge.
            for g in opt.param_groups:
                g["lr"] = args.lr * 0.1
            print(f"  {'':>6} backbone UNFROZEN at step {step}, lr -> {args.lr * 0.1:g}")

        x, y = make_batch(tr_idx, True)
        opt.zero_grad(set_to_none=True)
        emb = backbone(x)
        loss = head(emb, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in backbone.parameters() if p.requires_grad] or list(head.parameters()), 5.0
        )
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            v = validate()
            note = ""
            if v > best:
                best, bad, best_step = v, 0, step
                torch.save({"backbone": backbone.state_dict(), "val_margin": v,
                            "step": step, "n_classes": n_classes}, ck)
                note = "saved"
            else:
                bad += 1
                note = f"no improve ({bad}/{args.patience})"
            print(f"  {step:>6} {loss.item():>9.4f} {v:>12.4f}  {note}")
            if bad >= args.patience:
                print(f"\n  EARLY STOP at step {step} (item 41)")
                break

    print(f"\n  best val margin {best:+.4f} at step {best_step} "
          f"(baseline {base:+.4f}, delta {best - base:+.4f})")
    print(f"  elapsed {time.time() - t0:.0f}s   checkpoint {ck}")
    print("\n  NO ACCURACY IS CLAIMED HERE. The val margin is an internal proxy.")
    print("  Run benchmark_finetuned.py / benchmark_tinyface.py against the")
    print("  checkpoint for numbers comparable to BENCHMARKS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
