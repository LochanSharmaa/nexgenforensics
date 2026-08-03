#!/usr/bin/env python
"""LoRA fine-tune of ViT-B KP-RPE on REAL low-resolution surveillance imagery.

    python backend/scripts/train_lora.py --epochs 4

THE HYPOTHESIS, PRE-REGISTERED (ARCHITECTURE_DECISION.md section 6)
-------------------------------------------------------------------
Real low-resolution training data raises identity information on QMUL above the
2.92-bit baseline.

    PASS  QMUL bits(median) >= 4.0  AND clean-pack regression <= 0.5 pts
          AND condition leakage not inflated above +0.0039
    FAIL  bits < 3.5, or clean regression > 0.5 pts, or leakage inflated

A leakage check is part of the gate because a TAR gain that inflates leakage
means the model learned the camera, not the face. Corpus-level leakage is
already +0.1088; making that worse would be a regression dressed as progress.

WHY LoRA AND NOT FULL FINE-TUNING
----------------------------------
Measured on this hardware: LoRA rank 8 on the 96 attention/MLP projections is
1.38M trainable of 115.1M (1.20%), peak 4.58 GiB at batch 24, 81 img/s. Full
fine-tuning does not fit in 6 GB at usable batch size. It is also the safer
science: the project's previous full fine-tune degraded every benchmark, with
catastrophic forgetting of calibration among the diagnosed causes. Constraining
the update to 1.2% of parameters bounds how far the representation can drift.

LOSS
----
AdaFace margin. The base checkpoint was trained with AdaFace, so this keeps the
objective consistent rather than pulling the head toward a different geometry.
Its adaptive margin scales with feature norm -- a quality proxy -- which is the
correct behaviour on a corpus where image quality varies by orders of magnitude:
low-quality samples get a smaller margin instead of dominating the gradient as
hard positives.

Outputs: runtime/lora/lora_<tag>.pt, runtime/forensics/lora_training_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.models.cvlface_backbone import CvlfaceViTKprpe  # noqa: E402

_ROOT = _BACKEND.parent
DATA = _ROOT / "runtime" / "lora"
OUT = _ROOT / "runtime" / "forensics"


class LoRALinear(nn.Module):
    """y = W0 x + (alpha/r) B A x, with W0 frozen."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zero => identity at init
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def inject_lora(model: nn.Module, r: int, alpha: int, targets=("qkv", "proj", "fc1", "fc2")) -> int:
    n = 0
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(mod, name, LoRALinear(child, r=r, alpha=alpha))
                n += 1
    return n


class AdaFace(nn.Module):
    """AdaFace margin head (Kim et al., CVPR 2022).

    Margin is modulated by the feature norm, batch-normalised against a running
    mean/std of norms. High-norm (high-quality) samples get a larger angular
    margin; low-norm (low-quality, possibly unidentifiable) samples get less,
    so noise does not act as an infinitely hard positive.
    """

    def __init__(self, dim: int, n_classes: int, m: float = 0.4, h: float = 0.333, s: float = 64.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.m, self.h, self.s = m, h, s
        self.register_buffer("t_mu", torch.zeros(1))
        self.register_buffer("t_std", torch.ones(1) * 100.0)

    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(feat, dim=1, keepdim=True).clamp(0.001, 100)
        f = F.normalize(feat)
        W = F.normalize(self.W)
        cos = (f @ W.T).clamp(-1 + 1e-7, 1 - 1e-7)

        with torch.no_grad():  # running statistics of feature norm
            self.t_mu = 0.99 * self.t_mu + 0.01 * norm.mean()
            self.t_std = 0.99 * self.t_std + 0.01 * norm.std().clamp(min=1e-3)
        margin_scaler = ((norm - self.t_mu) / (self.t_std + 1e-3) * self.h).clamp(-1, 1).detach()

        theta = torch.acos(cos)
        oh = F.one_hot(label, cos.size(1)).float()
        g_angle = -self.m * margin_scaler          # angular margin, quality-scaled
        g_add = self.m + self.m * margin_scaler    # additive margin, quality-scaled
        theta_m = (theta + oh * g_angle).clamp(1e-7, math.pi - 1e-7)
        return self.s * (torch.cos(theta_m) - oh * g_add)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=24)  # measured optimum at 6 GB
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--tag", default="r8")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (DATA / "manifest.json").exists():
        raise SystemExit("run prepare_lora_data.py first")
    man = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    crops = np.load(DATA / "crops.npy", mmap_mode="r")
    ldmks = np.load(DATA / "ldmks.npy")
    labels = np.load(DATA / "labels.npy")
    n, n_cls = len(labels), man["n_classes"]
    print(f"data: {n:,} images / {n_cls:,} identities  {man['corpora']}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"split: {len(tr_idx):,} train / {len(val_idx):,} held-out")

    torch.manual_seed(args.seed)
    wrap = CvlfaceViTKprpe(batch_size=args.batch)
    net = wrap.model
    for p in net.parameters():
        p.requires_grad = False
    k = inject_lora(net, args.rank, args.alpha)
    net = net.cuda().train()
    head = AdaFace(512, n_cls).cuda()

    lora_params = [p for p in net.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in lora_params)
    print(f"LoRA: {k} layers, rank {args.rank}, {n_tr/1e6:.2f}M trainable ({n_tr/115.1e6*100:.2f}%)")

    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr},
         {"params": head.parameters(), "lr": args.head_lr}], weight_decay=5e-4
    )
    steps = args.epochs * (len(tr_idx) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.head_lr],
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()

    def batch_of(idx):
        c = np.asarray(crops[idx])                       # (B,112,112,3) BGR uint8
        x = c[:, :, :, ::-1].astype(np.float32) / 255.0  # -> RGB
        x = (x - 0.5) / 0.5
        return (torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).cuda(),
                torch.from_numpy(ldmks[idx]).cuda(),
                torch.from_numpy(labels[idx]).cuda())

    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        order = rng.permutation(tr_idx)
        run_loss = run_acc = seen = 0
        for bi in range(len(order) // args.batch):
            idx = np.sort(order[bi * args.batch : (bi + 1) * args.batch])
            x, kp, y = batch_of(idx)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                feat = net(x, kp)
                logits = head(feat, y)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(idx)
            run_acc += (logits.argmax(1) == y).sum().item()
            seen += len(idx)
            if bi % 100 == 0:
                r = seen / max(time.time() - t0 - sum(h["val_s"] for h in hist), 1e-6)
                print(f"  ep{ep+1} {seen:,}/{len(order):,}  loss {run_loss/seen:.4f}  "
                      f"acc {run_acc/seen*100:.2f}%  {r:.0f} img/s", end="\r", flush=True)

        tv = time.time()
        net.eval()
        vl = va = vn = 0
        with torch.no_grad():
            for bi in range(max(len(val_idx) // args.batch, 1)):
                idx = np.sort(val_idx[bi * args.batch : (bi + 1) * args.batch])
                if len(idx) == 0:
                    break
                x, kp, y = batch_of(idx)
                with torch.cuda.amp.autocast():
                    logits = head(net(x, kp), y)
                    vl += F.cross_entropy(logits, y).item() * len(idx)
                va += (logits.argmax(1) == y).sum().item()
                vn += len(idx)
        val_s = time.time() - tv
        hist.append({"epoch": ep + 1, "train_loss": run_loss / seen, "train_acc": run_acc / seen,
                     "val_loss": vl / max(vn, 1), "val_acc": va / max(vn, 1), "val_s": val_s})
        print(f"\n  epoch {ep+1}: train loss {run_loss/seen:.4f} acc {run_acc/seen*100:.2f}%  |  "
              f"held-out loss {vl/max(vn,1):.4f} acc {va/max(vn,1)*100:.2f}%")

        ckpt = {"lora": {n_: p.detach().cpu() for n_, p in net.named_parameters() if p.requires_grad},
                "rank": args.rank, "alpha": args.alpha, "epoch": ep + 1}
        torch.save(ckpt, DATA / f"lora_{args.tag}.pt")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"lora_training_{args.tag}.json").write_text(
        json.dumps({"config": vars(args), "data": man["corpora"], "n_classes": n_cls,
                    "trainable_params": int(n_tr), "lora_layers": k,
                    "history": hist, "wall_seconds": time.time() - t0,
                    "gate": {"qmul_bits_pass": 4.0, "qmul_bits_fail": 3.5,
                             "clean_regression_max_pts": 0.5,
                             "leakage_baseline": 0.0039}}, indent=2, default=str),
        encoding="utf-8")
    print(f"\ntrained in {(time.time()-t0)/60:.1f} min -> {DATA / f'lora_{args.tag}.pt'}")
    print("Next: embed with the adapter and re-run measure_capacity_official.py to test the gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
