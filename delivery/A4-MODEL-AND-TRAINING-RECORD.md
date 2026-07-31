# A4 — Model and Training Record

**Generated:** 2026-07-31 19:25 UTC ·
**Repository state:** `52c6c0e3adca`

Every model artefact this system contains or produced, every training run
attempted, and the outcome of each.

Two of the training runs recorded here made the model worse or changed nothing.
They appear at the same level of detail as the work that succeeded, because a
training record containing only successful runs cannot answer *"was this
approach tried, and what happened?"* — which is the question the record exists
to answer.

---

## Statement of origin

**The deployed recognition weights were not trained by this project.** They are
stock InsightFace `buffalo_l` weights. The system's clean-image accuracy is the
accuracy of the public state of the art, and its provenance is publicly
verifiable. Everything in the "Checkpoints produced by this project" section
below is experimental; **none of it is the deployed model**, and the one
candidate recommended for use (R4) is a *routing rule* over two models rather
than a new set of weights.

---

# Part I — Model inventory and integrity

SHA-256 digests are the reference values for this delivery. A deployment must be
able to demonstrate that the model which produced a result is the model that was
validated; comparing against these digests is how that is shown.

Verify with:

```bash
sha256sum <file>          # Linux/macOS
certutil -hashfile <file> SHA256   # Windows
```

## Checkpoints produced by this project

`C:\Users\hello\Desktop\nexgenforensics\runtime\checkpoints`

| File | Size | SHA-256 |
|---|---|---|
| `arcface_degraded_v1.pt` | 174.5 MB | `12d2be807e1b49773828e718a58fa33e29a19ccc7648c37939c7f65f5a813ebe` |
| `arcface_ft_v1_20260730.pt` | 98.6 MB | `a29ed53d7b28f53e95027bc81751b32e1f4e6159fd7f9d555633a00dc80f9c26` |
| `arcface_qmul_v2.pt` | 174.5 MB | `b04a077fc075f444ef40b4c6304dda695a2ecc2505f15a38415ffbcc2106fef2` |
| `finetuned_resnet50_arcface.pt` | 98.6 MB | `7b5f3d76ec48081187ff995fd9d41e6da01a1681cb421d5fabe689494b46377b` |
| `test.pt` | 0.0 MB | `29528b2a51d707dfc2515c2a9f711d77494b9ce0d52fabc177c73ee2c90c141f` |

## Deployed InsightFace pack

`C:\Users\hello\.insightface\models\buffalo_l`

| File | Size | SHA-256 |
|---|---|---|
| `1k3d68.onnx` | 143.6 MB | `df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc` |
| `2d106det.onnx` | 5.0 MB | `f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf` |
| `det_10g.onnx` | 16.9 MB | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| `genderage.onnx` | 1.3 MB | `4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb` |
| `w600k_r50.onnx` | 174.4 MB | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |


---

# Part II — Training runs


## R1 — Initial fine-tune attempt (superseded)

### Method and rationale, as recorded in `backend/nexgen_engine/training/arcface_loss.py`

```text
Additive Angular Margin Loss (ArcFace)
    Paper: https://arxiv.org/abs/1801.07698

    Parameters:
    - in_features: embedding dimension (e.g. 512)
    - out_features: number of identities / classes
    - s: norm feature scale (default 64.0)
    - m: angular margin penalty in radians (default 0.50)
```

### Complete implementation — `backend/nexgen_engine/training/arcface_loss.py`

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace)
    Paper: https://arxiv.org/abs/1801.07698

    Parameters:
    - in_features: embedding dimension (e.g. 512)
    - out_features: number of identities / classes
    - s: norm feature scale (default 64.0)
    - m: angular margin penalty in radians (default 0.50)
    """
    def __init__(self, in_features=512, out_features=200, s=64.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # 1. Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 2. Compute sin(theta) and cos(theta + m)
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Keep margin valid for large angles
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 3. One-hot target encoding and apply margin only to ground-truth class
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # 4. Cross Entropy Loss
        loss = F.cross_entropy(output, label)
        return loss

```


## R2 — Synthetic-degradation fine-tune — NEGATIVE RESULT

### Method and rationale, as recorded in `backend/scripts/finetune_degraded.py`

```text
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
```

### Evaluation — `runtime/benchmarks/finetuned_v1.json`

```json
{
  "checkpoint": "C:\\Users\\hello\\Desktop\\nexgenforensics\\runtime\\checkpoints\\arcface_degraded_v1.pt",
  "checkpoint_meta": {
    "val_margin": 0.60551618039608,
    "step": 3000,
    "n_classes": 9380
  },
  "baseline_model": "w600k_r50 (deployed)",
  "note": "Both models scored in the same run on identical pair lists.",
  "results": {
    "lfw": {
      "deployed": {
        "accuracy_pct": 99.783,
        "tar_far_1e3_pct": 99.7,
        "auc": 0.99943
      },
      "finetuned": {
        "accuracy_pct": 99.75,
        "tar_far_1e3_pct": 99.667,
        "auc": 0.99963
      }
    },
    "agedb_30": {
      "deployed": {
        "accuracy_pct": 98.15,
        "tar_far_1e3_pct": 96.033,
        "auc": 0.9913
      },
      "finetuned": {
        "accuracy_pct": 97.383,
        "tar_far_1e3_pct": 86.967,
        "auc": 0.99184
      }
    },
    "cfp_fp": {
      "deployed": {
        "accuracy_pct": 97.443,
        "tar_far_1e3_pct": 94.686,
        "auc": 0.98023
      },
      "finetuned": {
        "accuracy_pct": 97.229,
        "tar_far_1e3_pct": 93.943,
        "auc": 0.98135
      }
    },
    "calfw": {
      "deployed": {
        "accuracy_pct": 95.95,
        "tar_far_1e3_pct": 92.1,
        "auc": 0.97755
      },
      "finetuned": {
        "accuracy_pct": 95.617,
        "tar_far_1e3_pct": 88.633,
        "auc": 0.97877
      }
    },
    "cplfw": {
      "deployed": {
        "accuracy_pct": 94.467,
        "tar_far_1e3_pct": 87.4,
        "auc": 0.96425
      },
      "finetuned": {
        "accuracy_pct": 93.883,
        "tar_far_1e3_pct": 85.133,
        "auc": 0.96017
      }
    },
    "tinyface": {
      "deployed": {
        "accuracy_pct": 82.45,
        "tar_far_1e3_pct": 33.133,
        "auc": 0.89217
      },
      "finetuned": {
        "accuracy_pct": 79.383,
        "tar_far_1e3_pct": 22.233,
        "auc": 0.8694
      }
    }
  }
}
```

### Complete implementation — `backend/scripts/finetune_degraded.py`

```python
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

```


## R3 — Real-degraded-data fine-tune (QMUL-SurvFace)

### Method and rationale, as recorded in `backend/scripts/finetune_qmul.py`

```text
Fine-tune for degraded imagery using REAL surveillance capture (QMUL-SurvFace),
not synthetic blur.

    python backend/scripts/finetune_qmul.py --steps 6000

This is the second attempt. The first (BENCHMARKS.md §6d) used synthetic
degradation -- bicubic down/up, Gaussian blur, JPEG -- and made the model WORSE
on every benchmark, worst of all on TinyFace (-3.07pp), the exact condition it
targeted. The diagnosis was a domain gap: the model learned to invert that
specific synthetic pipeline, which is not what a distant camera produces.

WHAT IS DIFFERENT HERE
----------------------
  degraded source   real QMUL-SurvFace capture (median 27x22px, 84% under 32px)
                    instead of synthetically degraded clean photos. NO synthetic
                    blur/JPEG is applied anywhere in this script.
  clean anchor      every batch also carries CASIA clean images, so clean-set
                    accuracy is trained against rather than sacrificed (item 6).
  validation        FIXED, PUBLISHED pair lists -- never a sampled proxy.

THE PROXY RULE (item 8) -- THE MOST IMPORTANT PART OF THIS FILE
---------------------------------------------------------------
The last run's training-time proxy reported +0.058 "improvement" while the real
benchmarks showed regression. The proxy was resampled every evaluation, so its
own noise (~0.06, measurable during the frozen-backbone phase where learning was
impossible) was as large as the effect it claimed to detect.

So nothing here early-stops on a resampled quantity. Both validation signals are
FIXED pair lists, scored through `evaluate_pairs` -- the same 10-fold harness
that produces every number in BENCHMARKS.md §2:

  degraded  QMUL's own published verification protocol: 5,320 positive and
            5,320 negative pairs over 4,888 identities that are VERIFIED
            disjoint from the 5,319 training identities (0 overlap).
  clean     a fixed pair list over 500 CASIA identities held out of training.

Neither is among the seven reporting benchmarks, so early stopping cannot leak
into the reported result. The reported result comes only from
eval_finetuned_checkpoint.py.
```

### Training history — `runtime/checkpoints/arcface_qmul_v2_history.json`

| step | loss | degraded | clean |
|---|---|---|---|
| 250 | 42.231 | 0.67143 | 0.941 |
| 500 | 40.6856 | 0.67152 | 0.94083 |
| 750 | 38.5669 | 0.66936 | 0.94033 |
| 1000 | 37.7322 | 0.69906 | 0.93917 |
| 1250 | 37.4616 | 0.71006 | 0.93883 |
| 1500 | 36.663 | 0.72575 | 0.9395 |
| 1750 | 34.8949 | 0.73355 | 0.9385 |
| 2000 | 35.3956 | 0.75047 | 0.93983 |
| 2250 | 33.6782 | 0.75761 | 0.939 |
| 2500 | 34.6233 | 0.76729 | 0.94033 |
| 2750 | 33.2824 | 0.77735 | 0.942 |
| 3000 | 35.4087 | 0.78252 | 0.94017 |
| 3250 | 33.133 | 0.78412 | 0.93967 |
| 3500 | 31.9 | 0.79549 | 0.9395 |
| 3750 | 32.7763 | 0.79774 | 0.94183 |
| 4000 | 33.0421 | 0.80047 | 0.942 |
| 4250 | 31.3031 | 0.80902 | 0.941 |
| 4500 | 31.838 | 0.80254 | 0.93983 |
| 4750 | 32.0299 | 0.80508 | 0.94 |
| 5000 | 30.0871 | 0.80733 | 0.9415 |
| 5250 | 29.0353 | 0.80855 | 0.9405 |
| 5500 | 28.3484 | 0.81739 | 0.94167 |
| 5750 | 29.254 | 0.81617 | 0.94017 |
| 6000 | 28.4292 | 0.81532 | 0.94133 |

| Field | Value |
|---|---|
| `baseline_degraded` | 0.6900375939849624 |
| `baseline_clean` | 0.943 |
| `best_step` | 5500 |
| `best_degraded` | 0.8173872180451127 |

#### Raw artefact

```json
{
  "baseline_degraded": 0.6900375939849624,
  "baseline_clean": 0.943,
  "best_step": 5500,
  "best_degraded": 0.8173872180451127,
  "history": [
    {
      "step": 250,
      "loss": 42.231,
      "degraded": 0.67143,
      "clean": 0.941
    },
    {
      "step": 500,
      "loss": 40.6856,
      "degraded": 0.67152,
      "clean": 0.94083
    },
    {
      "step": 750,
      "loss": 38.5669,
      "degraded": 0.66936,
      "clean": 0.94033
    },
    {
      "step": 1000,
      "loss": 37.7322,
      "degraded": 0.69906,
      "clean": 0.93917
    },
    {
      "step": 1250,
      "loss": 37.4616,
      "degraded": 0.71006,
      "clean": 0.93883
    },
    {
      "step": 1500,
      "loss": 36.663,
      "degraded": 0.72575,
      "clean": 0.9395
    },
    {
      "step": 1750,
      "loss": 34.8949,
      "degraded": 0.73355,
      "clean": 0.9385
    },
    {
      "step": 2000,
      "loss": 35.3956,
      "degraded": 0.75047,
      "clean": 0.93983
    },
    {
      "step": 2250,
      "loss": 33.6782,
      "degraded": 0.75761,
      "clean": 0.939
    },
    {
      "step": 2500,
      "loss": 34.6233,
      "degraded": 0.76729,
      "clean": 0.94033
    },
    {
      "step": 2750,
      "loss": 33.2824,
      "degraded": 0.77735,
      "clean": 0.942
    },
    {
      "step": 3000,
      "loss": 35.4087,
      "degraded": 0.78252,
      "clean": 0.94017
    },
    {
      "step": 3250,
      "loss": 33.133,
      "degraded": 0.78412,
      "clean": 0.93967
    },
    {
      "step": 3500,
      "loss": 31.9,
      "degraded": 0.79549,
      "clean": 0.9395
    },
    {
      "step": 3750,
      "loss": 32.7763,
      "degraded": 0.79774,
      "clean": 0.94183
    },
    {
      "step": 4000,
      "loss": 33.0421,
      "degraded": 0.80047,
      "clean": 0.942
    },
    {
      "step": 4250,
      "loss": 31.3031,
      "degraded": 0.80902,
      "clean": 0.941
    },
    {
      "step": 4500,
      "loss": 31.838,
      "degraded": 0.80254,
      "clean": 0.93983
    },
    {
      "step": 4750,
      "loss": 32.0299,
      "degraded": 0.80508,
      "clean": 0.94
    },
    {
      "step": 5000,
      "loss": 30.0871,
      "degraded": 0.80733,
      "clean": 0.9415
    },
    {
      "step": 5250,
      "loss": 29.0353,
      "degraded": 0.80855,
      "clean": 0.9405
    },
    {
      "step": 5500,
      "loss": 28.3484,
      "degraded": 0.81739,
      "clean": 0.94167
    },
    {
      "step": 5750,
      "loss": 29.254,
      "degraded": 0.81617,
      "clean": 0.94017
    },
    {
      "step": 6000,
      "loss": 28.4292,
      "degraded": 0.81532,
      "clean": 0.94133
    }
  ]
}
```

### Evaluation — `runtime/benchmarks/finetuned_qmul_v2.json`

```json
{
  "checkpoint": "runtime\\checkpoints\\arcface_qmul_v2.pt",
  "checkpoint_meta": {
    "step": 5500,
    "degraded_val": 0.8173872180451127,
    "clean_val": 0.9416666666666667,
    "baseline_degraded": 0.6900375939849624,
    "baseline_clean": 0.943,
    "n_classes": 15199
  },
  "baseline_model": "w600k_r50 (deployed)",
  "note": "Both models scored in the same run on identical pair lists.",
  "results": {
    "lfw": {
      "deployed": {
        "accuracy_pct": 99.783,
        "tar_far_1e3_pct": 99.7,
        "auc": 0.99943
      },
      "finetuned": {
        "accuracy_pct": 99.717,
        "tar_far_1e3_pct": 99.667,
        "auc": 0.99941
      }
    },
    "agedb_30": {
      "deployed": {
        "accuracy_pct": 98.15,
        "tar_far_1e3_pct": 96.033,
        "auc": 0.9913
      },
      "finetuned": {
        "accuracy_pct": 97.783,
        "tar_far_1e3_pct": 88.1,
        "auc": 0.99142
      }
    },
    "cfp_fp": {
      "deployed": {
        "accuracy_pct": 97.443,
        "tar_far_1e3_pct": 94.686,
        "auc": 0.98023
      },
      "finetuned": {
        "accuracy_pct": 97.171,
        "tar_far_1e3_pct": 92.829,
        "auc": 0.97406
      }
    },
    "cfp_ff": {
      "deployed": {
        "accuracy_pct": 99.871,
        "tar_far_1e3_pct": 99.857,
        "auc": 0.99978
      },
      "finetuned": {
        "accuracy_pct": 99.857,
        "tar_far_1e3_pct": 99.8,
        "auc": 0.99971
      }
    },
    "calfw": {
      "deployed": {
        "accuracy_pct": 95.95,
        "tar_far_1e3_pct": 92.1,
        "auc": 0.97755
      },
      "finetuned": {
        "accuracy_pct": 95.883,
        "tar_far_1e3_pct": 90.533,
        "auc": 0.97735
      }
    },
    "cplfw": {
      "deployed": {
        "accuracy_pct": 94.467,
        "tar_far_1e3_pct": 87.4,
        "auc": 0.96425
      },
      "finetuned": {
        "accuracy_pct": 93.333,
        "tar_far_1e3_pct": 81.733,
        "auc": 0.94468
      }
    },
    "tinyface": {
      "deployed": {
        "accuracy_pct": 82.45,
        "tar_far_1e3_pct": 33.133,
        "auc": 0.89217
      },
      "finetuned": {
        "accuracy_pct": 82.383,
        "tar_far_1e3_pct": 38.1,
        "auc": 0.901
      }
    }
  }
}
```

### Complete implementation — `backend/scripts/finetune_qmul.py`

```python
#!/usr/bin/env python
"""
Fine-tune for degraded imagery using REAL surveillance capture (QMUL-SurvFace),
not synthetic blur.

    python backend/scripts/finetune_qmul.py --steps 6000

This is the second attempt. The first (BENCHMARKS.md §6d) used synthetic
degradation -- bicubic down/up, Gaussian blur, JPEG -- and made the model WORSE
on every benchmark, worst of all on TinyFace (-3.07pp), the exact condition it
targeted. The diagnosis was a domain gap: the model learned to invert that
specific synthetic pipeline, which is not what a distant camera produces.

WHAT IS DIFFERENT HERE
----------------------
  degraded source   real QMUL-SurvFace capture (median 27x22px, 84% under 32px)
                    instead of synthetically degraded clean photos. NO synthetic
                    blur/JPEG is applied anywhere in this script.
  clean anchor      every batch also carries CASIA clean images, so clean-set
                    accuracy is trained against rather than sacrificed (item 6).
  validation        FIXED, PUBLISHED pair lists -- never a sampled proxy.

THE PROXY RULE (item 8) -- THE MOST IMPORTANT PART OF THIS FILE
---------------------------------------------------------------
The last run's training-time proxy reported +0.058 "improvement" while the real
benchmarks showed regression. The proxy was resampled every evaluation, so its
own noise (~0.06, measurable during the frozen-backbone phase where learning was
impossible) was as large as the effect it claimed to detect.

So nothing here early-stops on a resampled quantity. Both validation signals are
FIXED pair lists, scored through `evaluate_pairs` -- the same 10-fold harness
that produces every number in BENCHMARKS.md §2:

  degraded  QMUL's own published verification protocol: 5,320 positive and
            5,320 negative pairs over 4,888 identities that are VERIFIED
            disjoint from the 5,319 training identities (0 overlap).
  clean     a fixed pair list over 500 CASIA identities held out of training.

Neither is among the seven reporting benchmarks, so early stopping cannot leak
into the reported result. The reported result comes only from
eval_finetuned_checkpoint.py.
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

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "scripts"))

from nexgen_engine.benchmarks.verification import evaluate_pairs, l2n  # noqa: E402

_ROOT = _BACKEND.parent
QMUL_ROOT = Path("C:/Users/hello/Downloads/QMUL-SurvFace-v1/QMUL-SurvFace")
QMUL_TRAIN = QMUL_ROOT / "training_set"
QMUL_VER = QMUL_ROOT / "Face_Verification_Test_Set"


def to_tensor(batch: list[np.ndarray], device) -> torch.Tensor:
    a = np.stack(batch).astype(np.float32)[..., ::-1].copy()
    a = (a - 127.5) / 127.5
    return torch.from_numpy(a).permute(0, 3, 1, 2).to(device, non_blocking=True)


def load_backbone(device):
    import onnx
    import onnx2torch

    p = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
    return onnx2torch.convert(onnx.load_model(io.BytesIO(p.read_bytes()))).to(device)


def read112(path: Path) -> np.ndarray | None:
    im = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return None
    return cv2.resize(im, (112, 112))


def qmul_val_pairs() -> tuple[list[Path], np.ndarray, np.ndarray]:
    """QMUL's published verification protocol. Fixed pairs, fixed order."""
    from scipy.io import loadmat

    d = QMUL_VER / "verification_images"
    pos = loadmat(QMUL_VER / "positive_pairs_names.mat")["positive_pairs_names"]
    neg = loadmat(QMUL_VER / "negative_pairs_names.mat")["negative_pairs_names"]
    pairs, labels = [], []
    for arr, lab in ((pos, True), (neg, False)):
        for i in range(arr.shape[0]):
            pairs.append((d / str(arr[i, 0][0]), d / str(arr[i, 1][0])))
            labels.append(lab)
    # Interleave so each contiguous 10-fold slice holds both classes.
    n = min(len(pos), len(neg))
    inter, lab2 = [], []
    for i in range(n):
        inter.append(pairs[i]); lab2.append(True)
        inter.append(pairs[len(pos) + i]); lab2.append(False)
    files = sorted({p for pr in inter for p in pr})
    return files, np.array(lab2, dtype=bool), inter


class FixedPairSet:
    """A fixed pair list, embedded on demand and scored by evaluate_pairs."""

    def __init__(self, name: str, images: list[np.ndarray], pairs_idx, labels):
        self.name, self.images, self.pairs_idx, self.labels = name, images, pairs_idx, labels

    @torch.no_grad()
    def score(self, net, device, batch: int = 128) -> float:
        net.eval()
        n = len(self.images)
        e = np.zeros((n, 512), dtype=np.float32)
        for i in range(0, n, batch):
            chunk = self.images[i : i + batch]
            t = to_tensor(chunk, device)
            f = net(t).cpu().numpy()
            tf = net(to_tensor([c[:, ::-1] for c in chunk], device)).cpu().numpy()
            e[i : i + len(chunk)] = f + tf
        net.train()
        e = l2n(e.astype(np.float64))
        a = np.stack([e[i] for i, _ in self.pairs_idx])
        b = np.stack([e[j] for _, j in self.pairs_idx])
        return float(evaluate_pairs(a, b, self.labels, self.name, "ft").accuracy_mean)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--degraded-frac", type=float, default=0.5,
                    help="fraction of each batch drawn from QMUL (item 6)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--clean-guard", type=float, default=0.005,
                    help="reject a checkpoint whose clean val drops more than this")
    ap.add_argument("--qmul-per-identity", type=int, default=40)
    ap.add_argument("--casia-per-identity", type=int, default=6)
    ap.add_argument("--casia-val-identities", type=int, default=500)
    ap.add_argument("--val-pairs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclusion", default=str(_ROOT / "runtime/benchmarks/exclusion_list.json"))
    ap.add_argument("--out", default=str(_ROOT / "runtime/checkpoints/arcface_qmul_v2.pt"))
    args = ap.parse_args()

    from audit_train_eval_overlap import TRAIN_SETS, read_idx, read_record

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    print("=" * 78)
    print(f"  Fine-tune on REAL degraded capture (QMUL-SurvFace) - {device}")
    print("=" * 78)

    # ---------------- QMUL training identities (real degraded) -------------
    id_dirs = sorted([d for d in QMUL_TRAIN.iterdir() if d.is_dir()])
    q_imgs, q_lab = [], []
    print(f"  reading QMUL training_set ({len(id_dirs):,} identities) ...")
    for k, d in enumerate(id_dirs):
        fs = sorted(d.glob("*.jpg"))
        if len(fs) > args.qmul_per_identity:
            fs = [fs[i] for i in rng.choice(len(fs), args.qmul_per_identity, replace=False)]
        for f in fs:
            im = read112(f)
            if im is not None:
                q_imgs.append(im)
                q_lab.append(k)
        if k % 800 == 0:
            print(f"    {k:,}/{len(id_dirs):,}", end="\r", flush=True)
    n_q_ids = len(id_dirs)
    print(f"  QMUL   : {len(q_imgs):,} images / {n_q_ids:,} identities (REAL degraded)")

    # ---------------- CASIA clean anchor -----------------------------------
    keep = set(json.loads(Path(args.exclusion).read_text())["kept_labels"])
    d = TRAIN_SETS["faces_webface_112x112"]
    offsets = read_idx(d / "train.idx")
    order = rng.permutation(len(offsets))
    per: dict[int, int] = {}
    c_imgs, c_lab = [], []
    print("  reading CASIA clean records ...")
    with open(d / "train.rec", "rb") as fh:
        for i in order:
            r = read_record(fh, offsets[int(i)])
            if r is None:
                continue
            lab, blob = r
            if lab not in keep or per.get(lab, 0) >= args.casia_per_identity:
                continue
            im = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            if im.shape[:2] != (112, 112):
                im = cv2.resize(im, (112, 112))
            c_imgs.append(im)
            c_lab.append(lab)
            per[lab] = per.get(lab, 0) + 1
    c_lab = np.asarray(c_lab)
    casia_ids = sorted(set(c_lab.tolist()))
    print(f"  CASIA  : {len(c_imgs):,} images / {len(casia_ids):,} identities (clean anchor)")

    # ---- clean validation identities, held OUT of training (item 3) ----
    val_ids = set(rng.choice(casia_ids,
                             min(args.casia_val_identities, len(casia_ids) // 5),
                             replace=False).tolist())
    c_tr = np.array([i for i, l in enumerate(c_lab) if l not in val_ids])
    c_va = [i for i, l in enumerate(c_lab) if l in val_ids]
    print(f"  CASIA validation identities held out: {len(val_ids):,} "
          f"({len(c_va):,} images) - DISJOINT BY IDENTITY")

    # fixed clean pair list, built once from a fixed seed
    by_id: dict[int, list[int]] = {}
    for i in c_va:
        by_id.setdefault(int(c_lab[i]), []).append(i)
    multi = [v for v in by_id.values() if len(v) >= 2]
    prng = np.random.default_rng(12345)
    cpairs, clabels = [], []
    keys = list(by_id.values())
    for _ in range(args.val_pairs):
        g = multi[prng.integers(0, len(multi))]
        x, y = prng.choice(len(g), 2, replace=False)
        cpairs.append((g[int(x)], g[int(y)])); clabels.append(True)
        while True:
            p, q = prng.integers(0, len(keys)), prng.integers(0, len(keys))
            if p != q:
                break
        cpairs.append((keys[p][prng.integers(0, len(keys[p]))],
                       keys[q][prng.integers(0, len(keys[q]))]))
        clabels.append(False)
    pos_map = {orig: k for k, orig in enumerate(c_va)}
    clean_val = FixedPairSet("casia_clean_val", [c_imgs[i] for i in c_va],
                             [(pos_map[a], pos_map[b]) for a, b in cpairs],
                             np.array(clabels, dtype=bool))
    print(f"  clean val   : {len(cpairs):,} FIXED pairs")

    # ---- degraded validation: QMUL published protocol ----
    files, qlabels, qpairs = qmul_val_pairs()
    qv_imgs, qpos = [], {}
    for f in files:
        im = read112(f)
        if im is not None:
            qpos[f] = len(qv_imgs)
            qv_imgs.append(im)
    qidx = [(qpos[a], qpos[b]) for a, b in qpairs if a in qpos and b in qpos]
    qlabels = qlabels[: len(qidx)]
    degraded_val = FixedPairSet("qmul_verification", qv_imgs, qidx, qlabels)
    print(f"  degraded val: {len(qidx):,} FIXED pairs over {len(qv_imgs):,} images "
          f"(QMUL published protocol, 0 identity overlap with training)")

    # ---------------- model ------------------------------------------------
    n_classes = n_q_ids + len(casia_ids)
    remap_c = {l: n_q_ids + k for k, l in enumerate(casia_ids)}
    backbone = load_backbone(device)
    from nexgen_engine.training.arcface_loss import ArcFaceLoss

    head = ArcFaceLoss(in_features=512, out_features=n_classes, s=64.0, m=0.50).to(device)
    opt = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=5e-4)
    print(f"\n  backbone {sum(p.numel() for p in backbone.parameters()):,} params "
          f"(ArcFace init via onnx2torch, NOT ImageNet)")
    print(f"  head over {n_classes:,} classes "
          f"({n_q_ids:,} QMUL + {len(casia_ids):,} CASIA)")

    n_deg = int(round(args.batch_size * args.degraded_frac))
    n_cln = args.batch_size - n_deg
    print(f"  batch = {n_deg} real-degraded + {n_cln} clean")

    q_lab_arr = np.asarray(q_lab)

    def make_batch():
        qi = rng.choice(len(q_imgs), n_deg, replace=False)
        ci = rng.choice(c_tr, n_cln, replace=False)
        imgs = [q_imgs[i] for i in qi] + [c_imgs[i] for i in ci]
        ys = [int(q_lab_arr[i]) for i in qi] + [remap_c[int(c_lab[i])] for i in ci]
        return to_tensor(imgs, device), torch.tensor(ys, device=device)

    # ---------------- train ------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    base_deg = degraded_val.score(backbone, device)
    base_cln = clean_val.score(backbone, device)
    print(f"\n  BASELINE (deployed weights, before training)")
    print(f"    degraded val {base_deg * 100:.2f}%    clean val {base_cln * 100:.2f}%")

    for p in backbone.parameters():
        p.requires_grad_(False)
    print(f"\n  backbone FROZEN for {args.warmup} steps (head warm-up)")
    print(f"\n  {'step':>6} {'loss':>9} {'degraded':>10} {'clean':>9}  note")

    best, bad, best_step = -1.0, 0, 0
    history = []
    backbone.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if step == args.warmup + 1:
            for p in backbone.parameters():
                p.requires_grad_(True)
            for g in opt.param_groups:
                g["lr"] = args.lr * 0.1
            print(f"  {'':>6} backbone UNFROZEN, lr -> {args.lr * 0.1:g}")

        x, y = make_batch()
        opt.zero_grad(set_to_none=True)
        loss = head(backbone(x), y)
        loss.backward()
        trainable = [p for p in backbone.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable or list(head.parameters()), 5.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            dv = degraded_val.score(backbone, device)
            cv_ = clean_val.score(backbone, device)
            history.append({"step": step, "loss": round(loss.item(), 4),
                            "degraded": round(dv, 5), "clean": round(cv_, 5)})
            # Accept only if degraded improves AND clean has not fallen off a
            # cliff. A degraded gain bought with clean collapse is not a win.
            guarded = cv_ >= base_cln - args.clean_guard
            if dv > best and guarded:
                best, bad, best_step = dv, 0, step
                torch.save({"backbone": backbone.state_dict(), "step": step,
                            "degraded_val": dv, "clean_val": cv_,
                            "baseline_degraded": base_deg, "baseline_clean": base_cln,
                            "n_classes": n_classes}, out)
                note = "saved"
            else:
                bad += 1
                note = "clean guard" if not guarded else f"no improve ({bad}/{args.patience})"
            print(f"  {step:>6} {loss.item():>9.4f} {dv * 100:>9.2f}% "
                  f"{cv_ * 100:>8.2f}%  {note}")
            if bad >= args.patience:
                print(f"\n  EARLY STOP at step {step}")
                break

    print(f"\n  best degraded val {best * 100:.2f}% at step {best_step} "
          f"(baseline {base_deg * 100:.2f}%, delta {(best - base_deg) * 100:+.2f}pp)")
    print(f"  elapsed {time.time() - t0:.0f}s")
    (out.parent / "arcface_qmul_v2_history.json").write_text(json.dumps({
        "baseline_degraded": base_deg, "baseline_clean": base_cln,
        "best_step": best_step, "best_degraded": best, "history": history,
    }, indent=2))
    print(f"  checkpoint {out}")
    print("\n  These validation numbers are NOT the result. Run")
    print("  eval_finetuned_checkpoint.py for the seven reporting benchmarks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```


## R4 — Quality-routed model selection — ADOPTED CANDIDATE

### Method and rationale, as recorded in `backend/scripts/evaluate_routed_engine.py`

```text
Can the QMUL checkpoint be USED, rather than shelved?

    python backend/scripts/evaluate_routed_engine.py

BENCHMARKS.md records the fine-tune as "no improvement": accuracy moved nowhere
on any of the seven benchmarks. But the per-metric table says something more
specific than the accuracy column does. At the 0.1% false-match operating point
a forensic deployment actually uses:

    TinyFace   TAR@FAR0.1%   33.13 -> 38.10   (+4.97pp)
    AgeDB-30   TAR@FAR0.1%   96.03 -> 88.10   (-7.93pp)
    CPLFW      TAR@FAR0.1%   87.40 -> 81.73   (-5.67pp)

That is not a worse model. It is a DIFFERENT model: better where images are
degraded, worse where they are clean but hard (age, pose). A single global
choice between them throws away whichever advantage it does not pick.

This script tests whether choosing PER PROBE recovers both, using the quality
score the pipeline already computes on every request -- so routing costs no
extra inference.

THREE QUESTIONS, IN ORDER. The third only matters if the first two hold.

 1. Are the two embedding spaces compatible? If a template enrolled under one
    model can be compared against a probe under the other, routing is free
    everywhere. If not, 1:N search needs BOTH templates stored per subject and
    that is a real cost, not a detail.

 2. Does the quality score actually separate degraded from clean imagery? If it
    does not, there is nothing to route on and the idea dies here.

 3. Does routing beat the deployed model on TinyFace WITHOUT regressing the
    clean sets? This is the only claim worth making, and it is the one that
    would be quoted, so it is measured end to end rather than inferred from the
    two columns above.
```

### Training history — `runtime/benchmarks/routing_threshold.json`

| Field | Value |
|---|---|
| `disjoint_from_reporting_benchmarks` | True |
| `qmul_median` | 0.4677 |
| `casia_median` | 0.7547 |
| `threshold` | 0.539 |

#### Raw artefact

```json
{
  "derived_from": [
    "QMUL-SurvFace training_set",
    "CASIA-WebFace train.rec"
  ],
  "disjoint_from_reporting_benchmarks": true,
  "qmul_median": 0.4677,
  "casia_median": 0.7547,
  "threshold": 0.539
}
```

### Evaluation — `runtime/benchmarks/routed_engine_validated.json`

```json
{
  "cross_model_same_image_median": 0.8558,
  "embedding_spaces_compatible": true,
  "quality_by_dataset": {
    "lfw": 0.7424,
    "agedb_30": 0.7957,
    "cfp_fp": 0.7753,
    "cfp_ff": 0.7859,
    "calfw": 0.7863,
    "cplfw": 0.7551,
    "tinyface": 0.5023
  },
  "quality_separation": 0.2783,
  "verdict": "ADOPT",
  "routing_threshold": 0.539,
  "results": {
    "lfw": {
      "deployed_acc": 99.783,
      "specialist_acc": 99.717,
      "routed_acc": 99.783,
      "deployed_tar_1e3": 99.7,
      "specialist_tar_1e3": 99.667,
      "routed_tar_1e3": 99.7,
      "fraction_routed_to_specialist": 0.0275
    },
    "agedb_30": {
      "deployed_acc": 98.15,
      "specialist_acc": 97.783,
      "routed_acc": 98.117,
      "deployed_tar_1e3": 96.033,
      "specialist_tar_1e3": 88.1,
      "routed_tar_1e3": 95.967,
      "fraction_routed_to_specialist": 0.0138
    },
    "cfp_fp": {
      "deployed_acc": 97.443,
      "specialist_acc": 97.171,
      "routed_acc": 97.429,
      "deployed_tar_1e3": 94.686,
      "specialist_tar_1e3": 92.829,
      "routed_tar_1e3": 94.657,
      "fraction_routed_to_specialist": 0.0331
    },
    "cfp_ff": {
      "deployed_acc": 99.871,
      "specialist_acc": 99.857,
      "routed_acc": 99.871,
      "deployed_tar_1e3": 99.857,
      "specialist_tar_1e3": 99.8,
      "routed_tar_1e3": 99.857,
      "fraction_routed_to_specialist": 0.0007
    },
    "calfw": {
      "deployed_acc": 95.95,
      "specialist_acc": 95.883,
      "routed_acc": 96.0,
      "deployed_tar_1e3": 92.1,
      "specialist_tar_1e3": 90.533,
      "routed_tar_1e3": 92.1,
      "fraction_routed_to_specialist": 0.0038
    },
    "cplfw": {
      "deployed_acc": 94.467,
      "specialist_acc": 93.333,
      "routed_acc": 94.333,
      "deployed_tar_1e3": 87.4,
      "specialist_tar_1e3": 81.733,
      "routed_tar_1e3": 87.267,
      "fraction_routed_to_specialist": 0.0745
    },
    "tinyface": {
      "deployed_acc": 82.45,
      "specialist_acc": 82.383,
      "routed_acc": 82.533,
      "deployed_tar_1e3": 33.133,
      "specialist_tar_1e3": 38.1,
      "routed_tar_1e3": 37.367,
      "fraction_routed_to_specialist": 0.9223
    }
  }
}
```

### Complete implementation — `backend/scripts/evaluate_routed_engine.py`

```python
#!/usr/bin/env python
"""
Can the QMUL checkpoint be USED, rather than shelved?

    python backend/scripts/evaluate_routed_engine.py

BENCHMARKS.md records the fine-tune as "no improvement": accuracy moved nowhere
on any of the seven benchmarks. But the per-metric table says something more
specific than the accuracy column does. At the 0.1% false-match operating point
a forensic deployment actually uses:

    TinyFace   TAR@FAR0.1%   33.13 -> 38.10   (+4.97pp)
    AgeDB-30   TAR@FAR0.1%   96.03 -> 88.10   (-7.93pp)
    CPLFW      TAR@FAR0.1%   87.40 -> 81.73   (-5.67pp)

That is not a worse model. It is a DIFFERENT model: better where images are
degraded, worse where they are clean but hard (age, pose). A single global
choice between them throws away whichever advantage it does not pick.

This script tests whether choosing PER PROBE recovers both, using the quality
score the pipeline already computes on every request -- so routing costs no
extra inference.

THREE QUESTIONS, IN ORDER. The third only matters if the first two hold.

 1. Are the two embedding spaces compatible? If a template enrolled under one
    model can be compared against a probe under the other, routing is free
    everywhere. If not, 1:N search needs BOTH templates stored per subject and
    that is a real cost, not a detail.

 2. Does the quality score actually separate degraded from clean imagery? If it
    does not, there is nothing to route on and the idea dies here.

 3. Does routing beat the deployed model on TinyFace WITHOUT regressing the
    clean sets? This is the only claim worth making, and it is the one that
    would be quoted, so it is measured end to end rather than inferred from the
    two columns above.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "scripts"))

from nexgen_engine.benchmarks.verification import (  # noqa: E402
    decode_pack,
    evaluate_pairs,
    l2n,
    load_pack,
)

_ROOT = _BACKEND.parent
CLEAN = ["lfw", "agedb_30", "cfp_fp", "cfp_ff", "calfw", "cplfw"]


def quality_scores(images: list[np.ndarray]) -> np.ndarray:
    """The pipeline's own quality score, one per image."""
    from PIL import Image

    from nexgen_engine.config import QualityConfig
    from nexgen_engine.data.quality_filter import ImageQualityFilter

    qf = ImageQualityFilter(QualityConfig())
    out = np.zeros(len(images), dtype=np.float64)
    for i, img in enumerate(images):
        try:
            # face=None so the whole frame is measured -- correct here, because
            # every benchmark image IS an aligned face crop already.
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            out[i] = qf.evaluate(pil).score
        except Exception:
            out[i] = float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(_ROOT / "runtime/checkpoints/arcface_qmul_v2.pt"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tinyface-pairs", type=int, default=3000)
    ap.add_argument("--threshold", type=float, default=None,
                    help="route below this quality to the specialist; derived if omitted")
    ap.add_argument("--sweep", action="store_true", help="print the threshold tradeoff curve")
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/routed_engine.json"))
    args = ap.parse_args()

    from benchmark_verification import find_pack, load_recognizer
    from eval_finetuned_checkpoint import TorchRecognizer, embed_all, tinyface_pairs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("  Quality-routed engine: can the QMUL checkpoint be used?")
    print("=" * 78)

    specialist = TorchRecognizer(Path(args.checkpoint), device)
    generalist = load_recognizer("w600k_r50")
    report: dict = {}

    # ---- gather every dataset once -------------------------------------
    datasets: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    for ds in CLEAN:
        try:
            bins, issame = load_pack(find_pack(ds))
        except Exception as exc:
            print(f"  {ds}: unavailable ({exc}); skipped")
            continue
        datasets[ds] = (list(decode_pack(bins)), np.asarray(issame, dtype=bool))

    tf = tinyface_pairs(args.tinyface_pairs, 0)
    tf_pairs = None
    if tf is not None:
        pairs, labels = tf
        needed = sorted({p for pr in pairs for p in pr})
        imgs = []
        for p in needed:
            im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            imgs.append(cv2.resize(im, (112, 112)))
        pos = {p: i for i, p in enumerate(needed)}
        tf_pairs = (imgs, pairs, labels, pos)

    # ---- Q1: are the embedding spaces compatible? ----------------------
    print("\n1. EMBEDDING SPACE COMPATIBILITY")
    probe_imgs = datasets["lfw"][0][:256] if "lfw" in datasets else []
    if probe_imgs:
        a = l2n(np.asarray(generalist.get_feat(probe_imgs)).astype(np.float64))
        b = l2n(np.asarray(specialist.get_feat(probe_imgs)).astype(np.float64))
        same_image = np.sum(a * b, axis=1)
        print(f"   same image, both models : median cosine {np.median(same_image):+.4f}")
        compatible = bool(np.median(same_image) > 0.5)
        report["cross_model_same_image_median"] = round(float(np.median(same_image)), 4)
        report["embedding_spaces_compatible"] = compatible
        if compatible:
            print("   -> COMPATIBLE. A template from one model can be searched with the other.")
        else:
            print("   -> INCOMPATIBLE, as expected: fine-tuning rotated the space.")
            print("      1:1 verification can route freely (both images use one model).")
            print("      1:N search CANNOT: a routed probe would be compared against")
            print("      templates from the other space and match nothing. Serving both")
            print("      means storing TWO templates per subject.")

    # ---- Q2: does quality separate the conditions? ---------------------
    print("\n2. DOES THE QUALITY SCORE SEPARATE DEGRADED FROM CLEAN?")
    q_by_ds: dict[str, np.ndarray] = {}
    for ds, (imgs, _) in datasets.items():
        q_by_ds[ds] = quality_scores(imgs[:1500])
    if tf_pairs:
        q_by_ds["tinyface"] = quality_scores(tf_pairs[0][:1500])

    for ds, q in q_by_ds.items():
        finite = q[np.isfinite(q)]
        if finite.size:
            print(f"   {ds:10s} median {np.median(finite):.4f}   "
                  f"p10 {np.percentile(finite, 10):.4f}  p90 {np.percentile(finite, 90):.4f}")
    report["quality_by_dataset"] = {
        k: round(float(np.median(v[np.isfinite(v)])), 4)
        for k, v in q_by_ds.items() if np.isfinite(v).any()
    }

    clean_med = np.median([v for k, v in report["quality_by_dataset"].items() if k != "tinyface"])
    tiny_med = report["quality_by_dataset"].get("tinyface")
    if tiny_med is not None:
        gap = clean_med - tiny_med
        print(f"\n   clean median {clean_med:.4f}  vs  tinyface median {tiny_med:.4f}"
              f"   separation {gap:+.4f}")
        report["quality_separation"] = round(float(gap), 4)
        if gap < 0.05:
            print("   -> The score does NOT separate the conditions. There is nothing to")
            print("      route on; a quality-gated engine cannot be built from this signal.")
        else:
            print("   -> Separated. A threshold between them is a usable routing rule.")

    # ---- Q3: measure the routed system end to end ----------------------
    print("\n3. ROUTED SYSTEM vs DEPLOYED, on identical pairs")
    print(f"   {'dataset':10s} {'deployed':>9s} {'specialist':>11s} {'ROUTED':>9s}"
          f" | {'TAR dep':>8s} {'TAR rt':>8s}")
    print("   " + "-" * 66)

    rows = []
    # THRESHOLD CHOICE.
    #
    # NOT the midpoint between the two medians -- that lands inside the clean
    # distribution's lower tail and misroutes a third of LFW and CPLFW for no
    # gain. The distributions overlap far less than their medians suggest:
    # TinyFace's p90 sits below the lowest clean p10, so a threshold in that
    # gap captures essentially all degraded imagery and almost no clean.
    #
    # It is set from the QUALITY DISTRIBUTIONS, which are a property of image
    # capture, and NOT from benchmark accuracy -- tuning it on the seven
    # reporting sets would be fitting the operating point to the test set and
    # every number after it would be unquotable.
    threshold = args.threshold
    if threshold is None:
        tiny_q = q_by_ds.get("tinyface")
        clean_p10 = [np.percentile(v[np.isfinite(v)], 10)
                     for k, v in q_by_ds.items() if k != "tinyface" and np.isfinite(v).any()]
        if tiny_q is not None and clean_p10:
            hi = float(min(clean_p10))
            lo = float(np.percentile(tiny_q[np.isfinite(tiny_q)], 90))
            threshold = round((lo + hi) / 2, 3) if lo < hi else round(lo, 3)
        else:
            threshold = 0.58
    print(f"   routing rule: quality < {threshold} -> specialist, else deployed\n")

    cache: dict[str, tuple] = {}

    def score_set(name, imgs, pair_idx, labels, thr=None):
        thr = threshold if thr is None else thr
        if name not in cache:
            cache[name] = (
                l2n(embed_all(generalist, imgs, args.batch_size).astype(np.float64)),
                l2n(embed_all(specialist, imgs, args.batch_size).astype(np.float64)),
                quality_scores(imgs),
            )
        eg, es, q = cache[name]
        a_i = np.array([i for i, _ in pair_idx])
        b_i = np.array([j for _, j in pair_idx])
        # A pair is routed to the specialist when EITHER image is degraded: the
        # weaker image is what limits the comparison.
        use_spec = (np.nan_to_num(q[a_i], nan=1.0) < thr) | (
            np.nan_to_num(q[b_i], nan=1.0) < thr)
        ga, gb = eg[a_i], eg[b_i]
        sa, sb = es[a_i], es[b_i]
        ra = np.where(use_spec[:, None], sa, ga)
        rb = np.where(use_spec[:, None], sb, gb)
        return (
            evaluate_pairs(ga, gb, labels, name, "deployed"),
            evaluate_pairs(sa, sb, labels, name, "specialist"),
            evaluate_pairs(ra, rb, labels, name, "routed"),
            float(use_spec.mean()),
        )

    for ds, (imgs, issame) in datasets.items():
        idx = [(2 * i, 2 * i + 1) for i in range(len(issame))]
        dep, spec, routed, frac = score_set(ds, imgs, idx, issame)
        rows.append((ds, dep, spec, routed, frac))

    if tf_pairs:
        imgs, pairs, labels, pos = tf_pairs
        idx = [(pos[p[0]], pos[p[1]]) for p in pairs]
        dep, spec, routed, frac = score_set("tinyface", imgs, idx, labels)
        rows.append(("tinyface", dep, spec, routed, frac))

    payload = {}
    for ds, dep, spec, routed, frac in rows:
        print(f"   {ds:10s} {dep.accuracy_mean * 100:8.2f}% {spec.accuracy_mean * 100:10.2f}%"
              f" {routed.accuracy_mean * 100:8.2f}% | {dep.tar_at_far_1e3 * 100:7.2f}%"
              f" {routed.tar_at_far_1e3 * 100:7.2f}%   ({frac * 100:.0f}% routed)")
        payload[ds] = {
            "deployed_acc": round(dep.accuracy_mean * 100, 3),
            "specialist_acc": round(spec.accuracy_mean * 100, 3),
            "routed_acc": round(routed.accuracy_mean * 100, 3),
            "deployed_tar_1e3": round(dep.tar_at_far_1e3 * 100, 3),
            "specialist_tar_1e3": round(spec.tar_at_far_1e3 * 100, 3),
            "routed_tar_1e3": round(routed.tar_at_far_1e3 * 100, 3),
            "fraction_routed_to_specialist": round(frac, 4),
        }

    if args.sweep:
        print("\n   threshold sweep (TAR@FAR0.1%)")
        print(f"   {'thr':>6s} {'tinyface':>9s} {'worst clean delta':>19s}")
        for thr in [0.50, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.68]:
            deltas, tiny_tar = [], None
            for ds, (imgs, issame) in datasets.items():
                idx = [(2 * i, 2 * i + 1) for i in range(len(issame))]
                d, _, r, _ = score_set(ds, imgs, idx, issame, thr=thr)
                deltas.append(r.tar_at_far_1e3 - d.tar_at_far_1e3)
            if tf_pairs:
                imgs, pairs, labels, pos = tf_pairs
                idx = [(pos[p[0]], pos[p[1]]) for p in pairs]
                d, _, r, _ = score_set("tinyface", imgs, idx, labels, thr=thr)
                tiny_tar = r.tar_at_far_1e3 * 100
            mark = "  <- chosen" if abs(thr - threshold) < 1e-9 else ""
            print(f"   {thr:6.2f} {tiny_tar:8.2f}% {min(deltas) * 100:+18.2f}pp{mark}")

    # ---- verdict --------------------------------------------------------
    print("\n" + "=" * 78)
    tiny = payload.get("tinyface")
    clean_rows = [(k, v) for k, v in payload.items() if k != "tinyface"]
    worst_clean = min((v["routed_tar_1e3"] - v["deployed_tar_1e3"] for _, v in clean_rows),
                      default=0.0)
    if tiny:
        tar_gain = tiny["routed_tar_1e3"] - tiny["deployed_tar_1e3"]
        print(f"  TinyFace TAR@FAR0.1%  {tiny['deployed_tar_1e3']:.2f} -> "
              f"{tiny['routed_tar_1e3']:.2f}  ({tar_gain:+.2f}pp)")
        print(f"  worst clean-set TAR change under routing: {worst_clean:+.2f}pp")
        report["verdict"] = (
            "ADOPT" if tar_gain > 1.0 and worst_clean > -0.5 else "DO NOT ADOPT"
        )
        print(f"\n  VERDICT: {report['verdict']}")
    print("=" * 78)

    report["routing_threshold"] = threshold
    report["results"] = payload
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```


---

# Part III — Supporting instrumentation

Scripts that established dataset integrity and evaluated the resulting checkpoints. Included in full because the validity of every training outcome above depends on them being correct.

## `backend/scripts/build_exclusion_list.py`

```python
#!/usr/bin/env python
"""
Phase 6 step 3 — build the training-identity EXCLUSION LIST.

    python backend/scripts/build_exclusion_list.py --per-identity 10

audit_train_eval_overlap.py answers "is there overlap?" from the EVAL side: for
each evaluation image, how similar is the nearest training image. That proves
contamination exists but cannot be acted on, because it never names the
training identities responsible.

This answers the other direction: for each TRAINING identity, how similar is its
nearest evaluation image. Identities above the threshold are written to an
exclusion list so a fine-tune can drop them.

THRESHOLD CHOICE — DELIBERATELY CONSERVATIVE
--------------------------------------------
Default 0.40, which is BELOW the ~0.49 mean of genuine same-person pairs in this
system. That is intentional and asymmetric:

  * Excluding a clean identity costs a little training data. Cheap.
  * Keeping a contaminated one means the model memorises a face it will later
    be tested on, and every downstream accuracy number becomes unfalsifiable.

So the threshold errs toward over-exclusion. A tighter value (0.70, the
"probable same identity" line used in the audit) would leave same-person pairs
scoring 0.40-0.70 in the training set, and those are exactly the cross-age and
cross-pose pairs the benchmarks are built from.

Uses faiss IndexFlatIP for the nearest-neighbour search -- exact, no recall
loss, and the one place BENCHMARKS.md §7d found it genuinely worth using.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "scripts"))

from audit_train_eval_overlap import TRAIN_SETS, embed, sample_training  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"


def eval_gallery(model: str) -> tuple[np.ndarray, list[str]]:
    """Every cached evaluation embedding, pooled into one matrix."""
    parts, names = [], []
    for p in sorted(CACHE.glob(f"*__{model}.npz")):
        d = np.load(p)
        e = (d["orig"] + d["flip"]).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        parts.append(e)
        names.append(p.name.split("__")[0])
    if not parts:
        raise SystemExit("no cached eval embeddings; run benchmark_verification.py first")
    return np.concatenate(parts, axis=0), names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-set", default="faces_webface_112x112", choices=list(TRAIN_SETS))
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--per-identity", type=int, default=10)
    ap.add_argument("--max-images", type=int, default=120000)
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/exclusion_list.json"))
    args = ap.parse_args()

    import faiss
    from benchmark_verification import load_recognizer

    print("=" * 78)
    print("  Phase 6 step 3 - training-identity exclusion list")
    print(f"  threshold {args.threshold} (below the ~0.49 genuine mean: errs toward")
    print("  over-exclusion, because keeping a contaminated identity is worse)")
    print("=" * 78)

    gallery, ev_names = eval_gallery(args.model)
    print(f"\n  eval gallery : {gallery.shape[0]:,} embeddings from {ev_names}")

    d = TRAIN_SETS[args.train_set]
    print(f"  sampling {args.train_set} at <= {args.per_identity}/identity ...")
    imgs, labels = sample_training(args.train_set, d, args.per_identity, args.max_images, args.seed)
    if not imgs:
        print("  could not read training records")
        return 1
    ids = sorted(set(labels.tolist()))
    print(f"  sampled {len(imgs):,} images across {len(ids):,} identities")

    model = load_recognizer(args.model)
    temb = embed(model, imgs).astype(np.float32)
    print(f"  embedded {temb.shape[0]:,} training images")

    # Exact nearest-neighbour: recall 1.000 by construction. No approximation
    # here -- a missed match would silently leave a contaminated identity in.
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)
    sims, _ = index.search(np.ascontiguousarray(temb), 1)
    best = sims[:, 0]

    per_identity: dict[int, float] = defaultdict(lambda: -1.0)
    for lab, s in zip(labels.tolist(), best.tolist()):
        if s > per_identity[lab]:
            per_identity[lab] = s

    excluded = sorted(i for i, s in per_identity.items() if s >= args.threshold)
    kept = sorted(i for i, s in per_identity.items() if s < args.threshold)

    print(f"\n  {'identity max-similarity distribution':38s}")
    for lo, hi in [(0.9, 2.0), (0.7, 0.9), (0.5, 0.7), (0.4, 0.5), (0.3, 0.4), (-2.0, 0.3)]:
        n = sum(1 for s in per_identity.values() if lo <= s < hi)
        band = f">={lo:.1f}" if hi > 1 else f"{lo:.1f}-{hi:.1f}"
        print(f"    {band:>10s}  {n:>6,} identities  {n / len(per_identity) * 100:5.1f}%")

    print(f"\n  EXCLUDE : {len(excluded):,} identities ({len(excluded) / len(ids) * 100:.1f}%)")
    print(f"  KEEP    : {len(kept):,} identities ({len(kept) / len(ids) * 100:.1f}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "train_set": args.train_set,
        "model": args.model,
        "threshold": args.threshold,
        "per_identity_sampled": args.per_identity,
        "images_sampled": len(imgs),
        "identities_total": len(ids),
        "identities_excluded": len(excluded),
        "identities_kept": len(kept),
        "eval_sets": ev_names,
        "limitation": (
            "Sampling. An identity whose sampled images happen not to resemble "
            "the eval shots is NOT excluded. This list is a floor: the true "
            "contaminated set is at least this large."
        ),
        "excluded_labels": excluded,
        "kept_labels": kept,
    }, indent=2))
    print(f"\n  Wrote {out}")
    print("\n  NOTE: this list is a FLOOR. Sampling cannot prove an identity clean,")
    print("  only that its sampled images did not match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

## `backend/scripts/audit_qmul_survface.py`

```python
#!/usr/bin/env python
"""
QMUL-SurvFace identity-overlap audit + exclusion list.

    python backend/scripts/audit_qmul_survface.py

Same methodology as build_exclusion_list.py (CASIA), with two additions that
this dataset specifically requires.

WHY THIS ONE NEEDS EXTRA CARE
-----------------------------
1. TinyFace and QMUL-SurvFace come from THE SAME LAB (Cheng, Zhu & Gong at
   QMUL). TinyFace is the degraded-condition benchmark this whole exercise is
   trying to improve. If the two share source imagery, training on SurvFace and
   reporting a TinyFace gain would be measuring memorisation. So the nearest
   neighbour is attributed back to WHICH eval set it came from, not just scored.

2. SurvFace images are native low-resolution surveillance crops; the five clean
   eval sets are high-quality portraits. Degraded probes produce systematically
   WEAKER embeddings, which compresses cosine similarity downward for everything
   including true matches. A fixed 0.40 threshold carried over from the CASIA
   audit is therefore a LOOSER filter here in real terms, not a stricter one.
   Counts are reported at several thresholds so that sensitivity is visible
   rather than hidden behind one number.

The gallery covers all seven evaluation sets: LFW, AgeDB-30, CFP-FP, CFP-FF,
CALFW, CPLFW and TinyFace.
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
sys.path.insert(0, str(_BACKEND / "scripts"))

from nexgen_engine.benchmarks.verification import decode_pack, load_pack  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
QMUL = Path("C:/Users/hello/Downloads/QMUL-SurvFace-v1/QMUL-SurvFace/training_set")
TINYFACE = _ROOT / "src_extracted/tinyface/tinyface/Testing_Set"
CACHED_SETS = ["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"]
ID_RE = re.compile(r"^(\d+)_")


def embed_images(model, images: list[np.ndarray], batch: int = 64) -> np.ndarray:
    """Original + flip, summed then L2-normalised -- as every other benchmark."""
    n = len(images)
    out = np.zeros((n, 512), dtype=np.float32)
    for i in range(0, n, batch):
        chunk = images[i : i + batch]
        out[i : i + len(chunk)] = (
            np.asarray(model.get_feat(list(chunk)))
            + np.asarray(model.get_feat([c[:, ::-1] for c in chunk]))
        )
        if (i // batch) % 40 == 0:
            print(f"      {min(i + batch, n)}/{n}", end="\r", flush=True)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out


def build_gallery(model_key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (embeddings, source_id_per_row, source_names)."""
    from benchmark_verification import find_pack, load_recognizer

    parts, src, names = [], [], []
    model = None

    for ds in CACHED_SETS:
        p = CACHE / f"{ds}__{model_key}.npz"
        if not p.exists():
            print(f"    {ds}: no cache; SKIPPED")
            continue
        d = np.load(p)
        e = (d["orig"] + d["flip"]).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
        parts.append(e)
        src.append(np.full(len(e), len(names), dtype=np.int32))
        names.append(ds)
        print(f"    {ds}: {len(e):,} (cached)")

    # CFP-FF -- a published pack, never previously embedded here.
    ff_cache = CACHE / f"cfp_ff__{model_key}.npz"
    try:
        if ff_cache.exists():
            d = np.load(ff_cache)
            e = (d["orig"] + d["flip"]).astype(np.float32)
        else:
            bins, issame = load_pack(find_pack("cfp_ff"))
            imgs = decode_pack(bins)
            model = model or load_recognizer(model_key)
            print(f"    cfp_ff: embedding {len(imgs):,} ...")
            raw = np.zeros((len(imgs), 512), dtype=np.float32)
            for i in range(0, len(imgs), 64):
                c = imgs[i : i + 64]
                raw[i : i + len(c)] = (np.asarray(model.get_feat([x for x in c]))
                                       + np.asarray(model.get_feat([x[:, ::-1] for x in c])))
            np.savez_compressed(ff_cache, orig=raw, flip=np.zeros((1, 1)), issame=issame)
            e = raw
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        parts.append(e)
        src.append(np.full(len(e), len(names), dtype=np.int32))
        names.append("cfp_ff")
        print(f"    cfp_ff: {len(e):,}")
    except Exception as exc:
        print(f"    cfp_ff: unavailable ({exc}); SKIPPED")

    # TinyFace -- the labelled benchmark surface (Gallery_Match + Probe).
    tf_cache = CACHE / f"tinyface_labelled__{model_key}.npz"
    if tf_cache.exists():
        e = np.load(tf_cache)["emb"]
    else:
        files = []
        for sub in ("Gallery_Match", "Probe"):
            d = TINYFACE / sub
            if d.is_dir():
                files += sorted(d.glob("*.jpg"))
        if files:
            imgs = []
            for f in files:
                im = cv2.imdecode(np.frombuffer(f.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if im is not None:
                    imgs.append(cv2.resize(im, (112, 112)))
            model = model or load_recognizer(model_key)
            print(f"    tinyface: embedding {len(imgs):,} ...")
            e = embed_images(model, imgs)
            np.savez_compressed(tf_cache, emb=e)
        else:
            e = None
    if e is not None and len(e):
        parts.append(e.astype(np.float32))
        src.append(np.full(len(e), len(names), dtype=np.int32))
        names.append("tinyface")
        print(f"    tinyface: {len(e):,}")

    return np.concatenate(parts), np.concatenate(src), names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--per-identity", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/qmul_exclusion_list.json"))
    args = ap.parse_args()

    import faiss
    from benchmark_verification import load_recognizer

    print("=" * 78)
    print("  QMUL-SurvFace identity-overlap audit")
    print("=" * 78)

    if not QMUL.is_dir():
        print(f"  training set not found at {QMUL}")
        return 1

    print("\n  Building evaluation gallery (all 7 sets):")
    gallery, gsrc, names = build_gallery(args.model)
    print(f"\n  gallery total : {gallery.shape[0]:,} embeddings from {names}")

    # ---- sample QMUL training identities ----
    rng = np.random.default_rng(args.seed)
    id_dirs = sorted([d for d in QMUL.iterdir() if d.is_dir()], key=lambda p: p.name)
    print(f"  QMUL identities: {len(id_dirs):,}")

    paths, labels, dims = [], [], []
    for d in id_dirs:
        fs = sorted(d.glob("*.jpg"))
        if len(fs) > args.per_identity:
            fs = [fs[i] for i in rng.choice(len(fs), args.per_identity, replace=False)]
        for f in fs:
            paths.append(f)
            labels.append(d.name)
    print(f"  sampled {len(paths):,} images at <= {args.per_identity}/identity")

    model = load_recognizer(args.model)
    print("  embedding QMUL images ...")
    embs = np.zeros((len(paths), 512), dtype=np.float32)
    B = 64
    for i in range(0, len(paths), B):
        chunk = paths[i : i + B]
        imgs = []
        for f in chunk:
            im = cv2.imdecode(np.frombuffer(f.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                im = np.zeros((112, 112, 3), np.uint8)
            else:
                dims.append(im.shape[:2])
            imgs.append(cv2.resize(im, (112, 112)))
        embs[i : i + len(chunk)] = (
            np.asarray(model.get_feat(imgs))
            + np.asarray(model.get_feat([x[:, ::-1] for x in imgs]))
        )
        if (i // B) % 100 == 0:
            print(f"    {min(i + B, len(paths)):,}/{len(paths):,}", end="\r", flush=True)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    print(f"  embedded {len(paths):,} images                    ")

    # ---- exact nearest neighbour ----
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(np.ascontiguousarray(gallery))
    sims, idx = index.search(np.ascontiguousarray(embs), 1)
    best, who = sims[:, 0], gsrc[idx[:, 0]]

    per_id: dict[str, float] = defaultdict(lambda: -2.0)
    per_id_src: dict[str, int] = {}
    for lab, s, w in zip(labels, best.tolist(), who.tolist()):
        if s > per_id[lab]:
            per_id[lab] = s
            per_id_src[lab] = w

    n_ids = len(per_id)
    print(f"\n  identity max-similarity distribution ({n_ids:,} identities)")
    for lo, hi in [(0.9, 2.0), (0.7, 0.9), (0.5, 0.7), (0.4, 0.5),
                   (0.35, 0.4), (0.3, 0.35), (-2.0, 0.3)]:
        n = sum(1 for s in per_id.values() if lo <= s < hi)
        band = f">={lo:.2f}" if hi > 1 else f"{lo:.2f}-{hi:.2f}"
        print(f"    {band:>12s}  {n:>6,}  {n / n_ids * 100:5.1f}%")

    print("\n  threshold sensitivity")
    for t in (0.30, 0.35, 0.40, 0.45, 0.50):
        n = sum(1 for s in per_id.values() if s >= t)
        mark = "  <- primary (matches CASIA audit)" if abs(t - args.threshold) < 1e-9 else ""
        print(f"    >= {t:.2f} : exclude {n:>6,} ({n / n_ids * 100:5.1f}%){mark}")

    excluded = sorted(i for i, s in per_id.items() if s >= args.threshold)
    kept = sorted(i for i, s in per_id.items() if s < args.threshold)

    print("\n  which eval set the nearest neighbour came from (ALL identities)")
    tally = defaultdict(int)
    for lab in per_id:
        tally[names[per_id_src[lab]]] += 1
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"    {k:12s} {v:>6,}  {v / n_ids * 100:5.1f}%")

    if excluded:
        print("\n  ... and among the EXCLUDED identities only")
        tally_x = defaultdict(int)
        for lab in excluded:
            tally_x[names[per_id_src[lab]]] += 1
        for k, v in sorted(tally_x.items(), key=lambda kv: -kv[1]):
            print(f"    {k:12s} {v:>6,}  {v / len(excluded) * 100:5.1f}%")

    print(f"\n  EXCLUDE : {len(excluded):,} identities "
          f"({len(excluded) / n_ids * 100:.1f}%) at threshold {args.threshold}")
    print(f"  KEEP    : {len(kept):,} identities ({len(kept) / n_ids * 100:.1f}%)")
    print(f"  peak similarity anywhere : {float(best.max()):.4f}")

    if dims:
        a = np.array(dims)
        print(f"\n  native resolution of sampled QMUL images (pre-resize)")
        print(f"    median {int(np.median(a[:, 0]))}x{int(np.median(a[:, 1]))} px   "
              f"min {a[:, 0].min()}x{a[:, 1].min()}   max {a[:, 0].max()}x{a[:, 1].max()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": "QMUL-SurvFace training_set",
        "license": "research purposes only; images sourced from person re-identification "
                   "datasets, copyright with original owners (qmul-survface.github.io)",
        "model": args.model,
        "threshold": args.threshold,
        "per_identity_sampled": args.per_identity,
        "images_sampled": len(paths),
        "eval_sets": names,
        "gallery_embeddings": int(gallery.shape[0]),
        "identities_total": n_ids,
        "identities_excluded": len(excluded),
        "identities_kept": len(kept),
        "peak_similarity": round(float(best.max()), 4),
        "nearest_eval_set_tally": dict(tally),
        "threshold_sensitivity": {
            f"{t:.2f}": sum(1 for s in per_id.values() if s >= t)
            for t in (0.30, 0.35, 0.40, 0.45, 0.50)
        },
        "caveat": (
            "Degraded probes yield weaker embeddings, compressing cosine similarity "
            "downward for true matches too. A 0.40 threshold carried over from the "
            "clean-vs-clean CASIA audit is a LOOSER filter here, not a stricter one."
        ),
        "excluded_labels": excluded,
        "kept_labels": kept,
    }, indent=2))
    print(f"\n  Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

## `backend/scripts/qmul_overlap_control.py`

```python
#!/usr/bin/env python
"""
CONTROL for the QMUL-SurvFace overlap audit.

    python backend/scripts/qmul_overlap_control.py

The audit reported 96.9% of QMUL identities above the 0.40 exclusion threshold,
with 78% of nearest neighbours landing in TinyFace. Taken at face value that
says the dataset is almost entirely contaminated.

That reading is probably WRONG, and this script exists to find out before any
decision is made on it.

THE COMPETING EXPLANATION
------------------------
ArcFace embeddings of very low quality faces are known to collapse toward a
common region of the hypersphere. A 26x21px blurred face carries little identity
signal, so what the embedding mostly encodes is "degraded face", not "this
person". Two unrelated degraded faces can then sit at cosine 0.6 purely because
both are degraded. QMUL is native surveillance capture and TinyFace is native
low-resolution capture, so a QMUL-to-TinyFace affinity is exactly what this
artefact would produce -- with no shared identities at all.

THE DISCRIMINATING TEST
-----------------------
Measure the IMPOSTOR floor within QMUL itself: similarity between images of
DIFFERENT QMUL identities. Ground truth is known here -- the dataset is ordered
by identity, so different directories are different people by construction.

  If different-person QMUL pairs also score ~0.5-0.7, then 0.5-0.7 is simply the
  noise floor for degraded imagery, the audit threshold is meaningless at that
  scale, and the "96.9% contamination" is an artefact.

  If different-person QMUL pairs score far lower (~0.1-0.2) while the nearest
  TinyFace neighbour scores 0.6+, the eval-set affinity is specific and the
  contamination is real.

A genuine-pair distribution (same identity, different images) is measured too,
to show where a true match actually sits under these conditions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "scripts"))

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
QMUL = Path("C:/Users/hello/Downloads/QMUL-SurvFace-v1/QMUL-SurvFace/training_set")


def pct(a: np.ndarray) -> str:
    q = np.percentile(a, [5, 25, 50, 75, 95])
    return f"p5 {q[0]:.3f}  p25 {q[1]:.3f}  MEDIAN {q[2]:.3f}  p75 {q[3]:.3f}  p95 {q[4]:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="w600k_r50")
    ap.add_argument("--identities", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/qmul_overlap_control.json"))
    args = ap.parse_args()

    from benchmark_verification import load_recognizer

    rng = np.random.default_rng(args.seed)
    print("=" * 78)
    print("  CONTROL - is the QMUL/TinyFace affinity identity, or just low quality?")
    print("=" * 78)

    # Two images per identity, so genuine and impostor come from the same pool.
    id_dirs = sorted([d for d in QMUL.iterdir() if d.is_dir()])
    usable = []
    for d in id_dirs:
        fs = sorted(d.glob("*.jpg"))
        if len(fs) >= 2:
            usable.append((d.name, fs))
    if len(usable) > args.identities:
        sel = rng.choice(len(usable), args.identities, replace=False)
        usable = [usable[i] for i in sel]
    print(f"  QMUL identities with >=2 images: {len(usable):,}")

    paths, owner = [], []
    for name, fs in usable:
        pick = rng.choice(len(fs), 2, replace=False)
        for i in pick:
            paths.append(fs[int(i)])
            owner.append(name)

    model = load_recognizer(args.model)
    print(f"  embedding {len(paths):,} QMUL images ...")
    e = np.zeros((len(paths), 512), dtype=np.float32)
    B = 64
    for i in range(0, len(paths), B):
        chunk = paths[i : i + B]
        imgs = []
        for f in chunk:
            im = cv2.imdecode(np.frombuffer(f.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            imgs.append(cv2.resize(im if im is not None else np.zeros((8, 8, 3), np.uint8),
                                   (112, 112)))
        e[i : i + len(chunk)] = (np.asarray(model.get_feat(imgs))
                                 + np.asarray(model.get_feat([x[:, ::-1] for x in imgs])))
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12

    a, b = e[0::2], e[1::2]          # a[i], b[i] are the same identity
    n = len(a)

    genuine = np.sum(a * b, axis=1)

    # Impostor: pair each identity with a DIFFERENT one. Ground truth by
    # construction -- separate directories are separate people.
    perm = rng.permutation(n)
    bad = perm == np.arange(n)
    perm[bad] = (perm[bad] + 1) % n
    impostor = np.sum(a * b[perm], axis=1)

    print(f"\n  QMUL genuine  (SAME person, {n:,} pairs)")
    print(f"    {pct(genuine)}")
    print(f"  QMUL impostor (DIFFERENT people, {n:,} pairs)  <- THE CONTROL")
    print(f"    {pct(impostor)}")

    # ------------------------------------------------------------------
    # MATCHED max-of-N control.
    #
    # A first version of this script compared the nearest TinyFace neighbour
    # (a MAXIMUM over 8,171 candidates) against a single random QMUL impostor
    # pair (ONE draw), and concluded the affinity was "specific". That
    # comparison is invalid: the maximum of 8,171 draws is far above a single
    # draw whatever the underlying distribution, so it would have declared
    # contamination even on unrelated data.
    #
    # The correct null is the maximum over the SAME number of candidates drawn
    # from identities that are known-different. QMUL directories are distinct
    # people by construction, so that null is directly measurable.
    # ------------------------------------------------------------------
    tf = CACHE / f"tinyface_labelled__{args.model}.npz"
    near_tf = near_qmul = None
    if tf.exists():
        import faiss

        g = np.load(tf)["emb"].astype(np.float32)
        n_cand = len(g)
        idx = faiss.IndexFlatIP(g.shape[1])
        idx.add(np.ascontiguousarray(g))
        sims, _ = idx.search(np.ascontiguousarray(a), 1)
        near_tf = sims[:, 0]

        # Null pool: `n_cand` QMUL images whose identities are disjoint from
        # the probes, so every candidate is guaranteed to be a different person.
        half = n // 2
        probe_ids = {owner[2 * i] for i in range(half)}
        pool = []
        for name, fs in usable:
            if name in probe_ids:
                continue
            pool.extend(fs[: min(len(fs), 8)])
        if len(pool) > n_cand:
            sel = rng.choice(len(pool), n_cand, replace=False)
            pool = [pool[i] for i in sel]
        print(f"\n  matched null pool: {len(pool):,} QMUL images from "
              f"{len(usable) - len(probe_ids):,} DIFFERENT identities")
        pe = np.zeros((len(pool), 512), dtype=np.float32)
        for i in range(0, len(pool), B):
            chunk = pool[i : i + B]
            imgs = []
            for f in chunk:
                im = cv2.imdecode(np.frombuffer(f.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                imgs.append(cv2.resize(im if im is not None else np.zeros((8, 8, 3), np.uint8),
                                       (112, 112)))
            pe[i : i + len(chunk)] = (np.asarray(model.get_feat(imgs))
                                      + np.asarray(model.get_feat([x[:, ::-1] for x in imgs])))
        pe /= np.linalg.norm(pe, axis=1, keepdims=True) + 1e-12
        idx2 = faiss.IndexFlatIP(pe.shape[1])
        idx2.add(np.ascontiguousarray(pe))
        s2, _ = idx2.search(np.ascontiguousarray(a[:half]), 1)
        near_qmul = s2[:, 0]

        print(f"\n  nearest TINYFACE neighbour   (max over {n_cand:,} candidates)")
        print(f"    {pct(near_tf)}")
        print(f"  nearest DIFFERENT-PERSON QMUL (max over {len(pool):,}) <- MATCHED NULL")
        print(f"    {pct(near_qmul)}")

    # A clean-vs-clean reference: what does an impostor pair look like when
    # quality is good? Straight from the cached LFW pairs.
    lfw = CACHE / f"lfw__{args.model}.npz"
    lfw_imp = None
    if lfw.exists():
        d = np.load(lfw)
        le = (d["orig"] + d["flip"]).astype(np.float64)
        le /= np.linalg.norm(le, axis=1, keepdims=True) + 1e-12
        iss = np.asarray(d["issame"], dtype=bool)
        s = np.sum(le[0::2] * le[1::2], axis=1)
        lfw_imp, lfw_gen = s[~iss], s[iss]
        print(f"\n  REFERENCE - LFW (clean) impostor pairs")
        print(f"    {pct(lfw_imp)}")
        print(f"  REFERENCE - LFW (clean) genuine pairs")
        print(f"    {pct(lfw_gen)}")

    print("\n" + "=" * 78)
    verdict = None
    if near_tf is not None and near_qmul is not None:
        # Like for like: max-over-N vs max-over-N, one pool known-different.
        sep = float(np.median(near_tf) - np.median(near_qmul))
        print(f"  median nearest-TinyFace {np.median(near_tf):.3f}  vs  "
              f"matched different-person QMUL null {np.median(near_qmul):.3f}"
              f"   separation {sep:+.3f}")
        print(f"  for scale: a TRUE same-person QMUL pair medians "
              f"{np.median(genuine):.3f}")
        if sep < 0.05:
            verdict = "ARTEFACT"
            print("\n  VERDICT: ARTEFACT, not contamination.")
            print("  Against a matched null of images that are certainly different")
            print("  people, the TinyFace affinity all but disappears. The audit's")
            print("  0.40 threshold is measuring IMAGE QUALITY, not identity: degraded")
            print("  embeddings collapse toward each other regardless of who they are.")
            print("  The 96.9% exclusion figure must NOT be reported as contamination.")
        else:
            verdict = "SPECIFIC"
            print(f"\n  VERDICT: affinity is SPECIFIC ({sep:+.3f} above a matched null).")
            print("  The overlap survives the control; act on the exclusion list.")
    print("=" * 78)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "verdict": verdict,
        "pairs": int(n),
        "qmul_genuine": {"median": round(float(np.median(genuine)), 4),
                         "p5": round(float(np.percentile(genuine, 5)), 4),
                         "p95": round(float(np.percentile(genuine, 95)), 4)},
        "qmul_impostor_control": {"median": round(float(np.median(impostor)), 4),
                                  "p95": round(float(np.percentile(impostor, 95)), 4)},
        "nearest_tinyface_max_of_N": None if near_tf is None else {
            "median": round(float(np.median(near_tf)), 4),
            "p95": round(float(np.percentile(near_tf, 95)), 4)},
        "matched_null_diff_person_qmul_max_of_N": None if near_qmul is None else {
            "median": round(float(np.median(near_qmul)), 4),
            "p95": round(float(np.percentile(near_qmul, 95)), 4)},
        "lfw_clean_impostor": None if lfw_imp is None else {
            "median": round(float(np.median(lfw_imp)), 4),
            "p95": round(float(np.percentile(lfw_imp, 95)), 4)},
    }, indent=2))
    print(f"\n  Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

## `backend/scripts/eval_finetuned_checkpoint.py`

```python
#!/usr/bin/env python
"""
Phase 6 step 4 (item 42) — score a fine-tuned checkpoint on EVERY benchmark,
against the deployed model, on identical inputs.

    python backend/scripts/eval_finetuned_checkpoint.py

The point of this script is that it cannot flatter the checkpoint. The
fine-tuned backbone is wrapped in a shim exposing the same `get_feat(images)`
signature insightface's recogniser has, so the pair lists, the flip
augmentation, the 10-fold cross-validation and the threshold fitting are the
SAME CODE that produced the numbers in BENCHMARKS.md §2 and §4. The only thing
that changes between the two columns is the weights.

Both models are scored in this run rather than reading the baseline from cache,
so a stale cache cannot produce a fake improvement.

REGRESSIONS ARE REPORTED, NOT HIDDEN. A fine-tune that trades clean accuracy
for degraded accuracy is a real and possibly acceptable trade, but it is only
assessable if both halves are printed.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "scripts"))

from nexgen_engine.benchmarks.verification import (  # noqa: E402
    decode_pack,
    evaluate_pairs,
    l2n,
    load_pack,
)

_ROOT = _BACKEND.parent
CLEAN = ["lfw", "agedb_30", "cfp_fp", "cfp_ff", "calfw", "cplfw"]
TINYFACE = _ROOT / "src_extracted/tinyface/tinyface/Testing_Set"
ID_RE = re.compile(r"^(\d+)_")


class TorchRecognizer:
    """Adapts the fine-tuned torch backbone to insightface's get_feat()."""

    def __init__(self, ckpt: Path, device: torch.device):
        import onnx
        import onnx2torch

        p = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
        self.net = onnx2torch.convert(onnx.load_model(io.BytesIO(p.read_bytes())))
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.net.load_state_dict(state["backbone"])
        self.net.to(device).eval()
        self.device = device
        self.meta = {k: v for k, v in state.items() if k != "backbone"}

    @torch.no_grad()
    def get_feat(self, images) -> np.ndarray:
        a = np.stack([np.ascontiguousarray(im) for im in images]).astype(np.float32)
        a = a[..., ::-1].copy()
        a = (a - 127.5) / 127.5
        t = torch.from_numpy(a).permute(0, 3, 1, 2).to(self.device)
        return self.net(t).cpu().numpy()


def embed_all(model, images, batch: int) -> np.ndarray:
    """Original + horizontal flip, summed — the same augmentation §2 uses."""
    n = len(images)
    out = np.zeros((n, 512), dtype=np.float32)
    for i in range(0, n, batch):
        chunk = images[i : i + batch]
        out[i : i + len(chunk)] = (
            np.asarray(model.get_feat([im for im in chunk]))
            + np.asarray(model.get_feat([im[:, ::-1] for im in chunk]))
        )
    return out


def tinyface_pairs(n_pairs: int, seed: int):
    by_id: dict[str, list[Path]] = defaultdict(list)
    for sub in ("Gallery_Match", "Probe"):
        d = TINYFACE / sub
        if d.is_dir():
            for p in d.glob("*.jpg"):
                m = ID_RE.match(p.name)
                if m:
                    by_id[m.group(1)].append(p)
    multi = {k: v for k, v in by_id.items() if len(v) >= 2}
    if len(multi) < 2:
        return None
    rng = np.random.default_rng(seed)  # seed 0 == BENCHMARKS.md §4, same pairs
    ids = sorted(multi)
    genuine, impostor = [], []
    while len(genuine) < n_pairs:
        imgs = multi[ids[rng.integers(0, len(ids))]]
        a, b = rng.choice(len(imgs), 2, replace=False)
        genuine.append((imgs[a], imgs[b]))
    while len(impostor) < n_pairs:
        i, j = ids[rng.integers(0, len(ids))], ids[rng.integers(0, len(ids))]
        if i == j:
            continue
        impostor.append((multi[i][rng.integers(0, len(multi[i]))],
                         multi[j][rng.integers(0, len(multi[j]))]))
    pairs, labels = [], []
    for g, im in zip(genuine, impostor):
        pairs.append(g); labels.append(True)
        pairs.append(im); labels.append(False)
    return pairs, np.array(labels, dtype=bool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(_ROOT / "runtime/checkpoints/arcface_degraded_v1.pt"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tinyface-pairs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/finetuned_v1.json"))
    args = ap.parse_args()

    from benchmark_verification import find_pack, load_recognizer

    ck = Path(args.checkpoint)
    if not ck.exists():
        print(f"no checkpoint at {ck} — run finetune_degraded.py first")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("  Phase 6 step 4 (item 42) - fine-tuned checkpoint vs deployed model")
    print("=" * 78)

    tuned = TorchRecognizer(ck, device)
    print(f"  checkpoint : {ck.name}  {tuned.meta}")
    base = load_recognizer("w600k_r50")
    print("  baseline   : w600k_r50 (DEPLOYED)\n")

    rows, payload = [], {}

    for ds in CLEAN:
        try:
            bins, issame = load_pack(find_pack(ds))
        except Exception as exc:
            print(f"  {ds}: unavailable ({exc}); SKIPPED")
            continue
        images = decode_pack(bins)
        issame = np.asarray(issame, dtype=bool)
        res = {}
        for name, m in (("deployed", base), ("finetuned", tuned)):
            e = l2n(embed_all(m, images, args.batch_size).astype(np.float64))
            r = evaluate_pairs(e[0::2], e[1::2], issame, ds, name)
            res[name] = r
        rows.append((ds, res["deployed"], res["finetuned"]))
        payload[ds] = {k: {"accuracy_pct": round(v.accuracy_mean * 100, 3),
                           "tar_far_1e3_pct": round(v.tar_at_far_1e3 * 100, 3),
                           "auc": round(v.auc, 5)} for k, v in res.items()}
        print(f"  scored {ds}", flush=True)

    tf = tinyface_pairs(args.tinyface_pairs, args.seed)
    if tf is None:
        print("  tinyface: not available; SKIPPED (this is the TARGET condition —")
        print("  without it the fine-tune cannot be said to have helped or not)")
    else:
        pairs, labels = tf
        needed = sorted({p for pr in pairs for p in pr})
        imgs = []
        for p in needed:
            im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            imgs.append(cv2.resize(im, (112, 112)))
        pos = {p: i for i, p in enumerate(needed)}
        res = {}
        for name, m in (("deployed", base), ("finetuned", tuned)):
            e = l2n(embed_all(m, imgs, args.batch_size).astype(np.float64))
            a = np.stack([e[pos[p[0]]] for p in pairs])
            b = np.stack([e[pos[p[1]]] for p in pairs])
            res[name] = evaluate_pairs(a, b, labels, "tinyface", name)
        rows.append(("tinyface", res["deployed"], res["finetuned"]))
        payload["tinyface"] = {k: {"accuracy_pct": round(v.accuracy_mean * 100, 3),
                                   "tar_far_1e3_pct": round(v.tar_at_far_1e3 * 100, 3),
                                   "auc": round(v.auc, 5)} for k, v in res.items()}
        print("  scored tinyface", flush=True)

    print(f"\n  {'dataset':12s} {'deployed':>10s} {'finetuned':>10s} {'delta':>9s}   verdict")
    print("  " + "-" * 62)
    for ds, d, f in rows:
        da, fa = d.accuracy_mean * 100, f.accuracy_mean * 100
        delta = fa - da
        # 1 std of the 10-fold mean is the smallest difference that means
        # anything here; below it the two models are indistinguishable.
        tol = max(d.accuracy_std, f.accuracy_std) * 100
        verdict = "BETTER" if delta > tol else ("WORSE" if delta < -tol else "no change")
        print(f"  {ds:12s} {da:>9.2f}% {fa:>9.2f}% {delta:>+8.2f}pp   {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(ck), "checkpoint_meta": tuned.meta,
        "baseline_model": "w600k_r50 (deployed)",
        "note": "Both models scored in the same run on identical pair lists.",
        "results": payload,
    }, indent=2, default=str))
    print(f"\n  Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

