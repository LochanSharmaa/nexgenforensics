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
