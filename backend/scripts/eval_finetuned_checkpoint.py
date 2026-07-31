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
CLEAN = ["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"]
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
