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
