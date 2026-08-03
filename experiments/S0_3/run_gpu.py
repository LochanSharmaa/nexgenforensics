#!/usr/bin/env python
"""S0.3 (G2) -- the programme go/no-go, run with the real recogniser.

    python experiments/S0_3/run_gpu.py --dataset tinyface
    python experiments/S0_3/run_gpu.py --dataset qmul
    python experiments/S0_3/run_gpu.py --dataset lfw_synth

THE QUESTION
------------
Does modelling the probe's actual imaging operator and applying it FORWARD to the
gallery beat simply matching resolution?

    A   no transform                        the current paradigm
    B1  downsample gallery to probe size    the well-known trick
    B2  estimate probe's PSF/JPEG/noise, apply that whole operator to gallery
    B3  project both onto their common MTF passband
    C   pixel-space likelihood, no embedding at all

**B2 - B1 is the number that decides the architecture.** The decision rule is
registered in arms/run.py BEFORE the run, not chosen afterwards:

    PASS  >= +2.0 points TAR@FAR=0.1%, bootstrap CI excluding zero  -> Stage 3 licensed
    FAIL  <= +0.5 points, or CI includes zero                       -> Stage 3 cancelled

A FAIL is a good outcome: it saves roughly two years and redirects effort to the
evidence layer, which our Cllr decomposition already shows is nearly exhausted at
this resolution.

THREE DATASETS, TWO ROLES
-------------------------
lfw_synth  MECHANISM CHECK. Real HR faces, probe side degraded by a KNOWN
           operator. If B2 cannot win here -- where the ground-truth operator is
           exactly what B2 tries to estimate -- it cannot win anywhere, and a
           failure is attributable to the idea rather than to estimation error.

tinyface   THE DECISION. Native low-resolution both sides, official
           Gallery_Match x Probe split. Note B1 is nearly a no-op here because
           both sides are already small, so on this dataset B2-B1 ~ B2-A.

qmul       REPLICATION. Independent surveillance corpus.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_BACKEND = _ROOT / "backend"
for p in (str(_BACKEND), str(_HERE), str(_BACKEND / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from arms import ARMS, arm_C_score  # noqa: E402

RESULTS = _HERE / "results"
DECISION = {"pass_threshold": 2.0, "fail_threshold": 0.5, "metric": "TAR@FAR=0.1% (points)"}
ID_RE = re.compile(r"^(\d+)_\d+\.jpg$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def tar_at_far(scores: np.ndarray, labels: np.ndarray, far: float = 1e-3) -> float:
    imp = scores[~labels]
    if imp.size == 0:
        return float("nan")
    return float((scores[labels] > np.quantile(imp, 1.0 - far)).mean())


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    n_p, n_n = int(labels.sum()), int((~labels).sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    return float((ranks[labels].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def paired_bootstrap(
    s_a: np.ndarray, s_b: np.ndarray, labels: np.ndarray, far: float = 1e-3,
    n_boot: int = 2000, seed: int = 0,
) -> tuple[float, float, float]:
    """CI on (TAR_b - TAR_a) in points, resampling PAIRS so the two arms stay paired."""
    rng = np.random.default_rng(seed)
    n = labels.size
    point = (tar_at_far(s_b, labels, far) - tar_at_far(s_a, labels, far)) * 100.0
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        lab = labels[idx]
        if lab.sum() == 0 or (~lab).sum() == 0:
            diffs[i] = np.nan
            continue
        diffs[i] = (tar_at_far(s_b[idx], lab, far) - tar_at_far(s_a[idx], lab, far)) * 100.0
    d = diffs[np.isfinite(diffs)]
    return point, float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


# --------------------------------------------------------------------------- #
# embedder
# --------------------------------------------------------------------------- #
def load_embedder(kind: str, batch_size: int = 96):
    if kind == "stub":
        from arms import stub_embedder
        return stub_embedder, "stub (CPU, not a recogniser)"
    if kind != "arcface":
        raise SystemExit(f"unknown embedder: {kind}")

    from benchmark_verification import load_recognizer
    model = load_recognizer("w600k_r50")  # raises unless CUDA binds

    def embed(images: list[np.ndarray]) -> np.ndarray:
        out = np.empty((len(images), 512), dtype=np.float64)
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            prepped = []
            for im in chunk:
                x = np.asarray(im, dtype=np.float64)
                if x.ndim == 2:
                    x = np.repeat(x[:, :, None], 3, axis=2)
                x = np.clip(x, 0, 1) * 255.0
                prepped.append(cv2.resize(x.astype(np.uint8), (112, 112)))
            out[i : i + len(chunk)] = np.asarray(model.get_feat(prepped), dtype=np.float64)
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)

    return embed, "w600k_r50 (CUDA)"


# --------------------------------------------------------------------------- #
# datasets -> (gallery_images, probe_images, labels)
# --------------------------------------------------------------------------- #
def _read(p: Path, gray: bool = False) -> np.ndarray:
    """Decode to float32, not float64.

    At 20,000 pairs the LFW gallery side alone is 20,000 x 250x250x3. In float64
    that is ~30 GB and the machine swaps -- a run that took 282 s to load at
    float64 on a smaller set stalled past 25 minutes here. float32 halves it and
    costs nothing: these are 8-bit pixel values, so the extra mantissa was never
    carrying information. The arms upcast internally where they need to.
    """
    im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"decode failed: {p}")
    return (im.astype(np.float32) / np.float32(255.0))


def load_tinyface(n_pairs: int, seed: int) -> tuple[list, list, np.ndarray]:
    """OFFICIAL split: gallery from Gallery_Match, probes from Probe.

    Using arbitrary within-identity pairs instead inflates results badly -- 39.4%
    of such pairs are same-subset near-duplicate track frames, mean score 0.4225
    vs 0.3268 for the official pairing. This loader exists to prevent that.
    """
    root = _ROOT / "src_extracted/tinyface/tinyface/Testing_Set"
    gal_by, prb_by = defaultdict(list), defaultdict(list)
    for sub, store in (("Gallery_Match", gal_by), ("Probe", prb_by)):
        for f in (root / sub).glob("*.jpg"):
            m = ID_RE.match(f.name)
            if m:
                store[int(m.group(1))].append(f)
    both = sorted(set(gal_by) & set(prb_by))
    rng = np.random.default_rng(seed)
    gal, prb, lab = [], [], []
    for _ in range(n_pairs // 2):
        i = both[rng.integers(0, len(both))]
        gal.append(gal_by[i][rng.integers(0, len(gal_by[i]))])
        prb.append(prb_by[i][rng.integers(0, len(prb_by[i]))])
        lab.append(True)
        j = both[rng.integers(0, len(both))]
        while j == i:
            j = both[rng.integers(0, len(both))]
        gal.append(gal_by[i][rng.integers(0, len(gal_by[i]))])
        prb.append(prb_by[j][rng.integers(0, len(prb_by[j]))])
        lab.append(False)
    with ThreadPoolExecutor(max_workers=8) as pool:
        G = list(pool.map(_read, gal))
        P = list(pool.map(_read, prb))
    return G, P, np.array(lab, dtype=bool)


def load_qmul(n_pairs: int, seed: int) -> tuple[list, list, np.ndarray]:
    for r in (Path("C:/Users/hello/Downloads/QMUL-SurvFace-v1/QMUL-SurvFace"),
              _ROOT / "src_extracted/QMUL-SurvFace"):
        if (r / "training_set").is_dir():
            root = r
            break
    else:
        raise SystemExit("QMUL not found")
    by = defaultdict(list)
    for d in sorted((root / "training_set").iterdir()):
        if d.is_dir():
            imgs = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
            if len(imgs) >= 2:
                by[d.name] = imgs
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    gal, prb, lab = [], [], []
    for _ in range(n_pairs // 2):
        i = keys[rng.integers(0, len(keys))]
        a, b = rng.choice(len(by[i]), 2, replace=False)
        gal.append(by[i][a]); prb.append(by[i][b]); lab.append(True)
        j = keys[rng.integers(0, len(keys))]
        while j == i:
            j = keys[rng.integers(0, len(keys))]
        gal.append(by[i][a]); prb.append(by[j][rng.integers(0, len(by[j]))]); lab.append(False)
    with ThreadPoolExecutor(max_workers=8) as pool:
        G = list(pool.map(_read, gal))
        P = list(pool.map(_read, prb))
    return G, P, np.array(lab, dtype=bool)


def load_lfw_synth(n_pairs: int, seed: int, downsample: float = 8.0) -> tuple[list, list, np.ndarray]:
    """MECHANISM CHECK: real HR faces, probe degraded by a KNOWN operator."""
    from nexgen_engine.degradation.psf import DegradationParams, apply_forward

    root = _ROOT / "src_extracted/lfw_deepfunneled/lfw-deepfunneled/lfw-deepfunneled"
    people = [d for d in sorted(root.iterdir()) if d.is_dir()]
    multi = [d for d in people if len(list(d.glob("*.jpg"))) >= 2]
    rng = np.random.default_rng(seed)
    op = DegradationParams(blur_sigma=1.6, downsample=downsample, noise_sigma=0.02, jpeg_quality=35)

    gal, prb, lab = [], [], []
    for _ in range(n_pairs // 2):
        d = multi[rng.integers(0, len(multi))]
        imgs = sorted(d.glob("*.jpg"))
        a, b = rng.choice(len(imgs), 2, replace=False)
        gal.append(imgs[a]); prb.append(imgs[b]); lab.append(True)
        e = people[rng.integers(0, len(people))]
        while e == d:
            e = people[rng.integers(0, len(people))]
        gal.append(imgs[a]); prb.append(sorted(e.glob("*.jpg"))[0]); lab.append(False)

    with ThreadPoolExecutor(max_workers=8) as pool:
        G = list(pool.map(_read, gal))
        raw = list(pool.map(_read, prb))
    # Degrade in place and release each source immediately: holding both the
    # full-resolution originals and the degraded probes doubles peak memory for
    # no reason, and the originals are never used again.
    P = []
    for k in range(len(raw)):
        P.append(apply_forward(raw[k], op, seed=k).astype(np.float32))
        raw[k] = None
    del raw
    return G, P, np.array(lab, dtype=bool)



def load_scface(n_pairs: int, seed: int, distance: int = 1) -> tuple[list, list, np.ndarray]:
    """SCface: REAL mugshot vs REAL CCTV at a known standoff. The decisive test.

    Every other S0.3 dataset is compromised for the asymmetric case:
      lfw_synth  HR/LR asymmetry exists, but the degradation is OUR operator and
                 arm B2r estimates a simplified form of it -- part of the +3.55
                 margin may be the experiment recovering its own assumption.
      tinyface   both sides low-resolution; no operator asymmetry to exploit.
      qmul       same, and 100% cross-camera.

    SCface has a genuine 1600x1200 mugshot gallery against real surveillance
    cameras at three MEASURED distances, so it tests the forward-operator thesis
    on optics nobody in this project chose:

        distance 1 = 4.20 m   ~100x75 px   most degraded
        distance 2 = 2.60 m   ~144x108 px
        distance 3 = 1.00 m   ~224x168 px  least degraded

    PRE-REGISTERED PREDICTION (DATA_ACQUISITION_REQUEST.md): if the thesis is
    real, B2r-B1 must (a) clear +2.0 pts at distance 1 with CI excluding zero,
    and (b) DECREASE monotonically as standoff shrinks -- the advantage should
    track how much degradation there is to model. A flat or inverted ordering
    refutes it even if a single distance passes, because a genuine mechanism
    cannot be indifferent to the amount of degradation present.
    """
    root = _ROOT / "src_extracted/scface/SCface_database"
    mug_dir = root / "mugshot_frontal_cropped_all"
    sur_dir = root / f"surveillance_cameras_distance_{distance}"
    if not (mug_dir.is_dir() and sur_dir.is_dir()):
        raise SystemExit(f"SCface not found under {root}")

    mugs: dict[str, Path] = {}
    for f in mug_dir.glob("*"):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            mugs[f.name.split("_")[0]] = f

    sur: dict[str, list[Path]] = defaultdict(list)
    for cam in sorted(p for p in sur_dir.iterdir() if p.is_dir()):
        for f in cam.glob("*"):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                sur[f.name.split("_")[0]].append(f)

    subjects = sorted(set(mugs) & set(sur))
    if len(subjects) < 2:
        raise SystemExit("SCface: could not pair mugshots to surveillance by subject id")

    rng = np.random.default_rng(seed)
    gal, prb, lab = [], [], []
    for _ in range(n_pairs // 2):
        i = subjects[rng.integers(0, len(subjects))]
        gal.append(mugs[i])
        prb.append(sur[i][rng.integers(0, len(sur[i]))])
        lab.append(True)
        j = subjects[rng.integers(0, len(subjects))]
        while j == i:
            j = subjects[rng.integers(0, len(subjects))]
        gal.append(mugs[i])
        prb.append(sur[j][rng.integers(0, len(sur[j]))])
        lab.append(False)

    with ThreadPoolExecutor(max_workers=8) as pool:
        G = list(pool.map(_read, gal))
        P = list(pool.map(_read, prb))
    return G, P, np.array(lab, dtype=bool)


LOADERS = {
    "tinyface": load_tinyface,
    "qmul": load_qmul,
    "lfw_synth": load_lfw_synth,
    "scface": load_scface,
}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--embedder", default="arcface", choices=["stub", "arcface"])
    ap.add_argument("--pairs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distance", type=int, default=1, choices=[1, 2, 3],
                    help="SCface standoff: 1=4.20m, 2=2.60m, 3=1.00m")
    ap.add_argument("--chunk", type=int, default=2500,
                    help="pairs held in memory at once; bounds peak RSS")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    embed, embed_name = load_embedder(args.embedder)
    print(f"embedder: {embed_name}")

    # CHUNKED EXECUTION -- memory, not GPU, is the binding constraint here.
    #
    # lfw_synth at 20,000 pairs holds 20,000 gallery images at 250x250x3. In
    # float64 that is ~30 GB and the machine swapped (a run stalled past 25
    # minutes without printing a single arm); float32 halved it to 15.5 GB, which
    # still thrashes. Neither is a GPU limit -- the A3000 never exceeded ~2 GB.
    #
    # So pairs are loaded, transformed, embedded and scored in bounded chunks and
    # the images are released immediately. Scores are identical to the monolithic
    # path because every arm and the embedder are per-pair pure; only peak RSS
    # changes. Chunking also means --pairs is no longer limited by RAM.
    chunk = max(1, min(args.chunk, args.pairs))
    n_chunks = (args.pairs + chunk - 1) // chunk

    arm_names = list(ARMS) + ["C"]
    acc: dict[str, list[np.ndarray]] = {k: [] for k in arm_names}
    all_labels: list[np.ndarray] = []
    reports: dict[str, dict] = {}
    dims: list[tuple[int, int]] = []

    t0 = time.time()
    for ci in range(n_chunks):
        take = min(chunk, args.pairs - ci * chunk)
        # Distinct seed per chunk so pairs are not re-drawn identically.
        if args.dataset == "scface":
            G, P, lab = load_scface(take, args.seed + ci * 7919, args.distance)
        else:
            G, P, lab = LOADERS[args.dataset](take, args.seed + ci * 7919)
        all_labels.append(lab)
        if not dims:
            dims = [(int(np.median([g.shape[0] for g in G])),
                     int(np.median([q.shape[0] for q in P])))]

        for name, fn in ARMS.items():
            gt, pt, rep = [], [], None
            with ThreadPoolExecutor(max_workers=8) as pool:
                for g, q, r in pool.map(lambda gp: fn(gp[0], gp[1]), zip(G, P)):
                    gt.append(g); pt.append(q); rep = rep or r
            acc[name].append(np.sum(embed(gt) * embed(pt), axis=1))
            reports.setdefault(name, rep)
            del gt, pt

        with ThreadPoolExecutor(max_workers=8) as pool:
            acc["C"].append(np.array(list(pool.map(lambda gp: arm_C_score(gp[0], gp[1])[0], zip(G, P)))))

        del G, P
        gc.collect()
        done = (ci + 1) * chunk
        print(f"  chunk {ci+1}/{n_chunks}  {min(done, args.pairs):,}/{args.pairs:,} pairs  "
              f"{time.time()-t0:.0f}s elapsed", flush=True)

    labels = np.concatenate(all_labels)
    arm_scores = {k: np.concatenate(v) for k, v in acc.items()}
    print(f"dataset {args.dataset}: {labels.size:,} pairs "
          f"({int(labels.sum())} genuine / {int((~labels).sum())} impostor) in {time.time()-t0:.0f}s")
    print(f"  gallery median {dims[0][0]}px, probe median {dims[0][1]}px")
    for name in arm_names:
        print(f"  arm {name}: TAR@FAR=0.1% {tar_at_far(arm_scores[name], labels)*100:6.2f}%  "
              f"AUC {auc(arm_scores[name], labels):.5f}")

    def judge(name: str) -> dict:
        pt, lo_, hi_ = paired_bootstrap(arm_scores["B1"], arm_scores[name], labels, seed=args.seed)
        if pt >= DECISION["pass_threshold"] and lo_ > 0:
            v = "PASS"
        elif pt <= DECISION["fail_threshold"] or lo_ <= 0 <= hi_:
            v = "FAIL"
        else:
            v = "INCONCLUSIVE"
        return {"points": pt, "ci95": [lo_, hi_], "verdict": v}

    # B2r is the DECISION metric. B2 applies the probe's ABSOLUTE operator, which
    # double-degrades a gallery that has already been through a similar imaging
    # chain -- the surveillance-to-surveillance case. B2r applies only the
    # residual and is the correct forward model. B2 is kept so the size of that
    # error stays visible rather than assumed.
    j2 = judge("B2")
    j2r = judge("B2r")
    point, lo, hi, verdict = j2r["points"], j2r["ci95"][0], j2r["ci95"][1], j2r["verdict"]

    print(f"\n{'=' * 74}")
    print(f"  B2  - B1 = {j2['points']:+6.2f} pts  CI [{j2['ci95'][0]:+6.2f}, {j2['ci95'][1]:+6.2f}]  "
          f"{j2['verdict']:<12} absolute operator")
    print(f"  B2r - B1 = {j2r['points']:+6.2f} pts  CI [{j2r['ci95'][0]:+6.2f}, {j2r['ci95'][1]:+6.2f}]  "
          f"{j2r['verdict']:<12} residual operator  <-- DECISION")
    print(f"  rule: PASS >= +{DECISION['pass_threshold']} with CI excluding 0; "
          f"FAIL <= +{DECISION['fail_threshold']} or CI spanning 0")
    print("=" * 74)

    payload = {
        "dataset": args.dataset,
        "scface_distance_m": {1: 4.20, 2: 2.60, 3: 1.00}.get(args.distance) if args.dataset == "scface" else None,
        "embedder": embed_name,
        "n_pairs": int(labels.size),
        "decision_rule": DECISION,
        "arms": {
            k: {
                "tar_at_far_1e3": tar_at_far(v, labels),
                "tar_at_far_1e2": tar_at_far(v, labels, 1e-2),
                "auc": auc(v, labels),
            }
            for k, v in arm_scores.items()
        },
        "B2_minus_B1": j2,
        "B2r_minus_B1": j2r,
        "decision_arm": "B2r",
        "arm_reports": reports,
    }
    suffix = f"_d{args.distance}" if args.dataset == "scface" else ""
    out = RESULTS / f"s0_3_{args.embedder}_{args.dataset}{suffix}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
