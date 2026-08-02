#!/usr/bin/env python
"""Evidence-layer measurement across every benchmark with a cached embedding set.

    python backend/scripts/analyze_evidence_capacity.py

Answers two questions the accuracy tables cannot:

  EXPERIMENT 3 -- CALIBRATION.  Do the reported likelihood ratios match observed
  error probabilities?  Measured by Cllr on HELD-OUT folds, split into the
  discrimination floor (Cllr_min) and the recoverable calibration loss
  (Cllr_cal).  A calibrator scored on its own training data is also computed and
  labelled `oracle`, so the optimism gap is visible rather than hidden.

  EXPERIMENT 4 -- FAILURE PREDICTION.  Does identity information in bits predict
  where the system fails?  If the bit estimate is a useful guard it must rank the
  datasets the same way the measured TAR@FAR=0.1% does.  If it does not, the
  guard is worthless and this script is how we find out.

Runs entirely from `runtime/benchmarks/embeddings/*.npz`.  No GPU, no model load.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.benchmarks.verification import l2n  # noqa: E402
from nexgen_engine.forensics import (  # noqa: E402
    capacity_from_pools,
    cllr_report,
    cross_validated_log10_lr,
    tippett,
)
from nexgen_engine.forensics.calibration import LogisticCalibrator  # noqa: E402

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
TINYFACE = _ROOT / "src_extracted/tinyface/tinyface/Testing_Set"
OUT = _ROOT / "runtime" / "forensics"

MODEL = "w600k_r50"
PAIRED = ["lfw", "cfp_ff", "agedb_30", "cfp_fp", "calfw", "cplfw"]
ID_RE = re.compile(r"^(\d+)_\d+\.jpg$", re.IGNORECASE)


def paired_scores(dataset: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Cosine scores + labels for a standard InsightFace protocol pack.

    Reproduces the exact convention of benchmark_verification.py: flip-TTA by
    summing original and mirrored embeddings, L2-normalising, then taking
    consecutive rows as the two sides of each pair.
    """
    path = CACHE / f"{dataset}__{MODEL}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    if "issame" not in d.files:
        return None
    emb = l2n((d["orig"] + d["flip"]).astype(np.float64))
    a, b = emb[0::2], emb[1::2]
    return np.sum(a * b, axis=1), np.asarray(d["issame"], dtype=bool)


def _chunked_dot(emb: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """Cosine for millions of index pairs without materialising millions of rows.

    Fancy-indexing 20M rows of a 512-d float64 matrix asks for 76 GiB. Chunking
    keeps peak memory at a few hundred MB and costs nothing in time.
    """
    out = np.empty(a.size, dtype=np.float32)
    for start in range(0, a.size, chunk):
        stop = min(start + chunk, a.size)
        out[start:stop] = np.einsum(
            "ij,ij->i", emb[a[start:stop]], emb[b[start:stop]]
        ).astype(np.float32)
    return out


def tinyface_pools(
    n_impostor: int = 20_000_000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """TinyFace genuine/impostor pools from real identity labels.

    The cache holds every labelled Testing_Set image (Gallery_Match + Probe) in
    sorted-path order, so identities are recoverable exactly from the filenames.
    That makes the impostor pool genuinely identity-disjoint rather than
    assumed-disjoint, which is the cleanest measurement available here.

    Returns ``(pair_scores, pair_labels, genuine_pool, impostor_pool)``: a
    balanced pair set for calibration, plus the large pools for capacity.
    """
    path = CACHE / f"tinyface_labelled__{MODEL}.npz"
    if not path.exists() or not TINYFACE.is_dir():
        return None

    files: list[Path] = []
    for sub in ("Gallery_Match", "Probe"):
        d = TINYFACE / sub
        if d.is_dir():
            files.extend(p for p in d.glob("*.jpg") if ID_RE.match(p.name))
    files = sorted(files)

    emb = l2n(np.load(path)["emb"].astype(np.float64))
    if emb.shape[0] != len(files):
        print(f"  tinyface: cache has {emb.shape[0]} rows, found {len(files)} images -- skipped")
        return None

    ids = np.array([int(ID_RE.match(p.name).group(1)) for p in files])
    rng = np.random.default_rng(seed)

    # Genuine: all within-identity pairs (capped, since some identities are large).
    by_id: dict[int, list[int]] = defaultdict(list)
    for idx, i in enumerate(ids):
        by_id[int(i)].append(idx)
    ga, gb = [], []
    for members in by_id.values():
        if len(members) < 2:
            continue
        m = np.array(members)
        for k in range(len(m)):
            for j in range(k + 1, len(m)):
                ga.append(m[k])
                gb.append(m[j])
    ga, gb = np.array(ga), np.array(gb)
    genuine_pool = np.sum(emb[ga] * emb[gb], axis=1)

    # Impostor: uniformly sampled cross-identity pairs, rejection-sampled.
    a = rng.integers(0, len(files), size=n_impostor)
    b = rng.integers(0, len(files), size=n_impostor)
    keep = ids[a] != ids[b]
    a, b = a[keep], b[keep]
    impostor_pool = _chunked_dot(emb, a, b)

    # Balanced pair set for the calibration experiment, interleaved so contiguous
    # folds hold both classes.
    n = min(3000, len(genuine_pool), len(impostor_pool))
    gsel = rng.choice(len(genuine_pool), n, replace=False)
    isel = rng.choice(len(impostor_pool), n, replace=False)
    scores = np.empty(2 * n)
    labels = np.zeros(2 * n, dtype=bool)
    scores[0::2] = genuine_pool[gsel]
    labels[0::2] = True
    scores[1::2] = impostor_pool[isel]
    return scores, labels, genuine_pool, impostor_pool


def impostor_pool_from_pairs(
    dataset: str, n_samples: int = 20_000_000, seed: int = 0
) -> np.ndarray | None:
    """DO NOT USE FOR CAPACITY. Retained only to document why it is invalid.

    The tempting move is to build a large impostor pool from a protocol pack by
    randomly pairing images from different pairs. It does not work, and the
    failure is instructive.

    Packs publish pairs, not identity labels. CFP and AgeDB draw thousands of
    pairs from 500-568 identities, so a random cross-pair sample is full of
    SAME-identity pairs. Measured contamination (fraction of sampled "impostors"
    scoring > 0.5) tracks identity count exactly as predicted:

        cfp_ff     500 ids   0.181%
        cfp_fp     500 ids   0.153%
        agedb_30   568 ids   0.202%
        cplfw    3,884 ids   0.039%
        calfw    4,025 ids   0.020%
        lfw      5,749 ids   0.030%

    0.2% contamination is TWICE the FAR being measured at 0.1%, so the estimated
    tail is dominated by mislabelled genuine pairs. Using this pool made AgeDB-30
    appear to drop from 96.03% to 8.40% TAR@FAR=0.1% -- an artefact, not a result.

    A capacity estimate is only as good as the identity-disjointness of its
    reference population. That is the whole lesson, and it is why capacity is
    computed ONLY where true identity labels exist.
    """
    path = CACHE / f"{dataset}__{MODEL}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    emb = l2n((d["orig"] + d["flip"]).astype(np.float64))
    n = emb.shape[0]
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, size=n_samples)
    b = rng.integers(0, n, size=n_samples)
    keep = (a // 2) != (b // 2)  # different pair => presumed different identity
    a, b = a[keep], b[keep]
    return _chunked_dot(emb, a, b)


def tar_at_far(scores: np.ndarray, labels: np.ndarray, far: float) -> float:
    imp = np.sort(scores[~labels])
    if imp.size == 0:
        return float("nan")
    thr = imp[max(0, int(np.ceil((1.0 - far) * imp.size)) - 1)]
    return float((scores[labels] > thr).mean())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, payload = [], {}

    # Calibration (experiment 3) runs on every pack: it uses the pack's own
    # curated pairs and needs no sampled population. Capacity (experiment 4)
    # runs ONLY where identity labels exist -- see impostor_pool_from_pairs.
    work: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]] = []
    for d in PAIRED:
        got = paired_scores(d)
        if got is None:
            print(f"  {d}: no cached embeddings -- skipped")
            continue
        work.append((d, got[0], got[1], None))
    tf = tinyface_pools()
    if tf is not None:
        work.append(("tinyface", tf[0], tf[1], tf[3]))
    else:
        print("  tinyface: unavailable -- skipped")

    for name, scores, labels, pool in work:
        # Experiment 3: held-out calibration, and the same calibrator scored on
        # its own training data for comparison.
        held_out = cross_validated_log10_lr(scores, labels, n_folds=10)
        oracle = LogisticCalibrator().fit(scores, labels).log10_lr(scores)
        cv = cllr_report(held_out, labels, scores=scores)
        orc = cllr_report(oracle, labels, scores=scores)
        tip = tippett(held_out, labels)

        # Experiment 4: only where the reference population is identity-disjoint.
        cap = capacity_from_pools(name, scores[labels], pool) if pool is not None else None

        rows.append(
            {
                "dataset": name,
                "cllr": cv.cllr,
                "cllr_min": cv.cllr_min,
                "cllr_cal": cv.cllr_cal,
                "cllr_oracle": orc.cllr,
                "bits_median": cap.bits_median if cap else float("nan"),
                "bits_p20": cap.bits_p20 if cap else float("nan"),
                "gallery_80": cap.gallery_at_rank1_80 if cap else float("nan"),
                "censored": cap.censored_fraction if cap else float("nan"),
                "tar_1e3": tar_at_far(scores, labels, 1e-3),
                "misleading_same": tip.rate_misleading_same_source,
            }
        )
        payload[name] = {
            "calibration_heldout": cv.as_dict(),
            "calibration_oracle": orc.as_dict(),
            "capacity": cap.as_dict() if cap else "UNAVAILABLE: no identity labels, see impostor_pool_from_pairs",
            "tippett": {
                "rate_misleading_same_source": tip.rate_misleading_same_source,
                "rate_misleading_different_source": tip.rate_misleading_different_source,
            },
            "tar_at_far_1e3": tar_at_far(scores, labels, 1e-3),
        }

    if not rows:
        print("no datasets available")
        return 1

    hdr = (
        f"{'dataset':<12}{'Cllr':>8}{'Cllr_min':>10}{'Cllr_cal':>10}"
        f"{'bits_med':>10}{'bits_p20':>10}{'gallery@80%':>16}{'TAR@FAR.1%':>12}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['dataset']:<12}{r['cllr']:>8.4f}{r['cllr_min']:>10.4f}{r['cllr_cal']:>10.4f}"
            + (
                f"{r['bits_median']:>10.2f}{r['bits_p20']:>10.2f}{r['gallery_80']:>16,.0f}"
                if np.isfinite(r["bits_median"])
                else f"{'--':>10}{'--':>10}{'no identity labels':>16}"
            )
            + f"{r['tar_1e3'] * 100:>11.2f}%"
        )

    # Experiment 4 verdict: does information content rank the datasets the same
    # way measured accuracy does? Spearman on ranks, computed directly.
    bits = np.array([r["bits_p20"] for r in rows])
    tar = np.array([r["tar_1e3"] for r in rows])
    ok = np.isfinite(bits)
    bits, tar = bits[ok], tar[ok]
    if ok.sum() > 2:
        rb = np.argsort(np.argsort(bits)).astype(float)
        rt = np.argsort(np.argsort(tar)).astype(float)
        rho = float(np.corrcoef(rb, rt)[0, 1])
        print(f"\nEXPERIMENT 4  Spearman(bits_p20, TAR@FAR=0.1%) = {rho:+.4f}  over {len(rows)} datasets")
        payload["_experiment_4"] = {"spearman_bits_vs_tar": rho, "n_datasets": int(ok.sum())}

    payload["_provenance"] = {
        "model": MODEL,
        "flip_tta": True,
        "calibration": "logistic, 10-fold, fitted on 9 applied to the held-out 1",
        "warning": (
            "The reference population is each benchmark's own impostor set, which is "
            "NOT a forensic reference population. These figures characterise the system, "
            "not any case. Gallery bounds assume independent gallery members and are "
            "therefore optimistic."
        ),
    }
    out = OUT / "capacity.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
