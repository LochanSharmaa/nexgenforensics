"""
1:1 face VERIFICATION benchmarking on the standard InsightFace protocol packs.

WHAT THIS MEASURES (and what it deliberately does not)
------------------------------------------------------
This module measures **1:1 verification**: given two images, decide
same-person / different-person by thresholding cosine similarity. That is the
task the product's "compare two faces" feature performs.

It is NOT rank-1 closed-set identification ("which of these N enrolled people
is this probe?"). Those two tasks produce different numbers and are not
interchangeable. Any identification result belongs in a separate report.

PROTOCOL
--------
Pairs come from the published `.bin` verification packs shipped with the
ArcFace/InsightFace training sets (lfw.bin, agedb_30.bin, cfp_fp.bin,
calfw.bin, cplfw.bin). Each pack is a pickle of
`(encoded_images, issame_flags)` where images 2i and 2i+1 form pair i. These
are the standard published pair lists -- 6,000 pairs for LFW / AgeDB-30 /
CALFW / CPLFW and 7,000 for CFP-FP -- not ad hoc pairs generated here, so the
numbers are directly comparable to published results.

Images in these packs are already ArcFace-aligned 112x112 crops. They are fed
straight to the recognition network. Re-running face detection on an
already-tight crop is what the reference implementations avoid, and doing so
measurably degrades accuracy.

THRESHOLD HANDLING (the part that is easy to get wrong)
-------------------------------------------------------
Accuracy is computed with the standard 10-fold cross-validation protocol. For
each fold, the decision threshold is selected on the OTHER 9 FOLDS and then
applied to the held-out fold. The reported accuracy is the mean over the 10
held-out folds, with standard deviation.

This means no reported number is ever measured at a threshold that was tuned
on the same pairs. A single global "best threshold accuracy" over all pairs
is also computed, but only as `oracle_accuracy` -- it is optimistically
biased and must never be quoted as the system's accuracy.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Published pair counts, used to assert a pack has not been truncated.
EXPECTED_PAIRS = {
    "lfw": 6000,
    "agedb_30": 6000,
    "cfp_fp": 7000,
    "cfp_ff": 7000,
    "calfw": 6000,
    "cplfw": 6000,
}


def l2n(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12, None)


def load_pack(path: str | Path) -> tuple[list[bytes], np.ndarray]:
    """Load an InsightFace .bin verification pack -> (encoded_images, issame)."""
    with open(path, "rb") as fh:
        try:
            bins, issame = pickle.load(fh)
        except UnicodeDecodeError:
            fh.seek(0)
            bins, issame = pickle.load(fh, encoding="bytes")
    issame = np.asarray(issame, dtype=bool)
    if len(bins) != 2 * len(issame):
        raise ValueError(
            f"{path}: {len(bins)} images for {len(issame)} pairs; expected 2x"
        )
    return bins, issame


def decode_pack(bins: list[bytes]) -> np.ndarray:
    """Decode encoded images to an (N,112,112,3) BGR uint8 array."""
    out = np.empty((len(bins), 112, 112, 3), dtype=np.uint8)
    for i, raw in enumerate(bins):
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"image {i} failed to decode")
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        out[i] = img
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    accuracy: float
    threshold: float


@dataclass
class VerificationResult:
    dataset: str
    config: str
    n_pairs: int
    n_genuine: int
    n_impostor: int
    accuracy_mean: float
    accuracy_std: float
    threshold_mean: float
    threshold_std: float
    oracle_accuracy: float
    oracle_threshold: float
    tar_at_far_1e2: float
    tar_at_far_1e3: float
    tar_at_far_1e4: float
    auc: float
    eer: float
    folds: list[FoldResult] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "dataset": self.dataset,
            "config": self.config,
            "pairs": self.n_pairs,
            "accuracy": f"{self.accuracy_mean * 100:.2f} ± {self.accuracy_std * 100:.2f}",
            "threshold": f"{self.threshold_mean:.4f}",
            "TAR@FAR=1%": f"{self.tar_at_far_1e2 * 100:.2f}",
            "TAR@FAR=0.1%": f"{self.tar_at_far_1e3 * 100:.2f}",
            "AUC": f"{self.auc:.5f}",
            "EER": f"{self.eer * 100:.2f}",
        }


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Threshold maximizing accuracy. Candidates are midpoints between
    consecutive unique scores, which is where accuracy can actually change."""
    order = np.argsort(scores)
    s = scores[order]
    y = labels[order]
    # predicting "same" when score > t. Sweep t across all split points.
    n_pos = int(y.sum())
    n = len(y)
    # cumulative positives below split index i
    pos_below = np.concatenate(([0], np.cumsum(y)))
    idx = np.arange(n + 1)
    # correct = (negatives below) + (positives at/above)
    neg_below = idx - pos_below
    pos_above = n_pos - pos_below
    correct = neg_below + pos_above
    best_i = int(np.argmax(correct))
    acc = correct[best_i] / n
    if best_i == 0:
        thr = s[0] - 1e-6
    elif best_i == n:
        thr = s[-1] + 1e-6
    else:
        thr = (s[best_i - 1] + s[best_i]) / 2.0
    return float(thr), float(acc)


def _accuracy_at(scores: np.ndarray, labels: np.ndarray, thr: float) -> float:
    return float(((scores > thr) == labels).mean())


def _tar_at_far(scores: np.ndarray, labels: np.ndarray, far_target: float) -> float:
    """True accept rate at a target false accept rate."""
    impostor = np.sort(scores[~labels])[::-1]
    if impostor.size == 0:
        return float("nan")
    k = int(np.floor(far_target * impostor.size))
    thr = impostor[0] + 1e-9 if k == 0 else impostor[k - 1]
    genuine = scores[labels]
    return float((genuine > thr).mean())


def _auc_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-scores)
    y = labels[order]
    P = int(y.sum())
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan"), float("nan")
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    tpr = tp / P
    fpr = fp / N
    # np.trapezoid is numpy>=2.0; np.trapz is the <2.0 spelling. Support both.
    _trap = getattr(np, "trapezoid", None) or np.trapz
    auc = float(_trap(np.concatenate(([0], tpr)), np.concatenate(([0], fpr))))
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fnr[i] + fpr[i]) / 2)
    return auc, eer


def evaluate_pairs(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    issame: np.ndarray,
    dataset: str,
    config: str,
    n_folds: int = 10,
) -> VerificationResult:
    """Standard k-fold verification evaluation.

    emb_a / emb_b are (n_pairs, D) L2-normalized embeddings for the first and
    second image of each pair.
    """
    emb_a = l2n(emb_a.astype(np.float64))
    emb_b = l2n(emb_b.astype(np.float64))
    scores = np.sum(emb_a * emb_b, axis=1)
    labels = np.asarray(issame, dtype=bool)
    n = len(labels)

    # Contiguous folds: the published pair lists are ordered so that each
    # block of n/10 pairs is one official fold. Do NOT shuffle -- shuffling
    # breaks comparability with published numbers.
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    folds: list[FoldResult] = []
    for i in range(n_folds):
        test_idx = np.zeros(n, dtype=bool)
        test_idx[bounds[i] : bounds[i + 1]] = True
        train_idx = ~test_idx
        thr, _ = _best_threshold(scores[train_idx], labels[train_idx])
        folds.append(FoldResult(_accuracy_at(scores[test_idx], labels[test_idx], thr), thr))

    accs = np.array([f.accuracy for f in folds])
    thrs = np.array([f.threshold for f in folds])
    oracle_thr, oracle_acc = _best_threshold(scores, labels)
    auc, eer = _auc_eer(scores, labels)

    return VerificationResult(
        dataset=dataset,
        config=config,
        n_pairs=n,
        n_genuine=int(labels.sum()),
        n_impostor=int((~labels).sum()),
        accuracy_mean=float(accs.mean()),
        accuracy_std=float(accs.std()),
        threshold_mean=float(thrs.mean()),
        threshold_std=float(thrs.std()),
        oracle_accuracy=oracle_acc,
        oracle_threshold=oracle_thr,
        tar_at_far_1e2=_tar_at_far(scores, labels, 1e-2),
        tar_at_far_1e3=_tar_at_far(scores, labels, 1e-3),
        tar_at_far_1e4=_tar_at_far(scores, labels, 1e-4),
        auc=auc,
        eer=eer,
        folds=folds,
    )
