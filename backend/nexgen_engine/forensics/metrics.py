"""Forensic performance metrics for a likelihood-ratio system.

Accuracy and TAR@FAR describe a *decision* system: one that emits same/different
against a threshold. A forensic system does not decide. It reports the weight of
evidence and leaves the decision to the trier of fact, so it must be measured by
how well-calibrated that weight is, not by how often a threshold lands correctly.

The metric for that is Cllr -- the empirical cross-entropy of the reported
likelihood ratios against ground truth, in bits, relative to a neutral prior.

    Cllr = 1  ->  the system delivers nothing. Reporting LR=1 for every
                  comparison scores exactly 1.0, so any system at or above 1 is
                  worse than silence.
    Cllr = 0  ->  perfect and perfectly calibrated.

Cllr decomposes into discrimination and calibration loss:

    Cllr = Cllr_min + Cllr_cal

Cllr_min is what the system would score if its LRs were optimally recalibrated
without changing their ranking -- the discrimination ceiling, obtained by the
pool-adjacent-violators algorithm. Cllr_cal is the part that is pure calibration
error and is recoverable by fitting a better calibrator. Reporting the split
matters: a system with Cllr = 0.9 that is all Cllr_cal is nearly fixable, and one
that is all Cllr_min is not.

Reference: Brümmer & du Preez, "Application-independent evaluation of speaker
detection", Computer Speech & Language, 2006.

Everything here takes log10 LR, which is the forensic reporting convention, and
converts internally. Natural log is used for the arithmetic because
``np.logaddexp`` is stable there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_LN10 = float(np.log(10.0))
_LN2 = float(np.log(2.0))


def _as_nat(log10_lr: np.ndarray) -> np.ndarray:
    return np.asarray(log10_lr, dtype=np.float64) * _LN10


def cllr(log10_lr_genuine: np.ndarray, log10_lr_impostor: np.ndarray) -> float:
    """Log-likelihood-ratio cost, in bits.

    The two classes are weighted equally regardless of how many samples each
    contributes, which is what makes the metric independent of the proportion of
    same-source pairs in the evaluation set. That property is the reason it is
    usable in court: the number does not change because someone assembled a
    test set with a different genuine/impostor ratio.
    """
    g = _as_nat(log10_lr_genuine)
    i = _as_nat(log10_lr_impostor)
    if g.size == 0 or i.size == 0:
        return float("nan")
    # log2(1 + 1/LR) for genuine, log2(1 + LR) for impostor.
    loss_g = np.logaddexp(0.0, -g).mean() / _LN2
    loss_i = np.logaddexp(0.0, i).mean() / _LN2
    return float(0.5 * (loss_g + loss_i))


def pav(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the monotone posterior that best fits the labels.

    Returns, for each input score, the isotonic estimate of P(same-source | score)
    under the *empirical* prior of the input. Callers who want an LR must divide
    out that prior -- see :func:`pav_log10_lr`.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(scores, kind="mergesort")
    y = labels[order].astype(np.float64)

    # Blocks of (sum, count); merge backwards whenever monotonicity is violated.
    sums: list[float] = []
    counts: list[float] = []
    for value in y:
        sums.append(value)
        counts.append(1.0)
        while len(sums) > 1 and sums[-2] / counts[-2] > sums[-1] / counts[-1]:
            s = sums.pop()
            c = counts.pop()
            sums[-1] += s
            counts[-1] += c

    fitted = np.empty(len(y), dtype=np.float64)
    pos = 0
    for s, c in zip(sums, counts):
        fitted[pos : pos + int(c)] = s / c
        pos += int(c)

    out = np.empty_like(fitted)
    out[order] = fitted
    return out


def pav_log10_lr(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Optimally-calibrated log10 LR for these scores, by PAV.

    This is an *oracle* calibration -- it is fitted on the same labels it is
    scored against, so it must never be used to report a case result. Its only
    legitimate use is computing Cllr_min, the discrimination ceiling.
    """
    labels = np.asarray(labels, dtype=bool)
    posterior = pav(scores, labels)
    prior = float(labels.mean())
    if not 0.0 < prior < 1.0:
        return np.zeros(len(labels), dtype=np.float64)

    # logit(posterior) - logit(prior), with the degenerate blocks left as +/-inf.
    # PAV guarantees a block with posterior 0 contains no genuine pairs and a
    # block with posterior 1 contains no impostors, so the infinities never land
    # on the class that would make Cllr diverge.
    with np.errstate(divide="ignore"):
        post_odds = np.log(posterior) - np.log1p(-posterior)
    prior_odds = np.log(prior) - np.log1p(-prior)
    return (post_odds - prior_odds) / _LN10


@dataclass(frozen=True)
class CllrReport:
    """Cllr with its discrimination/calibration split and the counts behind it."""

    cllr: float
    cllr_min: float
    cllr_cal: float
    n_genuine: int
    n_impostor: int
    #: 1 - Cllr, i.e. bits of the one-bit same/different question actually
    #: delivered. Negative when the system is worse than reporting LR=1.
    bits_delivered: float

    def as_dict(self) -> dict:
        return {
            "cllr": round(self.cllr, 6),
            "cllr_min": round(self.cllr_min, 6),
            "cllr_cal": round(self.cllr_cal, 6),
            "bits_delivered": round(self.bits_delivered, 6),
            "n_genuine": self.n_genuine,
            "n_impostor": self.n_impostor,
        }


def cllr_report(log10_lr: np.ndarray, labels: np.ndarray, scores: np.ndarray | None = None) -> CllrReport:
    """Full Cllr decomposition.

    ``scores`` defaults to ``log10_lr``; pass the raw similarity separately if
    the calibrator is not monotone in it, because Cllr_min is defined by the
    ranking the *system* produces.
    """
    labels = np.asarray(labels, dtype=bool)
    log10_lr = np.asarray(log10_lr, dtype=np.float64)
    ranking = log10_lr if scores is None else np.asarray(scores, dtype=np.float64)

    actual = cllr(log10_lr[labels], log10_lr[~labels])
    oracle_lr = pav_log10_lr(ranking, labels)
    minimum = cllr(oracle_lr[labels], oracle_lr[~labels])
    return CllrReport(
        cllr=actual,
        cllr_min=minimum,
        cllr_cal=actual - minimum,
        n_genuine=int(labels.sum()),
        n_impostor=int((~labels).sum()),
        bits_delivered=1.0 - actual,
    )


@dataclass(frozen=True)
class TippettCurve:
    """Tippett plot data: the proportion of each class whose LR exceeds x.

    The ENFSI-standard picture of a forensic system. The impostor curve is the
    one that matters -- it answers "how often does this system report support for
    same-source when the sources are different, and how strongly?"
    """

    log10_lr_axis: np.ndarray = field(repr=False)
    genuine_exceeding: np.ndarray = field(repr=False)
    impostor_exceeding: np.ndarray = field(repr=False)
    #: Proportion of impostor comparisons reported with LR > 1, i.e. misleading
    #: evidence in favour of same-source. The headline safety number.
    rate_misleading_same_source: float
    #: Proportion of genuine comparisons reported with LR < 1.
    rate_misleading_different_source: float

    def as_dict(self) -> dict:
        return {
            "rate_misleading_same_source": round(self.rate_misleading_same_source, 6),
            "rate_misleading_different_source": round(self.rate_misleading_different_source, 6),
            "log10_lr_axis": [round(float(v), 4) for v in self.log10_lr_axis],
            "genuine_exceeding": [round(float(v), 6) for v in self.genuine_exceeding],
            "impostor_exceeding": [round(float(v), 6) for v in self.impostor_exceeding],
        }


def tippett(log10_lr: np.ndarray, labels: np.ndarray, n_points: int = 201) -> TippettCurve:
    log10_lr = np.asarray(log10_lr, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    finite = log10_lr[np.isfinite(log10_lr)]
    lo = float(finite.min()) if finite.size else -1.0
    hi = float(finite.max()) if finite.size else 1.0
    if hi <= lo:
        lo, hi = lo - 1.0, hi + 1.0
    axis = np.linspace(lo, hi, n_points)

    g = log10_lr[labels]
    i = log10_lr[~labels]
    return TippettCurve(
        log10_lr_axis=axis,
        genuine_exceeding=np.array([(g > x).mean() for x in axis]),
        impostor_exceeding=np.array([(i > x).mean() for x in axis]),
        rate_misleading_same_source=float((i > 0.0).mean()),
        rate_misleading_different_source=float((g < 0.0).mean()),
    )


__all__ = [
    "CllrReport",
    "TippettCurve",
    "cllr",
    "cllr_report",
    "pav",
    "pav_log10_lr",
    "tippett",
]
