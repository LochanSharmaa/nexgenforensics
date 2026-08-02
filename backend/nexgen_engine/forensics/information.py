"""How much identity information does an observation actually carry?

Accuracy answers "how often was this system right on this test set". It says
nothing about whether the *evidence* could support the conclusion. Two systems
with identical accuracy on 6,000 pairs can be worlds apart when the gallery grows
from 6,000 to 10,000,000, and accuracy cannot tell you which.

The quantity that can is the identity information delivered by a comparison,
measured in bits against a reference population.

    For a genuine comparison scoring s, let

        p = P(impostor score >= s)

    read directly off the impostor distribution. In a gallery of N unrelated
    people, the expected number of non-mates outscoring the true mate is N * p.
    Reliable rank-1 identification therefore requires N * p << 1, so

        N_max ~ 1 / p = 2 ** I,   where   I = -log2(p)

This is the random-match-probability logic that makes forensic DNA defensible,
transplanted to face comparison. Note what it does NOT require: no mutual
information estimator, no density model in 512 dimensions, no neural estimate of
anything. It needs the empirical impostor tail, which is directly measurable.

FOUR LIMITATIONS, none of which are optional reading:

1. CENSORING. With M impostor comparisons the smallest resolvable tail is 1/M,
   so bits are capped at log2(M). Every report states the censored fraction. A
   number produced from a censored estimate is a floor, never a value.

2. THE REFERENCE POPULATION IS NOT FORENSIC. Impostor pairs from an academic
   benchmark are whoever the pack authors assembled. A real reference population
   is demographically matched to the case. Bits measured here characterise the
   *system*, not any particular case.

3. INDEPENDENCE IS ASSUMED, AND IT IS FALSE. The N*p argument treats gallery
   members as independent draws. Real populations contain relatives and
   doppelgangers, which cluster. Ignoring that structure makes every number here
   OPTIMISTIC -- the true defensible gallery is smaller than reported. That is
   the safe direction for a guard and the dangerous direction for a claim.

4. IT IS A PROPERTY OF (model, population, condition), NOT OF THE IMAGE. A
   different recognition model extracts a different number of bits from the same
   pixels. This measures what *this system* can recover, which is a lower bound
   on what the image contains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CapacityReport:
    """Identity information delivered on a set of genuine comparisons."""

    dataset: str
    n_genuine: int
    n_impostor: int
    #: Ceiling imposed by the impostor sample size: log2(M + 1).
    censoring_ceiling_bits: float
    #: Fraction of genuine comparisons whose tail estimate hit the ceiling.
    censored_fraction: float
    bits_p10: float
    bits_p20: float
    bits_median: float
    bits_mean: float
    #: Largest gallery at which the stated fraction of genuine probes would still
    #: be expected to outrank every non-mate. See :func:`gallery_for_rank1`.
    gallery_at_rank1_50: float
    gallery_at_rank1_80: float
    gallery_at_rank1_90: float

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "n_genuine": self.n_genuine,
            "n_impostor": self.n_impostor,
            "censoring_ceiling_bits": round(self.censoring_ceiling_bits, 3),
            "censored_fraction": round(self.censored_fraction, 5),
            "bits_p10": round(self.bits_p10, 3),
            "bits_p20": round(self.bits_p20, 3),
            "bits_median": round(self.bits_median, 3),
            "bits_mean": round(self.bits_mean, 3),
            "gallery_at_rank1_50": _round_sig(self.gallery_at_rank1_50),
            "gallery_at_rank1_80": _round_sig(self.gallery_at_rank1_80),
            "gallery_at_rank1_90": _round_sig(self.gallery_at_rank1_90),
        }


def _round_sig(x: float, sig: int = 3) -> float:
    if not np.isfinite(x) or x <= 0:
        return float(x)
    return float(round(x, -int(np.floor(np.log10(x))) + (sig - 1)))


def identity_bits(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bits of identity information per genuine comparison.

    Returns ``(bits, censored)`` where ``censored`` marks the comparisons whose
    impostor tail was empty, i.e. the estimate hit the sample-size ceiling and is
    a floor rather than a value.
    """
    g = np.asarray(genuine_scores, dtype=np.float64)
    imp = np.sort(np.asarray(impostor_scores, dtype=np.float64))
    m = imp.size
    if m == 0 or g.size == 0:
        return np.zeros(g.size), np.zeros(g.size, dtype=bool)

    # Count of impostors >= s, via the sorted array.
    n_ge = m - np.searchsorted(imp, g, side="left")
    censored = n_ge == 0
    # Laplace-style floor: an unobserved tail is treated as 1/(m+1), never 0.
    tail = np.where(censored, 1.0 / (m + 1.0), n_ge / m)
    return -np.log2(tail), censored


def gallery_for_rank1(
    bits: np.ndarray,
    target_rank1: float,
    tolerance: float = 0.5,
) -> float:
    """Largest gallery supporting ``target_rank1`` fraction of correct rank-1 hits.

    A probe delivering ``I`` bits tolerates a gallery of ``tolerance * 2**I``
    before the expected count of non-mates outscoring the true mate exceeds
    ``tolerance``. To serve a fraction ``r`` of probes, the binding constraint is
    the ``(1 - r)``-quantile of the bit distribution -- the weak comparisons, not
    the average one.
    """
    bits = np.asarray(bits, dtype=np.float64)
    if bits.size == 0:
        return float("nan")
    quantile = float(np.quantile(bits, 1.0 - target_rank1))
    return float(tolerance * (2.0**quantile))


def capacity_report(
    dataset: str,
    scores: np.ndarray,
    labels: np.ndarray,
) -> CapacityReport:
    """Convenience wrapper when genuine and impostor come from one labelled array.

    Beware: a benchmark pair list supplies only a few thousand impostors, which
    caps the estimate at log2(M) bits. Modern recognition delivers more than that
    on clean imagery, so this path saturates. Prefer
    :func:`capacity_from_pools` with a large independently-sampled impostor pool.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    return capacity_from_pools(dataset, scores[labels], scores[~labels])


def capacity_from_pools(
    dataset: str,
    genuine_scores: np.ndarray,
    impostor_pool: np.ndarray,
) -> CapacityReport:
    """Capacity from a genuine set and a separately-sampled impostor pool.

    The impostor pool sets the resolution of the whole measurement: it can only
    resolve tails down to 1/M, so M must exceed the gallery sizes being reasoned
    about. Millions of impostor comparisons cost nothing when embeddings are
    already cached, and they are the difference between a saturated estimate and
    a usable one.
    """
    g = np.asarray(genuine_scores, dtype=np.float64)
    imp = np.asarray(impostor_pool, dtype=np.float64)
    bits, censored = identity_bits(g, imp)

    return CapacityReport(
        dataset=dataset,
        n_genuine=int(g.size),
        n_impostor=int(imp.size),
        censoring_ceiling_bits=float(np.log2(imp.size + 1.0)) if imp.size else 0.0,
        censored_fraction=float(censored.mean()),
        bits_p10=float(np.quantile(bits, 0.10)),
        bits_p20=float(np.quantile(bits, 0.20)),
        bits_median=float(np.median(bits)),
        bits_mean=float(bits.mean()),
        gallery_at_rank1_50=gallery_for_rank1(bits, 0.50),
        gallery_at_rank1_80=gallery_for_rank1(bits, 0.80),
        gallery_at_rank1_90=gallery_for_rank1(bits, 0.90),
    )


__all__ = [
    "CapacityReport",
    "capacity_from_pools",
    "capacity_report",
    "gallery_for_rank1",
    "identity_bits",
]
