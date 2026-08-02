"""Open-set identification: the subject may not be enrolled at all.

Nearest-neighbour search answers "who is closest?", a question that ALWAYS has an
answer. Run an unenrolled probe against a gallery and a ranked list still comes
back, and the rank-1 entry is structurally indistinguishable from a true hit.
This is not an implementation flaw in any particular system -- it is definitional
for retrieval by proximity, and it is the property most likely to put the wrong
person in an interview room.

This module replaces the ranked list with three things:

    P(H_unknown)      an explicit hypothesis that nobody in the gallery is the
                      source, with a likelihood computed from the reference
                      population rather than assumed

    conformal set     a candidate set with a coverage guarantee -- "the subject
                      is in this set of 47, or is not enrolled, with 95%
                      confidence" -- instead of a fixed top-10

    abstention        "the evidence is insufficient" as a first-class outcome

The conformal construction is split-conformal on a calibration set of genuine
comparisons. Its guarantee is marginal coverage under exchangeability: over many
cases, the true source is inside the set at least (1 - alpha) of the time when
enrolled. It is NOT a per-case probability, and reporting it as one would be a
misstatement. Coverage must be validated empirically -- see
scripts/evaluate_baseline.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .population import ReferencePopulation


@dataclass(frozen=True)
class Candidate:
    identifier: str
    score: float
    log10_lr: float
    #: P(a random member of the reference population scores this high).
    typicality: float


@dataclass(frozen=True)
class OpenSetResult:
    """The output of a 1:N query. Deliberately not a ranked list."""

    candidates: list[Candidate]
    #: Posterior mass on "the source is not enrolled", under the stated prior.
    p_unknown: float
    prior_unknown: float
    #: Conformal set at the requested error level; may be empty or large.
    conformal_set: list[str]
    alpha: float
    coverage_guarantee: str
    abstained: bool
    abstain_reason: str = ""
    gallery_size: int = 0
    population: str = ""

    def as_dict(self) -> dict:
        return {
            "candidates": [
                {
                    "id": c.identifier,
                    "score": round(c.score, 6),
                    "log10_lr": round(c.log10_lr, 4),
                    "typicality": c.typicality,
                }
                for c in self.candidates
            ],
            "p_unknown": round(self.p_unknown, 6),
            "prior_unknown": self.prior_unknown,
            "conformal_set": self.conformal_set,
            "alpha": self.alpha,
            "coverage_guarantee": self.coverage_guarantee,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "gallery_size": self.gallery_size,
            "reference_population": self.population,
        }


@dataclass
class ConformalCalibrator:
    """Split-conformal threshold from held-out GENUINE comparison scores.

    The nonconformity measure is the negated similarity, so the threshold is the
    alpha-quantile of genuine scores: any gallery entry scoring at or above it is
    admitted to the set. Fitted on genuine comparisons the system has never used
    for anything else.
    """

    alpha: float = 0.05
    threshold: float = float("nan")
    n_calibration: int = 0

    def fit(self, genuine_scores: np.ndarray) -> "ConformalCalibrator":
        g = np.sort(np.asarray(genuine_scores, dtype=np.float64))
        n = g.size
        if n < 20:
            raise ValueError(f"conformal calibration needs >=20 genuine scores, got {n}")
        # Finite-sample corrected index: floor(alpha * (n + 1)) - 1, clipped.
        k = int(np.floor(self.alpha * (n + 1))) - 1
        self.threshold = float(g[int(np.clip(k, 0, n - 1))])
        self.n_calibration = n
        return self

    def admit(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=np.float64) >= self.threshold

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "n_calibration": self.n_calibration,
        }


def posterior_over_hypotheses(
    log10_lrs: np.ndarray,
    prior_unknown: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Posterior over {enrolled candidates} + H_unknown.

    Each enrolled hypothesis carries prior (1 - prior_unknown)/N; H_unknown has
    likelihood 1 by construction, because the LR is already defined *relative to*
    the population that H_unknown describes.

    ``prior_unknown`` is a case parameter, not a system constant. In a watchlist
    screen almost every probe is unenrolled and it should be near 1; in a
    confirmatory comparison against a named suspect it is much lower. It is
    reported alongside the result so the assumption is contestable.
    """
    lr = np.power(10.0, np.clip(np.asarray(log10_lrs, dtype=np.float64), -300, 300))
    n = lr.size
    if n == 0:
        return np.array([]), 1.0
    prior_each = (1.0 - prior_unknown) / n
    unnorm = np.concatenate([lr * prior_each, [prior_unknown]])
    total = unnorm.sum()
    if total <= 0 or not np.isfinite(total):
        return np.zeros(n), 1.0
    post = unnorm / total
    return post[:n], float(post[n])


def identify(
    scores: np.ndarray,
    identifiers: list[str],
    calibrator,
    population: ReferencePopulation,
    conformal: ConformalCalibrator,
    prior_unknown: float = 0.5,
    top_k: int = 10,
    min_bits: float = 1.0,
) -> OpenSetResult:
    """Run a 1:N query and return evidence, not a ranking.

    Abstains when the strongest candidate carries less than ``min_bits`` of
    identity information against the reference population. At that point the
    observation cannot distinguish the top candidate from a large fraction of the
    population, and returning a name would misrepresent the evidence.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size != len(identifiers):
        raise ValueError("scores and identifiers must align")
    population.require_usable()

    typ = population.typicality(scores)
    bits = -np.log2(typ)
    log10_lr = np.asarray(calibrator.log10_lr(scores), dtype=np.float64)

    order = np.argsort(-scores)
    admitted = [identifiers[i] for i in np.flatnonzero(conformal.admit(scores))]
    post, p_unknown = posterior_over_hypotheses(log10_lr, prior_unknown)

    best = int(order[0]) if order.size else -1
    abstain = best < 0 or bits[best] < min_bits
    reason = ""
    if abstain and best >= 0:
        reason = (
            f"Strongest candidate carries {bits[best]:.2f} bits against "
            f"'{population.name}'; {min_bits:.2f} required. The observation cannot "
            f"distinguish it from roughly 1 in {2 ** bits[best]:,.0f} of the population."
        )

    cands = [
        Candidate(identifiers[i], float(scores[i]), float(log10_lr[i]), float(typ[i]))
        for i in order[:top_k]
    ]
    return OpenSetResult(
        candidates=[] if abstain else cands,
        p_unknown=float(p_unknown),
        prior_unknown=prior_unknown,
        conformal_set=[] if abstain else admitted,
        alpha=conformal.alpha,
        coverage_guarantee=(
            f"Marginal coverage >= {1 - conformal.alpha:.0%} over repeated cases when the "
            f"source is enrolled, under exchangeability. Not a per-case probability."
        ),
        abstained=abstain,
        abstain_reason=reason,
        gallery_size=int(scores.size),
        population=population.name,
    )


__all__ = [
    "Candidate",
    "ConformalCalibrator",
    "OpenSetResult",
    "identify",
    "posterior_over_hypotheses",
]
