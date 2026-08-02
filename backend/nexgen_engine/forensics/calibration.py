"""Turning a similarity score into a likelihood ratio.

A cosine similarity is not evidence. It is a number whose meaning depends on the
model that produced it, the population it was compared against, and the capture
conditions on both sides. Calibration is the step that converts it into a
quantity with a defined meaning:

    LR = P(score | same source) / P(score | different source)

The standard method is logistic regression on the score, fitted with the two
classes weighted equally so that the fitted log-odds IS the log likelihood ratio
rather than a posterior under whatever genuine/impostor ratio the training set
happened to have. That prior-independence is what makes the output reportable:
the number does not change because someone assembled a test set differently.

    log LR = a * score + b

Deliberately kept to two parameters. A flexible calibrator fits the training
scores better and generalises worse, and in a forensic setting an overconfident
LR is a worse failure than a weak one. Reference: Brümmer & du Preez 2006;
Ramos & Gonzalez-Rodriguez on forensic calibration.

CONDITIONAL CALIBRATION. One global calibrator encodes the assumption that a
score of 0.45 means the same thing for a portrait pair and a 32x32 CCTV pair. It
does not. :class:`ConditionalCalibrator` fits one calibrator per capture-condition
bin and selects at inference time on a condition estimate the engine already
computes. This is the concrete form of the argument that comparison must be
conditioned on capture condition, and it is the piece that makes cross-quality
comparison principled rather than hand-tuned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_LN10 = float(np.log(10.0))


@dataclass
class LogisticCalibrator:
    """Two-parameter score -> log10 LR map, fitted by Newton's method.

    ``a`` and ``b`` are in natural-log space; :meth:`log10_lr` converts to the
    forensic reporting convention.
    """

    a: float = 0.0
    b: float = 0.0
    n_genuine: int = 0
    n_impostor: int = 0
    converged: bool = False

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> "LogisticCalibrator":
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=bool)
        n_g = int(y.sum())
        n_i = int((~y).sum())
        if n_g == 0 or n_i == 0:
            raise ValueError("calibration needs both genuine and impostor comparisons")

        # Equal total weight per class -> the fitted logit is an LLR at a neutral
        # prior, independent of the class balance in this particular sample.
        w = np.where(y, 0.5 / n_g, 0.5 / n_i)
        target = y.astype(np.float64)
        x = np.column_stack([s, np.ones_like(s)])
        beta = np.array([1.0, 0.0], dtype=np.float64)

        converged = False
        for _ in range(max_iter):
            eta = x @ beta
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
            grad = x.T @ (w * (p - target))
            # Ridge term keeps the Hessian invertible when the classes separate
            # perfectly; without it a saturated fit sends |a| to infinity and the
            # reported LRs become arbitrary rather than large.
            hess = (x * (w * p * (1.0 - p))[:, None]).T @ x + 1e-9 * np.eye(2)
            step = np.linalg.solve(hess, grad)
            beta -= step
            if float(np.max(np.abs(step))) < tol:
                converged = True
                break

        self.a, self.b = float(beta[0]), float(beta[1])
        self.n_genuine, self.n_impostor = n_g, n_i
        self.converged = converged
        return self

    def log10_lr(self, scores: np.ndarray) -> np.ndarray:
        return (self.a * np.asarray(scores, dtype=np.float64) + self.b) / _LN10

    def as_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "n_genuine": self.n_genuine,
            "n_impostor": self.n_impostor,
            "converged": self.converged,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticCalibrator":
        return cls(
            a=float(d["a"]),
            b=float(d["b"]),
            n_genuine=int(d.get("n_genuine", 0)),
            n_impostor=int(d.get("n_impostor", 0)),
            converged=bool(d.get("converged", False)),
        )


@dataclass
class ConditionalCalibrator:
    """One calibrator per capture-condition bin, with a global fallback.

    ``bin_edges`` partitions a scalar condition estimate -- the engine's quality
    score is the intended input, because it is available at case time. Bins with
    too few comparisons to fit reliably fall back to the global calibrator rather
    than emitting a confident number from six samples.
    """

    bin_edges: list[float] = field(default_factory=list)
    bins: dict[int, LogisticCalibrator] = field(default_factory=dict)
    fallback: LogisticCalibrator | None = None
    min_per_bin: int = 200

    def _bin_of(self, condition: np.ndarray) -> np.ndarray:
        return np.digitize(np.asarray(condition, dtype=np.float64), self.bin_edges)

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        condition: np.ndarray,
    ) -> "ConditionalCalibrator":
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=bool)
        self.fallback = LogisticCalibrator().fit(scores, labels)

        idx = self._bin_of(condition)
        for b in np.unique(idx):
            sel = idx == b
            y = labels[sel]
            if sel.sum() < self.min_per_bin or y.sum() == 0 or (~y).sum() == 0:
                continue
            self.bins[int(b)] = LogisticCalibrator().fit(scores[sel], y)
        return self

    def log10_lr(self, scores: np.ndarray, condition: np.ndarray) -> np.ndarray:
        if self.fallback is None:
            raise RuntimeError("ConditionalCalibrator.fit must be called first")
        scores = np.asarray(scores, dtype=np.float64)
        out = self.fallback.log10_lr(scores)
        idx = self._bin_of(condition)
        for b, cal in self.bins.items():
            sel = idx == b
            if sel.any():
                out[sel] = cal.log10_lr(scores[sel])
        return out

    def as_dict(self) -> dict:
        return {
            "bin_edges": list(self.bin_edges),
            "min_per_bin": self.min_per_bin,
            "bins": {str(k): v.as_dict() for k, v in self.bins.items()},
            "fallback": self.fallback.as_dict() if self.fallback else None,
        }


def cross_validated_log10_lr(
    scores: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 10,
) -> np.ndarray:
    """Held-out LRs: fit on n-1 folds, predict the remaining one, for every fold.

    A calibrator scored against the data it was fitted on reports a Cllr that
    cannot be achieved on a new case. This is the only number worth quoting, and
    it uses the same contiguous-fold convention as the existing verification
    protocol so the two are directly comparable.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n = scores.size
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    out = np.empty(n, dtype=np.float64)

    for i in range(n_folds):
        test = np.zeros(n, dtype=bool)
        test[bounds[i] : bounds[i + 1]] = True
        train = ~test
        if labels[train].sum() == 0 or (~labels[train]).sum() == 0:
            out[test] = 0.0
            continue
        out[test] = LogisticCalibrator().fit(scores[train], labels[train]).log10_lr(scores[test])
    return out


__all__ = [
    "ConditionalCalibrator",
    "LogisticCalibrator",
    "cross_validated_log10_lr",
]
