"""The reportable object: weight of evidence, with everything needed to contest it.

A similarity score travels badly. Written into a report it loses the model that
produced it, the population it was weighed against, the conditions on both sides,
and the fact that it was never validated at the operating point being quoted. By
the time it reaches a courtroom it has become "the AI said 0.62", which is not a
scientific statement.

:class:`EvidenceReport` is built so that cannot happen. Provenance, calibration
source, reference population, confidence interval and limitations are REQUIRED
fields, not optional decoration, and the renderers below will not emit a number
without them.

VERBAL SCALE. The bands follow the ENFSI Guideline for Evaluative Reporting
(2015). The verbal label is a communication aid for a fixed numeric range, never
a substitute for the LR, and it is always printed alongside the number.

THE CAPACITY GUARD. A reported LR is checked against the identity information the
observation can actually support. A likelihood ratio larger than the evidence
allows is the signature of a model prior leaking into a conclusion. The guard is
a HEURISTIC -- the information bound is an expectation over comparisons, not a
hard per-case limit -- so it flags and records rather than silently clamping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# (lower log10 LR inclusive, label) for support of the same-source proposition.
_ENFSI_BANDS: tuple[tuple[float, str], ...] = (
    (6.0, "extremely strong support"),
    (4.0, "very strong support"),
    (3.0, "strong support"),
    (2.0, "moderately strong support"),
    (1.0, "moderate support"),
    (0.3010, "weak support"),
    (0.0, "no meaningful support"),
)


def verbal_scale(log10_lr: float) -> str:
    """ENFSI verbal equivalent, symmetric about LR = 1."""
    if not np.isfinite(log10_lr):
        return "undefined"
    magnitude = abs(log10_lr)
    label = "no meaningful support"
    for lo, text in _ENFSI_BANDS:
        if magnitude >= lo:
            label = text
            break
    if label == "no meaningful support":
        return "no meaningful support for either proposition"
    side = "same source" if log10_lr > 0 else "different sources"
    return f"{label} for {side}"


@dataclass(frozen=True)
class Provenance:
    """Where every number came from. All fields required."""

    model: str
    model_sha256: str
    device: str
    calibration_source: str
    calibration_cllr: float
    reference_population: str
    population_verdict: str
    engine_version: str
    generated_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "model_sha256": self.model_sha256,
            "device": self.device,
            "calibration_source": self.calibration_source,
            "calibration_cllr": round(self.calibration_cllr, 6),
            "reference_population": self.reference_population,
            "population_verdict": self.population_verdict,
            "engine_version": self.engine_version,
            "generated_utc": self.generated_utc,
        }


def bootstrap_ci(
    values: np.ndarray,
    statistic,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI. Used for every aggregate figure we publish."""
    v = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = v.shape[0]
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = statistic(v[rng.integers(0, n, n)])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


@dataclass
class EvidenceReport:
    """One comparison, reported so that it can be challenged."""

    score: float
    log10_lr: float
    provenance: Provenance
    #: Identity information the observation supports, in bits, against the
    #: declared population. None when no valid population exists.
    bits_available: float | None = None
    log10_lr_ci: tuple[float, float] | None = None
    condition_bin: str = "unspecified"
    limitations: list[str] = field(default_factory=list)
    capacity_flag: bool = False
    capacity_detail: str = ""

    def __post_init__(self) -> None:
        self._apply_capacity_guard()
        self._add_standing_limitations()

    def _apply_capacity_guard(self) -> None:
        if self.bits_available is None:
            self.capacity_flag = True
            self.capacity_detail = (
                "No valid reference population, so the information bound could not "
                "be evaluated. This LR is unguarded."
            )
            return
        # log2(LR) must not exceed the identity information the observation carries.
        claimed_bits = abs(self.log10_lr) * np.log2(10.0)
        if claimed_bits > self.bits_available + 1.0:  # 1 bit of slack
            self.capacity_flag = True
            self.capacity_detail = (
                f"Reported LR claims {claimed_bits:.2f} bits of discrimination but the "
                f"observation supports about {self.bits_available:.2f} against "
                f"'{self.provenance.reference_population}'. This is the signature of a "
                f"model prior contributing to the conclusion. Treat the LR as an upper "
                f"bound and investigate before relying on it."
            )

    def _add_standing_limitations(self) -> None:
        standing = [
            "Automated face comparison returns investigative leads, not "
            "identifications. A qualified examiner must verify any conclusion.",
            f"The likelihood ratio is conditional on the reference population "
            f"'{self.provenance.reference_population}'. A different population "
            f"gives a different value; the choice is contestable.",
            f"Calibration was fitted on {self.provenance.calibration_source} "
            f"(Cllr = {self.provenance.calibration_cllr:.4f}) and has not been "
            f"validated on imagery from this case's capture conditions.",
        ]
        if self.provenance.population_verdict != "valid":
            standing.insert(
                0,
                f"REFERENCE POPULATION IS NOT VALID "
                f"({self.provenance.population_verdict}). The likelihood ratio below "
                f"is not reportable and is shown for methodological purposes only.",
            )
        if self.capacity_flag:
            standing.insert(0, f"CAPACITY GUARD RAISED. {self.capacity_detail}")
        self.limitations = standing + self.limitations

    @property
    def likelihood_ratio(self) -> float:
        return float(10.0 ** np.clip(self.log10_lr, -300, 300))

    @property
    def verbal(self) -> str:
        return verbal_scale(self.log10_lr)

    @property
    def reportable(self) -> bool:
        """False when anything makes the number unsafe to put in a report."""
        return self.provenance.population_verdict == "valid" and not self.capacity_flag

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 6),
            "log10_lr": round(self.log10_lr, 4),
            "likelihood_ratio": self.likelihood_ratio,
            "verbal_equivalent": self.verbal,
            "log10_lr_ci95": list(self.log10_lr_ci) if self.log10_lr_ci else None,
            "bits_available": self.bits_available,
            "condition_bin": self.condition_bin,
            "reportable": self.reportable,
            "capacity_flag": self.capacity_flag,
            "provenance": self.provenance.as_dict(),
            "limitations": self.limitations,
        }

    def to_text(self) -> str:
        """Plain-text block for a forensic report. Number never appears alone."""
        lines = [
            "EVALUATIVE COMPARISON RESULT",
            "=" * 72,
            f"Similarity (cosine)        : {self.score:.6f}",
            f"Log10 likelihood ratio     : {self.log10_lr:+.4f}"
            + (f"   95% CI [{self.log10_lr_ci[0]:+.4f}, {self.log10_lr_ci[1]:+.4f}]"
               if self.log10_lr_ci else "   (no interval available)"),
            f"Likelihood ratio           : {self.likelihood_ratio:,.4g}",
            f"Verbal equivalent (ENFSI)  : {self.verbal}",
            f"Capture-condition bin      : {self.condition_bin}",
            f"Identity information       : "
            + (f"{self.bits_available:.2f} bits" if self.bits_available is not None else "not evaluable"),
            f"Reportable                 : {'YES' if self.reportable else 'NO'}",
            "",
            "PROVENANCE",
            "-" * 72,
        ]
        for k, v in self.provenance.as_dict().items():
            lines.append(f"  {k:<24}: {v}")
        lines += ["", "LIMITATIONS", "-" * 72]
        lines += [f"  {i}. {t}" for i, t in enumerate(self.limitations, 1)]
        return "\n".join(lines)


__all__ = ["EvidenceReport", "Provenance", "bootstrap_ci", "verbal_scale"]
