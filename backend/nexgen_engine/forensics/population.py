"""The reference population -- the denominator of every likelihood ratio.

    LR = P(observation | this person) / P(observation | someone else)

The numerator is a model of the suspect. The denominator is a statement about
everyone else, and it is where forensic face comparison is weakest. A wrong
denominator does not produce an obviously wrong answer; it produces a confident
one, which is worse.

This module makes the denominator an explicit, declared, auditable object rather
than an implicit property of whatever impostor pairs happened to be lying around.

WHY THIS IS STRICT
------------------
On 2026-08-02 the first implementation of the capacity estimator in this package
was invalidated twice in one session:

  1. Against each protocol pack's own 3,000 impostor pairs the tail could not
     resolve below 1/3000, and LFW came back 99.7% CENSORED. No resolution.

  2. Rebuilt with a 20,000,000-sample pool drawn by randomly pairing images from
     different pairs -- assuming "different pair => different identity". That
     assumption is false. Contamination tracked identity count exactly:

         cfp_ff     500 ids   0.181%      cplfw    3,884 ids   0.039%
         cfp_fp     500 ids   0.153%      calfw    4,025 ids   0.020%
         agedb_30   568 ids   0.202%      lfw      5,749 ids   0.030%

     0.2% contamination is TWICE the FAR being measured at 0.1%. It made
     AgeDB-30 appear to fall from 96.03% to 8.40% TAR@FAR=0.1%. Artefact.

The lesson is not "be careful". It is that a reference population without
verified identity disjointness is not a weak measurement, it is a wrong one. So
this module REFUSES rather than degrades: construct a population without identity
labels and it raises. Every downstream number therefore either carries a valid
denominator or does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

#: Above this cosine, two ArcFace templates are almost certainly the same person.
#: Used only as a contamination *detector*, never as a decision threshold.
_CONTAMINATION_PROBE = 0.5


class PopulationPurityError(RuntimeError):
    """Raised when a reference population cannot be certified identity-disjoint."""


@dataclass(frozen=True)
class PurityAudit:
    """Evidence that a reference population is what it claims to be."""

    n_comparisons: int
    identity_labels_present: bool
    #: Fraction of "impostor" comparisons scoring above the contamination probe.
    suspect_fraction: float
    #: Smallest tail this population can resolve: 1 / n_comparisons.
    resolution: float
    #: Deepest FAR this population can support a claim about, with a safety
    #: factor of 10 so a claim never rests on a handful of samples.
    min_supportable_far: float
    verdict: Literal["valid", "label_uncertainty", "underpowered", "unlabelled"]
    detail: str

    @property
    def usable(self) -> bool:
        """Usable as a denominator.

        ``label_uncertainty`` is usable: identity disjointness was enforced by
        construction, and the flag records a caveat that travels with the result
        rather than a defect that invalidates it. ``unlabelled`` and
        ``underpowered`` are not usable under any circumstances.
        """
        return self.verdict in {"valid", "label_uncertainty"}

    def as_dict(self) -> dict:
        return {
            "n_comparisons": self.n_comparisons,
            "identity_labels_present": self.identity_labels_present,
            "suspect_fraction": round(self.suspect_fraction, 8),
            "resolution": self.resolution,
            "min_supportable_far": self.min_supportable_far,
            "verdict": self.verdict,
            "detail": self.detail,
        }


def audit_population(
    scores: np.ndarray,
    identity_labels_present: bool,
    target_far: float = 1e-3,
    genuine_reference: np.ndarray | None = None,
    contamination_probe: float | None = None,
) -> PurityAudit:
    """Decide whether an impostor score pool may be used as a denominator.

    Failure modes, in order of severity:

      unlabelled        -- disjointness assumed rather than verified. HARD REFUSE.
      underpowered      -- too few comparisons to resolve the target FAR. HARD REFUSE.
      label_uncertainty -- labels exist but the tail looks unusually heavy. FLAG.

    ON THE CONTAMINATION PROBE, and why it must be condition-relative
    -----------------------------------------------------------------
    The first version of this function used a fixed cosine of 0.5. That works on
    clean imagery, where impostors essentially never reach it, and it is WRONG on
    degraded imagery, where they routinely do. Applied to a TinyFace population
    built with true identity labels and rejection sampling -- disjoint *by
    construction* -- the fixed probe flagged 0.18% "contamination" and refused a
    valid measurement.

    That 0.18% was not contamination. It was genuine confusability: at 20 pixels,
    different people really do produce cosines above 0.5, which is precisely the
    phenomenon the whole capacity estimate exists to quantify. A detector that
    treats the signal as an artefact will refuse exactly the measurements that
    matter most.

    So when a genuine reference distribution is supplied the probe is derived from
    it -- the 95th percentile of genuine scores. Cross-identity pairs outscoring
    almost every same-identity pair are plausible label errors; pairs merely
    sitting in the overlap region are expected physics.

    And even then the verdict is a FLAG, not a refusal: on degraded imagery,
    duplicate identity labels and genuinely confusable people are not separable by
    score alone. Refusing on an unresolvable distinction would be false rigour.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = int(scores.size)
    resolution = 1.0 / n if n else float("inf")
    min_far = 10.0 / n if n else float("inf")

    if contamination_probe is None:
        if genuine_reference is not None and np.size(genuine_reference):
            probe = float(np.quantile(np.asarray(genuine_reference, dtype=np.float64), 0.95))
        else:
            probe = _CONTAMINATION_PROBE
    else:
        probe = float(contamination_probe)
    suspect = float((scores > probe).mean()) if n else 0.0

    if not identity_labels_present:
        return PurityAudit(
            n, False, suspect, resolution, min_far, "unlabelled",
            "Identity disjointness was assumed, not verified. A pool built by "
            "randomly pairing images without identity labels contains same-person "
            "pairs at a rate set by the corpus identity count.",
        )
    if min_far > target_far:
        return PurityAudit(
            n, True, suspect, resolution, min_far, "underpowered",
            f"{n:,} comparisons resolve to FAR {min_far:g} at best; {target_far:g} "
            f"was requested. Need at least {int(10 / target_far):,}.",
        )
    # Labels verified by construction. A heavy tail is reported, not refused.
    if suspect > 0.01:
        return PurityAudit(
            n, True, suspect, resolution, min_far, "label_uncertainty",
            f"{suspect:.4%} of cross-identity comparisons exceed {probe:.4f}, the "
            f"95th percentile of genuine scores. Usable, but a share of these may be "
            f"duplicate identity labels rather than confusable individuals. On "
            f"degraded imagery the two are not separable by score.",
        )
    return PurityAudit(
        n, True, suspect, resolution, min_far, "valid",
        f"{n:,} identity-disjoint comparisons (probe {probe:.4f}); supports claims "
        f"to FAR {min_far:g}.",
    )


@dataclass
class ReferencePopulation:
    """A declared, audited cohort against which evidence is weighed.

    ``name`` and ``description`` are not decoration. They are the parameter a
    court is entitled to contest: "why this population?" is a legitimate question
    and the system must be able to answer it from the record.
    """

    name: str
    description: str
    scores: np.ndarray = field(repr=False)
    audit: PurityAudit
    source: str = ""

    @classmethod
    def from_labelled(
        cls,
        name: str,
        description: str,
        embeddings: np.ndarray,
        identities: np.ndarray,
        n_samples: int = 20_000_000,
        seed: int = 0,
        target_far: float = 1e-3,
        source: str = "",
        strict: bool = True,
        genuine_reference: np.ndarray | None = None,
    ) -> "ReferencePopulation":
        """Build from embeddings WITH identity labels. The only supported route.

        Cross-identity pairs are sampled uniformly and rejection-filtered on the
        identity index, so disjointness is a property of construction rather than
        an assumption about the corpus.
        """
        emb = np.asarray(embeddings, dtype=np.float64)
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        ids = np.asarray(identities)
        if ids.shape[0] != emb.shape[0]:
            raise ValueError("embeddings and identities must align")

        rng = np.random.default_rng(seed)
        a = rng.integers(0, emb.shape[0], size=n_samples)
        b = rng.integers(0, emb.shape[0], size=n_samples)
        keep = ids[a] != ids[b]
        a, b = a[keep], b[keep]

        out = np.empty(a.size, dtype=np.float32)
        for s in range(0, a.size, 500_000):
            e = min(s + 500_000, a.size)
            out[s:e] = np.einsum("ij,ij->i", emb[a[s:e]], emb[b[s:e]]).astype(np.float32)

        audit = audit_population(
            out,
            identity_labels_present=True,
            target_far=target_far,
            genuine_reference=genuine_reference,
        )
        if strict and not audit.usable:
            raise PopulationPurityError(f"{name}: {audit.verdict} -- {audit.detail}")
        return cls(name=name, description=description, scores=out, audit=audit, source=source)

    @classmethod
    def unverified(cls, name: str, description: str, scores: np.ndarray, source: str = "") -> "ReferencePopulation":
        """Construct WITHOUT identity labels. Always audits as unusable.

        Provided so that an invalid population can be represented and reported
        rather than silently substituted for a valid one.
        """
        s = np.asarray(scores, dtype=np.float64)
        return cls(name, description, s, audit_population(s, identity_labels_present=False), source)

    # -- queries ---------------------------------------------------------------

    def typicality(self, score: float | np.ndarray) -> np.ndarray:
        """P(a random member of this population scores at least this high).

        The forensic "how common is this?" question. Floored at 1/(N+1) so an
        unobserved tail never becomes a zero probability and therefore never an
        infinite likelihood ratio.
        """
        self.require_usable()
        s = np.atleast_1d(np.asarray(score, dtype=np.float64))
        srt = np.sort(self.scores)
        n_ge = srt.size - np.searchsorted(srt, s, side="left")
        return np.maximum(n_ge / srt.size, 1.0 / (srt.size + 1.0))

    def random_match_probability(self, score: float, gallery_size: int) -> float:
        """Expected fraction of a gallery of this size scoring at least as high.

        The face analogue of the DNA random match probability, with the same
        caveat the DNA field states explicitly: it assumes population members are
        independent draws. Relatives and doppelgangers cluster, so the true value
        is HIGHER than this and the estimate is optimistic.
        """
        return float(np.clip(self.typicality(score)[0] * gallery_size, 0.0, 1.0))

    def require_usable(self) -> None:
        if not self.audit.usable:
            raise PopulationPurityError(
                f"reference population '{self.name}' is {self.audit.verdict}: {self.audit.detail}"
            )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "audit": self.audit.as_dict(),
        }


__all__ = [
    "PopulationPurityError",
    "PurityAudit",
    "ReferencePopulation",
    "audit_population",
]
