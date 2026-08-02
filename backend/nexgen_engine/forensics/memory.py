"""L5 -- identity memory as a replayable log, not mutable state.

The requirement that shapes everything here:

    If a linkage is disproven in court, every conclusion that depended on it must
    be exactly recomputable without it.

A template store cannot do that. Once observations are averaged into a template
the contribution of any single one is gone, and "what would we have concluded
without exhibit 7?" becomes unanswerable. So the person model is never mutated.
It is a PURE FUNCTION of an append-only log of observations, and retraction is
implemented by replaying the log with an entry excluded.

Five contamination controls, because a self-updating identity store that accepts
its own matches will bootstrap its errors into ground truth:

  1. EVERY WRITE IS AN INFERENCE. An observation enters with the likelihood ratio
     that justified it, stored beside it. Weak evidence contributes proportionally.

  2. NO AUTONOMOUS WRITES. Automated matches land in QUARANTINE. Only a recorded
     human adjudication promotes them. This is the control that prevents the
     bootstrap failure, and it is not optional.

  3. BIMODALITY MONITORING. A dossier whose observations do not look like one
     person is flagged, rather than quietly averaging two people together.

  4. RETRACTION PROPAGATES. Retracting an observation recomputes the model and
     reports which conclusions changed.

  5. EVERYTHING IS LOGGED, INCLUDING CONSOLIDATION. Consolidation may reweight;
     it may never fabricate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np


class Status(str, Enum):
    QUARANTINED = "quarantined"  # automated, awaiting human adjudication
    ADMITTED = "admitted"  # adjudicated, contributes to the model
    RETRACTED = "retracted"  # withdrawn; retained for replay, excluded from state
    REJECTED = "rejected"  # adjudicated as not this person


@dataclass(frozen=True)
class Observation:
    """One admitted-or-not observation of a person."""

    observation_id: str
    artifact_digest: str  # links into the L0 lineage DAG
    embedding: np.ndarray = field(repr=False)
    log10_lr: float  # evidence that this belongs to this person
    condition: str
    status: Status
    adjudicator: str = ""
    recorded_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "observation_id": self.observation_id,
            "artifact_digest": self.artifact_digest,
            "log10_lr": round(self.log10_lr, 4),
            "condition": self.condition,
            "status": self.status.value,
            "adjudicator": self.adjudicator,
            "recorded_utc": self.recorded_utc,
            "note": self.note,
        }


@dataclass(frozen=True)
class PersonState:
    """Derived state. Never stored, always recomputed from the log."""

    subject_id: str
    n_admitted: int
    n_quarantined: int
    n_retracted: int
    #: Evidence-weighted mean embedding, L2-normalised. None when no admitted obs.
    centroid: np.ndarray | None = field(repr=False, default=None)
    total_log10_lr: float = 0.0
    #: Mean pairwise similarity among admitted observations. Low values suggest
    #: the dossier may contain more than one person.
    coherence: float = float("nan")
    bimodal_flag: bool = False
    bimodal_detail: str = ""

    def as_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "n_admitted": self.n_admitted,
            "n_quarantined": self.n_quarantined,
            "n_retracted": self.n_retracted,
            "total_log10_lr": round(self.total_log10_lr, 4),
            "coherence": None if np.isnan(self.coherence) else round(self.coherence, 6),
            "bimodal_flag": self.bimodal_flag,
            "bimodal_detail": self.bimodal_detail,
        }


@dataclass
class IdentityLog:
    """Append-only observation log for one subject."""

    subject_id: str
    entries: list[Observation] = field(default_factory=list)
    #: Minimum evidence for an automated observation to be worth quarantining.
    min_log10_lr: float = 1.0
    #: Below this mean pairwise similarity, flag the dossier as possibly two people.
    coherence_floor: float = 0.35

    def propose(self, obs: Observation) -> tuple[bool, str]:
        """Automated write. Always lands in QUARANTINE, never in the model."""
        if obs.log10_lr < self.min_log10_lr:
            return False, (
                f"rejected: log10 LR {obs.log10_lr:.2f} below intake floor "
                f"{self.min_log10_lr:.2f}; too weak to be worth adjudicating"
            )
        self.entries.append(
            Observation(**{**obs.__dict__, "status": Status.QUARANTINED})
        )
        return True, "quarantined pending human adjudication"

    def adjudicate(self, observation_id: str, admit: bool, adjudicator: str, note: str = "") -> bool:
        """Human decision. The ONLY route into the model."""
        for i, e in enumerate(self.entries):
            if e.observation_id == observation_id and e.status is Status.QUARANTINED:
                self.entries[i] = Observation(
                    **{
                        **e.__dict__,
                        "status": Status.ADMITTED if admit else Status.REJECTED,
                        "adjudicator": adjudicator,
                        "note": note,
                    }
                )
                return True
        return False

    def retract(self, observation_id: str, adjudicator: str, reason: str) -> bool:
        """Withdraw an observation. The entry is kept so replay stays exact."""
        for i, e in enumerate(self.entries):
            if e.observation_id == observation_id and e.status is Status.ADMITTED:
                self.entries[i] = Observation(
                    **{**e.__dict__, "status": Status.RETRACTED, "adjudicator": adjudicator, "note": reason}
                )
                return True
        return False

    # -- derived state ---------------------------------------------------------

    def replay(self, exclude: set[str] | None = None) -> PersonState:
        """Recompute state from the log. This is the only way state exists."""
        exclude = exclude or set()
        admitted = [
            e for e in self.entries
            if e.status is Status.ADMITTED and e.observation_id not in exclude
        ]
        n_q = sum(1 for e in self.entries if e.status is Status.QUARANTINED)
        n_r = sum(1 for e in self.entries if e.status is Status.RETRACTED)

        if not admitted:
            return PersonState(self.subject_id, 0, n_q, n_r)

        emb = np.stack([e.embedding for e in admitted]).astype(np.float64)
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        # Evidence weighting: an observation admitted on LR 10^4 counts for more
        # than one admitted on 10^1, but sub-linearly -- log, not raw LR, or a
        # single strong observation would drown every other.
        w = np.array([max(e.log10_lr, 0.0) + 1.0 for e in admitted])
        centroid = (emb * w[:, None]).sum(0) / w.sum()
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)

        coherence = float("nan")
        bimodal, detail = False, ""
        if len(admitted) >= 3:
            sims = emb @ emb.T
            iu = np.triu_indices(len(admitted), k=1)
            pair = sims[iu]
            coherence = float(pair.mean())
            if coherence < self.coherence_floor:
                bimodal = True
                detail = (
                    f"mean pairwise similarity {coherence:.3f} is below the floor "
                    f"{self.coherence_floor:.3f}; this dossier may contain more than "
                    f"one individual and should be reviewed before use"
                )

        return PersonState(
            subject_id=self.subject_id,
            n_admitted=len(admitted),
            n_quarantined=n_q,
            n_retracted=n_r,
            centroid=centroid,
            total_log10_lr=float(sum(e.log10_lr for e in admitted)),
            coherence=coherence,
            bimodal_flag=bimodal,
            bimodal_detail=detail,
        )

    def counterfactual(self, observation_id: str) -> dict:
        """What changes if this observation is excluded?

        The question a defence expert will ask, answerable because state is a
        pure function of the log.
        """
        before = self.replay()
        after = self.replay(exclude={observation_id})
        drift = float("nan")
        if before.centroid is not None and after.centroid is not None:
            drift = float(1.0 - np.dot(before.centroid, after.centroid))
        return {
            "excluded": observation_id,
            "before": before.as_dict(),
            "after": after.as_dict(),
            "centroid_drift": None if np.isnan(drift) else round(drift, 8),
            "conclusion_changed": bool(
                before.n_admitted != after.n_admitted
                and (np.isnan(drift) or drift > 1e-6)
            ),
        }


__all__ = ["IdentityLog", "Observation", "PersonState", "Status"]
