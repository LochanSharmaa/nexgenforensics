"""Preconditions on investigation workflow transitions.

`enums.INVESTIGATION_TRANSITIONS` says which moves are *shaped* correctly.
This module says which are *permitted right now*, given the state of the case.
The two are separate because the transition table is a fixed graph while
preconditions depend on live data — how many review items are outstanding, how
many runs have completed.

Pure functions taking already-gathered counts, so the rules are testable without
a database and the caller does the querying.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import InvestigationStatus


@dataclass(frozen=True)
class PreconditionFailure:
    """Why a transition is refused, in terms an investigator can act on."""

    rule: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "message": self.message}


@dataclass(frozen=True)
class CaseState:
    """The live facts a precondition may consult."""

    pending_reviews: int = 0
    completed_runs: int = 0
    active_holds: int = 0


def check_transition(
    current: str, requested: str, state: CaseState
) -> tuple[PreconditionFailure, ...]:
    """Every precondition this move violates. Empty means permitted."""
    failures: list[PreconditionFailure] = []

    if requested == InvestigationStatus.UNDER_REVIEW and state.completed_runs < 1:
        failures.append(
            PreconditionFailure(
                rule="review_requires_a_completed_run",
                message=(
                    "There is nothing to review yet: no pipeline run has completed. "
                    "Run the pipeline before moving the case to review."
                ),
            )
        )

    if requested == InvestigationStatus.COMPLETED and state.pending_reviews > 0:
        failures.append(
            PreconditionFailure(
                rule="review_queue_must_be_empty",
                message=(
                    f"{state.pending_reviews} machine proposal(s) are still awaiting a "
                    "human decision. A case cannot be signed off with extraction output "
                    "nobody has looked at — the review queue is the line between what "
                    "the machine observed and what a person is willing to stand behind."
                ),
            )
        )

    if requested == InvestigationStatus.DELETED_PENDING_RETENTION and state.active_holds > 0:
        failures.append(
            PreconditionFailure(
                rule="retention_hold_active",
                message=(
                    f"{state.active_holds} unreleased retention hold(s) prevent marking "
                    "this case for deletion. Holds take precedence over retention policy "
                    "without exception."
                ),
            )
        )

    return tuple(failures)


def is_permitted(current: str, requested: str, state: CaseState) -> bool:
    return not check_transition(current, requested, state)


__all__ = [
    "CaseState",
    "PreconditionFailure",
    "check_transition",
    "is_permitted",
]
