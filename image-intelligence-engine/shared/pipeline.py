"""Pipeline resume logic, as pure functions.

An investigation is a durable state machine. The rules that decide *where a
crashed run picks up* determine whether a paid discovery call gets made twice,
so they live in the domain layer where they are testable with no database and no
event loop.

The central rule: **resume at the first stage that is not OK.** Everything before
it already produced persisted output; re-running it would waste money on
providers and, worse, would produce a second set of observations for the same
source — which would then be counted twice by independence scoring.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .enums import PIPELINE_ORDER, RunStatus, StageStatus

# A run in one of these states has work left in it and can be continued.
RESUMABLE_RUN_STATES: frozenset[str] = frozenset(
    {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.PAUSED, RunStatus.FAILED}
)

# Stage outcomes that count as "done"; anything else must be re-run.
SETTLED_STAGE_STATES: frozenset[str] = frozenset({StageStatus.OK, StageStatus.SKIPPED})

RUN_TRANSITIONS: Mapping[str, frozenset[str]] = {
    RunStatus.QUEUED: frozenset({RunStatus.RUNNING, RunStatus.FAILED}),
    RunStatus.RUNNING: frozenset(
        {RunStatus.PAUSED, RunStatus.COMPLETED, RunStatus.FAILED}
    ),
    RunStatus.PAUSED: frozenset({RunStatus.RUNNING, RunStatus.FAILED}),
    # A failed run is resumable: the point of per-stage state is that a late
    # failure does not discard the early stages' work.
    RunStatus.FAILED: frozenset({RunStatus.RUNNING}),
    RunStatus.COMPLETED: frozenset(),
}


@dataclass(frozen=True)
class StageState:
    """The persisted state of one stage, as the domain layer sees it."""

    stage: str
    status: str
    items_total: int = 0
    items_done: int = 0
    items_failed: int = 0

    @property
    def is_settled(self) -> bool:
        return self.status in SETTLED_STAGE_STATES


@dataclass(frozen=True)
class ResumePlan:
    """What a resume would actually do."""

    next_stage: str | None
    """First stage needing execution, or None when the run is complete."""
    completed: tuple[str, ...]
    remaining: tuple[str, ...]
    is_complete: bool

    @property
    def skipped_work(self) -> int:
        """How many stages resume avoids re-running. Reported to the operator so
        the saving is visible rather than assumed."""
        return len(self.completed)


def plan_resume(states: Iterable[StageState]) -> ResumePlan:
    """Work out where a run should pick up.

    Stages are evaluated in ``PIPELINE_ORDER``, not in the order they happen to
    arrive from the database. A stage recorded as OK *after* an earlier
    non-settled stage does not let us skip the earlier one — the earlier output
    is genuinely missing, and later stages read from it.
    """
    by_stage = {state.stage: state for state in states}

    completed: list[str] = []
    remaining: list[str] = []
    next_stage: str | None = None

    for stage in PIPELINE_ORDER:
        state = by_stage.get(stage)
        settled = state is not None and state.is_settled

        if settled and next_stage is None:
            completed.append(str(stage))
            continue

        # Once the first gap is found, everything after it is remaining work
        # even if it happens to be marked OK from a previous attempt.
        if next_stage is None:
            next_stage = str(stage)
        remaining.append(str(stage))

    return ResumePlan(
        next_stage=next_stage,
        completed=tuple(completed),
        remaining=tuple(remaining),
        is_complete=next_stage is None,
    )


def stages_after(stage: str) -> tuple[str, ...]:
    """Stages that must be invalidated when ``stage`` is re-run.

    Downstream output derived from a stage's results is stale once that stage
    runs again. Leaving it marked OK would silently mix findings from two
    different inputs.
    """
    order = [str(s) for s in PIPELINE_ORDER]
    try:
        index = order.index(str(stage))
    except ValueError:
        raise ValueError(f"Unknown pipeline stage {stage!r}") from None
    return tuple(order[index + 1 :])


def can_transition_run(current: str, requested: str) -> bool:
    return requested in RUN_TRANSITIONS.get(current, frozenset())


def progress_fraction(states: Iterable[StageState]) -> float:
    """Completion across the whole pipeline, 0.0 to 1.0.

    Weighted by stage count rather than by item count: stages have wildly
    different item volumes, and an operator watching a bar wants "how far
    through the process", not "how many rows".
    """
    total = len(PIPELINE_ORDER)
    if not total:
        return 1.0
    settled = sum(1 for state in states if state.is_settled)
    return min(settled / total, 1.0)


def is_resumable(run_status: str) -> bool:
    return run_status in RESUMABLE_RUN_STATES


__all__ = [
    "RESUMABLE_RUN_STATES",
    "RUN_TRANSITIONS",
    "SETTLED_STAGE_STATES",
    "ResumePlan",
    "StageState",
    "can_transition_run",
    "is_resumable",
    "plan_resume",
    "progress_fraction",
    "stages_after",
]
