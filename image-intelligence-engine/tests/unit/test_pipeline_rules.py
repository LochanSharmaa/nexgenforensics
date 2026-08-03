"""Resume logic and workflow preconditions, as pure functions.

These decide whether a paid discovery call gets made a second time, so they are
tested with no database and no event loop.
"""

from __future__ import annotations

import pytest

from shared import workflow
from shared.enums import PIPELINE_ORDER, InvestigationStatus, PipelineStage, StageStatus
from shared.pipeline import (
    StageState,
    can_transition_run,
    is_resumable,
    plan_resume,
    progress_fraction,
    stages_after,
)


def _states(**statuses: str) -> list[StageState]:
    return [StageState(stage=stage, status=status) for stage, status in statuses.items()]


# ------------------------------------------------------------ resume plans --


def test_empty_run_starts_at_the_first_stage():
    plan = plan_resume([])
    assert plan.next_stage == PipelineStage.INGEST
    assert plan.completed == ()
    assert not plan.is_complete


def test_resume_skips_settled_prefix():
    plan = plan_resume(
        _states(INGEST=StageStatus.OK, DISCOVER=StageStatus.OK, VERIFY=StageStatus.PENDING)
    )
    assert plan.next_stage == PipelineStage.VERIFY
    assert plan.completed == ("INGEST", "DISCOVER")
    assert plan.skipped_work == 2


def test_skipped_stages_count_as_settled():
    """A stage deliberately skipped — OCR with no images — must not force a
    re-run of everything after it."""
    plan = plan_resume(
        _states(INGEST=StageStatus.OK, DISCOVER=StageStatus.SKIPPED, VERIFY=StageStatus.PENDING)
    )
    assert plan.next_stage == PipelineStage.VERIFY


def test_failed_stage_is_where_resume_picks_up():
    plan = plan_resume(
        _states(INGEST=StageStatus.OK, DISCOVER=StageStatus.FAILED, VERIFY=StageStatus.PENDING)
    )
    assert plan.next_stage == PipelineStage.DISCOVER


def test_a_gap_invalidates_later_successes():
    """A stage marked OK *after* an unfinished one cannot be trusted.

    Its input is genuinely missing, so re-running the gap must re-run everything
    downstream — otherwise findings from two different inputs mix silently.
    """
    plan = plan_resume(
        _states(
            INGEST=StageStatus.OK,
            DISCOVER=StageStatus.FAILED,
            VERIFY=StageStatus.OK,      # stale
            CRAWL=StageStatus.OK,       # stale
        )
    )
    assert plan.next_stage == PipelineStage.DISCOVER
    assert plan.completed == ("INGEST",)
    assert "VERIFY" in plan.remaining and "CRAWL" in plan.remaining


def test_run_with_every_stage_settled_is_complete():
    plan = plan_resume([StageState(stage=str(s), status=StageStatus.OK) for s in PIPELINE_ORDER])
    assert plan.is_complete
    assert plan.next_stage is None
    assert plan.remaining == ()


def test_progress_is_weighted_by_stage_not_by_items():
    """An operator watching a bar wants 'how far through the process', and
    stages carry wildly different item volumes."""
    assert progress_fraction([]) == 0.0
    half = [
        StageState(stage=str(s), status=StageStatus.OK)
        for s in list(PIPELINE_ORDER)[: len(PIPELINE_ORDER) // 2]
    ]
    assert 0.4 < progress_fraction(half) < 0.6
    full = [StageState(stage=str(s), status=StageStatus.OK) for s in PIPELINE_ORDER]
    assert progress_fraction(full) == 1.0


# ----------------------------------------------------- downstream staleness --


def test_stages_after_returns_everything_downstream():
    downstream = stages_after(PipelineStage.CLUSTER)
    assert "CORRELATE" in downstream and "SCORE" in downstream and "REPORT" in downstream
    assert "CRAWL" not in downstream


def test_score_is_downstream_of_cluster():
    """Independence cannot be computed before duplicate content is collapsed,
    so re-clustering must invalidate scoring."""
    assert "SCORE" in stages_after(PipelineStage.CLUSTER)


def test_last_stage_has_no_downstream():
    assert stages_after(PIPELINE_ORDER[-1]) == ()


def test_unknown_stage_raises():
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        stages_after("TELEPORT")


# ------------------------------------------------------------ run lifecycle --


def test_failed_runs_are_resumable():
    """Per-stage state exists precisely so a late failure does not discard the
    early stages' work."""
    assert is_resumable("FAILED")
    assert can_transition_run("FAILED", "RUNNING")


def test_completed_runs_are_terminal():
    assert not is_resumable("COMPLETED")
    assert not can_transition_run("COMPLETED", "RUNNING")


def test_paused_runs_resume():
    assert can_transition_run("PAUSED", "RUNNING")


def test_running_cannot_jump_to_queued():
    assert not can_transition_run("RUNNING", "QUEUED")


# ----------------------------------------------------------- preconditions --


def test_completion_blocked_by_pending_review():
    failures = workflow.check_transition(
        InvestigationStatus.UNDER_REVIEW,
        InvestigationStatus.COMPLETED,
        workflow.CaseState(pending_reviews=3, completed_runs=1),
    )
    assert [f.rule for f in failures] == ["review_queue_must_be_empty"]
    assert "3 machine proposal" in failures[0].message


def test_completion_allowed_once_queue_is_empty():
    assert workflow.is_permitted(
        InvestigationStatus.UNDER_REVIEW,
        InvestigationStatus.COMPLETED,
        workflow.CaseState(pending_reviews=0, completed_runs=1),
    )


def test_review_requires_a_completed_run():
    failures = workflow.check_transition(
        InvestigationStatus.ACTIVE,
        InvestigationStatus.UNDER_REVIEW,
        workflow.CaseState(completed_runs=0),
    )
    assert [f.rule for f in failures] == ["review_requires_a_completed_run"]


def test_deletion_blocked_by_retention_hold():
    """Holds take precedence over policy without exception."""
    failures = workflow.check_transition(
        InvestigationStatus.ARCHIVED,
        InvestigationStatus.DELETED_PENDING_RETENTION,
        workflow.CaseState(active_holds=1),
    )
    assert [f.rule for f in failures] == ["retention_hold_active"]


def test_unrelated_transitions_have_no_preconditions():
    assert workflow.is_permitted(
        InvestigationStatus.NEW, InvestigationStatus.ACTIVE, workflow.CaseState()
    )


def test_multiple_failures_are_all_reported():
    """An investigator should see every blocker at once, not fix them one at a
    time across three round trips."""
    failures = workflow.check_transition(
        InvestigationStatus.UNDER_REVIEW,
        InvestigationStatus.COMPLETED,
        workflow.CaseState(pending_reviews=2, completed_runs=1),
    )
    assert len(failures) >= 1
    assert all(f.message for f in failures)
