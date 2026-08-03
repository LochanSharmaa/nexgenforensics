"""Domain rules: state machines, hashing, config, ids, clock.

These are the rules most likely to be challenged in a report, so they are tested
with zero infrastructure — no database, no network, no event loop.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from shared import enums
from shared.clock import FrozenClock, require_utc
from shared.config import Settings
from shared.enums import (
    IllegalTransition,
    InvestigationStatus,
    ProgressState,
    RetentionState,
)
from shared.hashing import GENESIS, chain_hash, content_hash, verify_chain
from shared.ids import SequentialIdGenerator, timestamp_ms_of, uuid7

# -- investigation workflow -------------------------------------------------


def test_new_case_can_become_active():
    assert enums.can_transition("investigation_status", InvestigationStatus.NEW,
                                InvestigationStatus.ACTIVE)


def test_case_cannot_skip_review_to_completed():
    """Completion must pass through review, so machine output is never signed
    off unread."""
    with pytest.raises(IllegalTransition) as exc:
        enums.assert_transition(
            "investigation_status", InvestigationStatus.ACTIVE, InvestigationStatus.COMPLETED
        )
    assert "ACTIVE" in str(exc.value) and "COMPLETED" in str(exc.value)


def test_completed_case_can_reopen():
    """Real investigations reopen; forbidding it would push work off-platform."""
    assert enums.can_transition(
        "investigation_status", InvestigationStatus.COMPLETED, InvestigationStatus.ACTIVE
    )


def test_reopening_requires_a_reason():
    assert enums.requires_reason(
        "investigation_status", InvestigationStatus.COMPLETED, InvestigationStatus.ACTIVE
    )


def test_forward_transitions_need_no_reason():
    assert not enums.requires_reason(
        "investigation_status", InvestigationStatus.NEW, InvestigationStatus.ACTIVE
    )


def test_deletion_intent_is_reversible():
    """A retention hold placed after a deletion request must be able to pull the
    case back — deletion intent is not deletion."""
    assert enums.can_transition(
        "investigation_status",
        InvestigationStatus.DELETED_PENDING_RETENTION,
        InvestigationStatus.ARCHIVED,
    )


def test_self_transition_is_rejected():
    """A no-op would write a lifecycle event recording no change."""
    with pytest.raises(IllegalTransition):
        enums.assert_transition(
            "investigation_status", InvestigationStatus.ACTIVE, InvestigationStatus.ACTIVE
        )


def test_every_status_appears_in_the_transition_table():
    """A status with no entry would raise KeyError the first time a real case
    reached it."""
    for status in InvestigationStatus:
        assert status in enums.INVESTIGATION_TRANSITIONS, f"{status} has no transition entry"


# -- evidence lifecycle: two independent axes -------------------------------


def test_progress_and_retention_are_separate_machines():
    """An artifact can be INCLUDED_IN_REPORT and RETAINED at once. A single
    column could not express that, and it is the combination compliance cares
    about."""
    assert enums.can_transition(
        "progress_state", ProgressState.REVIEWED, ProgressState.INCLUDED_IN_REPORT
    )
    assert enums.can_transition(
        "retention_state", RetentionState.ACTIVE, RetentionState.RETAINED
    )
    # A retention state is not reachable on the progress axis, and vice versa.
    with pytest.raises(IllegalTransition):
        enums.assert_transition("progress_state", ProgressState.DISCOVERED, "RETAINED")
    with pytest.raises(IllegalTransition):
        enums.assert_transition("retention_state", RetentionState.ACTIVE, "REVIEWED")


def test_purged_is_terminal():
    assert enums.allowed_transitions("retention_state", RetentionState.PURGED) == frozenset()


def test_purge_is_unreachable_from_active_directly():
    """Purge goes through ARCHIVED, so the retention engine always has a
    checkpoint at which to test for holds."""
    with pytest.raises(IllegalTransition):
        enums.assert_transition(
            "retention_state", RetentionState.ACTIVE, RetentionState.PURGED
        )


def test_verification_can_be_rerun():
    """A fetcher fix must be able to re-verify an already-verified artifact."""
    assert enums.can_transition(
        "progress_state", ProgressState.VERIFIED, ProgressState.DOWNLOADED
    )


def test_unknown_machine_and_state_raise_clearly():
    with pytest.raises(KeyError, match="Unknown state machine"):
        enums.allowed_transitions("nonexistent", "X")
    with pytest.raises(KeyError, match="unknown state"):
        enums.allowed_transitions("investigation_status", "NOT_A_STATUS")


# -- hash chains ------------------------------------------------------------


def _record(index: int, previous: str) -> dict:
    body = {"action": f"act-{index}", "outcome": "ok"}
    return {**body, "previous_hash": previous, "entry_hash": chain_hash(body, previous)}


def _chain(length: int) -> list[dict]:
    records, previous = [], GENESIS
    for i in range(length):
        record = _record(i, previous)
        records.append(record)
        previous = record["entry_hash"]
    return records


BODY = ("action", "outcome")


def test_intact_chain_verifies():
    result = verify_chain(_chain(5), body_fields=BODY)
    assert result.valid and result.records == 5 and result.broken_at is None


def test_empty_chain_verifies():
    result = verify_chain([], body_fields=BODY)
    assert result.valid and result.records == 0


def test_editing_a_record_is_detected_at_that_index():
    records = _chain(5)
    records[2]["outcome"] = "retrospectively altered"
    result = verify_chain(records, body_fields=BODY)
    assert not result.valid
    assert result.broken_at == 2
    assert "edited" in result.reason


def test_deleting_a_record_is_detected():
    records = _chain(5)
    del records[2]
    result = verify_chain(records, body_fields=BODY)
    assert not result.valid
    assert result.broken_at == 2


def test_reordering_is_detected():
    records = _chain(5)
    records[1], records[3] = records[3], records[1]
    result = verify_chain(records, body_fields=BODY)
    assert not result.valid


def test_canonical_hashing_is_key_order_independent():
    """If the same content hashed differently between processes, every chain
    check would fail and the guarantee would be worthless."""
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_different_content_hashes_differently():
    assert content_hash({"a": 1}) != content_hash({"a": 2})


# -- identifiers ------------------------------------------------------------


def test_uuid7_is_version_7_and_time_ordered():
    earlier = uuid7(1_700_000_000_000)
    later = uuid7(1_700_000_001_000)
    assert earlier.version == 7 and later.version == 7
    assert str(earlier) < str(later), "UUIDv7 must sort chronologically as text"


def test_uuid7_timestamp_round_trips():
    assert timestamp_ms_of(uuid7(1_700_000_000_123)) == 1_700_000_000_123


def test_sequential_generator_is_deterministic_and_ordered():
    generator = SequentialIdGenerator()
    ids = [generator() for _ in range(5)]
    assert ids == sorted(ids, key=str)
    assert len(set(ids)) == 5


# -- clock ------------------------------------------------------------------


def test_naive_datetime_is_rejected():
    """An artifact collected in one timezone and reviewed in another makes the
    'assume local' shortcut a defect."""
    with pytest.raises(ValueError, match="Naive datetime"):
        require_utc(datetime(2026, 1, 1))


def test_frozen_clock_advances_on_demand():
    clock = FrozenClock(datetime(2026, 1, 1, tzinfo=UTC))
    first = clock.now()
    clock.advance(hours=3)
    assert (clock.now() - first).total_seconds() == 3 * 3600


# -- configuration ----------------------------------------------------------


def test_sync_database_driver_is_rejected():
    """`postgresql://` selects psycopg2, which blocks the event loop — a
    mysteriously slow API rather than an error."""
    with pytest.raises(ValueError, match="async driver"):
        Settings(database_url="postgresql://user:pass@localhost/db")


def test_async_drivers_are_accepted():
    assert Settings(database_url="postgresql+asyncpg://u:p@localhost/db")
    assert Settings(database_url="sqlite+aiosqlite:///./x.db").is_sqlite


def test_production_requires_an_explicit_secret_key():
    with pytest.raises(ValueError, match="IIE_SECRET_KEY"):
        Settings(
            environment="production",
            secret_key="short",
            database_url="postgresql+asyncpg://u:p@db/iie",
        )


def test_production_refuses_to_disable_lawful_basis():
    with pytest.raises(ValueError, match="LAWFUL_BASIS"):
        Settings(
            environment="production",
            secret_key="x" * 40,
            require_lawful_basis=False,
            database_url="postgresql+asyncpg://u:p@db/iie",
        )


def test_production_refuses_sqlite():
    with pytest.raises(ValueError, match="SQLite is not supported"):
        Settings(
            environment="production",
            secret_key="x" * 40,
            database_url="sqlite+aiosqlite:///./iie.db",
        )


def test_local_environment_boots_without_configuration():
    """The stack must come up with no .env for a first-run investigator."""
    settings = Settings(environment="local")
    assert len(settings.secret_key) >= 32
    assert settings.require_lawful_basis is True


def test_restrictive_defaults():
    """Widening the blast radius must be opt-in."""
    settings = Settings()
    assert settings.require_lawful_basis is True
    assert settings.respect_robots is True
    assert settings.enable_plate_extraction is False
    assert settings.enable_face_detection_for_redaction is False


def test_alembic_url_strips_the_async_driver():
    settings = Settings(database_url="postgresql+asyncpg://u:p@localhost/db")
    assert "+asyncpg" not in settings.alembic_url
