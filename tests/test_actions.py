"""Tests for PlannerAction enum."""

from fugu.core.actions import (
    ACTION_NAMES,
    ACTIVE_ACTIONS,
    DISABLED_WORKER_ACTIONS,
    ENABLED_WORKER_ACTIONS,
    NUM_ACTIONS,
    WORKER_ACTIONS,
    PlannerAction,
    is_enabled_worker,
)


def test_action_count():
    assert NUM_ACTIONS == 7
    assert len(list(PlannerAction)) == 7


def test_action_values():
    assert PlannerAction.CALL_QWEN == 0
    assert PlannerAction.STOP == 6


def test_worker_calls():
    assert PlannerAction.CALL_QWEN.is_worker_call
    assert PlannerAction.CALL_GEMMA.is_worker_call  # still a worker enum value
    assert PlannerAction.CALL_ORNITH.is_worker_call
    assert not PlannerAction.RUN_TESTS.is_worker_call
    assert not PlannerAction.RETRY.is_worker_call
    assert WORKER_ACTIONS == {
        PlannerAction.CALL_QWEN,
        PlannerAction.CALL_GEMMA,
        PlannerAction.CALL_ORNITH,
    }


def test_gemma_disabled_two_active_workers():
    assert ENABLED_WORKER_ACTIONS == {
        PlannerAction.CALL_QWEN,
        PlannerAction.CALL_ORNITH,
    }
    assert PlannerAction.CALL_GEMMA in DISABLED_WORKER_ACTIONS
    assert is_enabled_worker(PlannerAction.CALL_QWEN)
    assert is_enabled_worker(PlannerAction.CALL_ORNITH)
    assert not is_enabled_worker(PlannerAction.CALL_GEMMA)
    assert PlannerAction.CALL_GEMMA not in ACTIVE_ACTIONS
    assert PlannerAction.CALL_QWEN in ACTIVE_ACTIONS


def test_from_string():
    assert PlannerAction.from_string("call_qwen") is PlannerAction.CALL_QWEN
    assert PlannerAction.from_string("STOP") is PlannerAction.STOP


def test_from_string_invalid():
    try:
        PlannerAction.from_string("NOPE")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_action_names():
    assert ACTION_NAMES[PlannerAction.RETRY] == "RETRY"
