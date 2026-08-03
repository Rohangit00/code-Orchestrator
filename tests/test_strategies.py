"""Tests for fixed strategies (issue #8 — no double CALL after RETRY)."""

import pytest

from fugu.core.actions import PlannerAction
from fugu.core.state import PlannerState, TestResults
from fugu.trajectory.strategies import (
    RetryOnFailStrategy,
    RoundRobinStrategy,
    SingleWorkerStrategy,
)


def test_retry_on_fail_no_double_primary():
    strat = RetryOnFailStrategy(
        primary_action=PlannerAction.CALL_ORNITH,
        max_retries=2,
        fallback_actions=[PlannerAction.CALL_QWEN],
    )
    plan = strat._build_plan()
    assert plan[0] is PlannerAction.CALL_ORNITH
    assert plan[1] is PlannerAction.RETRY
    assert plan[2] is PlannerAction.RETRY
    assert plan[3] is PlannerAction.CALL_QWEN
    assert PlannerAction.CALL_GEMMA not in plan
    for i in range(len(plan) - 1):
        if plan[i] is PlannerAction.RETRY:
            assert plan[i + 1] is not PlannerAction.CALL_ORNITH


def test_single_worker_retry_then_stop():
    strat = SingleWorkerStrategy(PlannerAction.CALL_ORNITH, max_retries=2)
    state = PlannerState(test_results=TestResults(passed=0, failed=1, errors=0))
    assert strat.select_action(state, 0) is PlannerAction.CALL_ORNITH
    assert strat.select_action(state, 1) is PlannerAction.RETRY
    assert strat.select_action(state, 2) is PlannerAction.RETRY
    assert strat.select_action(state, 3) is PlannerAction.STOP


def test_strategy_stops_when_tests_pass():
    strat = SingleWorkerStrategy(PlannerAction.CALL_QWEN, max_retries=5)
    state = PlannerState(test_results=TestResults(passed=2, failed=0, errors=0))
    assert strat.select_action(state, 1) is PlannerAction.STOP


def test_single_gemma_rejected():
    with pytest.raises(ValueError, match="disabled"):
        SingleWorkerStrategy(PlannerAction.CALL_GEMMA)


def test_round_robin_skips_gemma():
    strat = RoundRobinStrategy()
    state = PlannerState(test_results=TestResults(passed=0, failed=1, errors=0))
    assert strat.select_action(state, 0) is PlannerAction.CALL_ORNITH
    assert strat.select_action(state, 1) is PlannerAction.CALL_QWEN
    assert strat.select_action(state, 2) is PlannerAction.STOP
