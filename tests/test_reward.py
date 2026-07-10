"""Tests for RewardCalculator."""

from fugu.core.actions import PlannerAction
from fugu.core.reward import RewardCalculator, RewardConfig
from fugu.core.state import CompileStatus, TestResults


def test_test_improvement_positive():
    calc = RewardCalculator(RewardConfig(w_tests=1.0, w_compile=0.0, w_cost=0.0, w_latency=0.0, w_retry=0.0))
    before = TestResults(passed=1, failed=1, errors=0)
    after = TestResults(passed=2, failed=0, errors=0)
    r = calc.compute(
        action=PlannerAction.RUN_TESTS,
        tests_before=before,
        tests_after=after,
        compile_status=None,
        tokens_used=0,
        latency_ms=0.0,
        is_terminal=False,
        all_tests_passed=True,
        budget_exhausted=False,
    )
    assert r > 0


def test_terminal_bonus():
    calc = RewardCalculator(RewardConfig(terminal_bonus=2.0, w_tests=0.0, w_compile=0.0, w_cost=0.0, w_latency=0.0))
    r = calc.compute(
        action=PlannerAction.STOP,
        tests_before=None,
        tests_after=TestResults(passed=1, failed=0, errors=0),
        compile_status=CompileStatus(success=True),
        tokens_used=0,
        latency_ms=0.0,
        is_terminal=True,
        all_tests_passed=True,
        budget_exhausted=False,
    )
    assert r >= 2.0


def test_retry_penalty():
    calc = RewardCalculator(RewardConfig(w_retry=0.5, w_tests=0.0, w_compile=0.0, w_cost=0.0, w_latency=0.0))
    r = calc.compute(
        action=PlannerAction.RETRY,
        tests_before=None,
        tests_after=None,
        compile_status=None,
        tokens_used=0,
        latency_ms=0.0,
        is_terminal=False,
        all_tests_passed=False,
        budget_exhausted=False,
    )
    assert r == -0.5
