"""Tests for RewardCalculator (worker-aware cost; latency off; Gemma disabled)."""

from fugu.core.actions import PlannerAction
from fugu.core.reward import RewardCalculator, RewardConfig
from fugu.core.state import CompileStatus, Metadata, TestResults


def test_test_improvement_positive():
    calc = RewardCalculator(
        RewardConfig(w_tests=1.0, w_compile=0.0, w_cost=0.0, w_latency=0.0, w_retry=0.0)
    )
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
    calc = RewardCalculator(
        RewardConfig(terminal_bonus=2.0, w_tests=0.0, w_compile=0.0, w_cost=0.0, w_latency=0.0)
    )
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
    calc = RewardCalculator(
        RewardConfig(w_retry=0.5, w_tests=0.0, w_compile=0.0, w_cost=0.0, w_latency=0.0)
    )
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


def test_default_latency_weight_is_zero():
    assert RewardConfig().w_latency == 0.0


def test_worker_cost_multipliers_two_workers():
    cfg = RewardConfig()
    calc = RewardCalculator(cfg)
    assert calc.worker_cost_multiplier(PlannerAction.CALL_ORNITH) == cfg.cost_ornith
    assert calc.worker_cost_multiplier(PlannerAction.CALL_QWEN) == cfg.cost_qwen
    assert calc.worker_cost_multiplier(PlannerAction.RUN_TESTS) == cfg.cost_other
    # Disabled Gemma is not a billable cost tier
    assert calc.worker_cost_multiplier(PlannerAction.CALL_GEMMA) == cfg.cost_other
    assert (
        calc.worker_cost_multiplier(
            PlannerAction.RETRY, billed_worker=PlannerAction.CALL_QWEN
        )
        == cfg.cost_qwen
    )


def test_ornith_cheaper_than_qwen_same_tokens():
    """Same solve signal → Ornith step reward > Qwen step reward."""
    calc = RewardCalculator()
    common = dict(
        tests_before=TestResults(passed=0, failed=1, errors=0),
        tests_after=TestResults(passed=1, failed=0, errors=0),
        compile_status=CompileStatus(success=True),
        tokens_used=2048,
        latency_ms=5000.0,
        is_terminal=True,
        all_tests_passed=True,
        budget_exhausted=False,
    )
    r_ornith = calc.compute(action=PlannerAction.CALL_ORNITH, **common)
    r_qwen = calc.compute(action=PlannerAction.CALL_QWEN, **common)
    assert r_ornith > r_qwen
    assert r_qwen > 1.0


def test_latency_disabled_by_default():
    calc = RewardCalculator()
    base = dict(
        action=PlannerAction.CALL_QWEN,
        tests_before=None,
        tests_after=None,
        compile_status=None,
        tokens_used=100,
        is_terminal=False,
        all_tests_passed=False,
        budget_exhausted=False,
    )
    r_fast = calc.compute(latency_ms=1.0, **base)
    r_slow = calc.compute(latency_ms=30_000.0, **base)
    assert r_fast == r_slow


def test_latency_applies_when_enabled():
    calc = RewardCalculator(RewardConfig(w_latency=0.5, w_cost=0.0, w_tests=0.0, w_compile=0.0))
    common = dict(
        action=PlannerAction.CALL_QWEN,
        tests_before=None,
        tests_after=None,
        compile_status=None,
        tokens_used=0,
        is_terminal=False,
        all_tests_passed=False,
        budget_exhausted=False,
    )
    r_fast = calc.compute(latency_ms=0.0, **common)
    r_slow = calc.compute(latency_ms=30_000.0, **common)
    assert r_slow < r_fast
    assert abs(r_slow - (-0.5)) < 1e-6


def test_solve_beats_fail_even_with_expensive_worker():
    calc = RewardCalculator()
    solved = calc.compute(
        action=PlannerAction.CALL_QWEN,
        tests_before=TestResults(passed=0, failed=1, errors=0),
        tests_after=TestResults(passed=1, failed=0, errors=0),
        compile_status=CompileStatus(success=True),
        tokens_used=8192,
        latency_ms=0.0,
        is_terminal=True,
        all_tests_passed=True,
        budget_exhausted=False,
    )
    failed = calc.compute(
        action=PlannerAction.CALL_QWEN,
        tests_before=TestResults(passed=0, failed=1, errors=0),
        tests_after=TestResults(passed=0, failed=1, errors=0),
        compile_status=CompileStatus(success=True),
        tokens_used=8192,
        latency_ms=0.0,
        is_terminal=True,
        all_tests_passed=False,
        budget_exhausted=True,
    )
    assert solved > failed
    assert solved > 0
    assert failed < 0


def test_compute_from_metadata_uses_worker_name():
    calc = RewardCalculator(RewardConfig(w_tests=0.0, w_compile=0.0, w_retry=0.0))
    meta_q = Metadata(worker_name="qwen", tokens_used=4096)
    meta_o = Metadata(worker_name="ornith", tokens_used=4096)
    r_q = calc.compute_from_metadata(
        PlannerAction.CALL_QWEN,
        meta_q,
        is_terminal=False,
        all_tests_passed=False,
        budget_exhausted=False,
    )
    r_o = calc.compute_from_metadata(
        PlannerAction.CALL_ORNITH,
        meta_o,
        is_terminal=False,
        all_tests_passed=False,
        budget_exhausted=False,
    )
    assert r_o > r_q
