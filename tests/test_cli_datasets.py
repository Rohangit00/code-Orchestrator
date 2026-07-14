"""Tests for CLI dataset maps (split= not variant=) and strategy loading."""

from fugu.cli.collect import (
    _DATASET_MAP as COLLECT_MAP,
    _STRATEGY_CLI_CHOICES,
    _load_strategies,
)
from fugu.cli.eval import _DATASET_MAP as EVAL_MAP
from fugu.core.actions import PlannerAction
from fugu.trajectory.strategies import SingleWorkerStrategy


def test_collect_only_swebench():
    assert set(COLLECT_MAP) == {
        "swebench-lite",
        "swebench-full",
        "swebench-verified",
    }
    assert "humaneval" not in COLLECT_MAP
    assert "mbpp" not in COLLECT_MAP


def test_collect_uses_split_kwarg():
    for name, (_mod, _cls, kwargs) in COLLECT_MAP.items():
        assert "split" in kwargs, name
        assert "variant" not in kwargs, name


def test_eval_matches_collect():
    assert COLLECT_MAP == EVAL_MAP


def test_load_strategies_single_qwen_cli_alias():
    """CLI flag single-qwen must not fall back to ALL_STRATEGIES."""
    strategies = _load_strategies("single-qwen")
    assert len(strategies) == 1
    s = strategies[0]
    assert isinstance(s, SingleWorkerStrategy)
    assert s.name == "single_worker_call_qwen"
    # First action is CALL_QWEN
    from fugu.core.state import PlannerState

    assert s.select_action(PlannerState(), 0) is PlannerAction.CALL_QWEN


def test_load_strategies_all_cli_choices_resolve_to_one():
    for name in _STRATEGY_CLI_CHOICES:
        if name == "all":
            assert len(_load_strategies(name)) > 1
        else:
            assert len(_load_strategies(name)) == 1, name
