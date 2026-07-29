"""Tests for CLI dataset maps and strategy loading."""

from fugu.cli.collect import (
    _DATASET_MAP as COLLECT_MAP,
    _STRATEGY_CLI_CHOICES,
    _load_strategies,
)
from fugu.cli.eval import _DATASET_MAP as EVAL_MAP
from fugu.core.actions import PlannerAction
from fugu.core.state import PlannerState
from fugu.trajectory.strategies import SingleWorkerStrategy


def test_collect_includes_livecodebench_and_swebench():
    assert "livecodebench-train" in COLLECT_MAP
    assert "livecodebench-test" in COLLECT_MAP
    assert "swebench-lite" in COLLECT_MAP
    assert "humaneval" not in COLLECT_MAP


def test_collect_swebench_uses_split_kwarg():
    for name, (_mod, _cls, kwargs) in COLLECT_MAP.items():
        if name.startswith("swebench"):
            assert "split" in kwargs, name
            assert "variant" not in kwargs, name
        if name.startswith("livecodebench"):
            assert "split" in kwargs, name


def test_eval_dataset_keys_match_collect():
    assert set(COLLECT_MAP) == set(EVAL_MAP)


def test_load_strategies_single_qwen_cli_alias():
    strategies = _load_strategies("single-qwen")
    assert len(strategies) == 1
    s = strategies[0]
    assert isinstance(s, SingleWorkerStrategy)
    assert s.name == "single_worker_call_qwen"
    assert s.select_action(PlannerState(), 0) is PlannerAction.CALL_QWEN


def test_load_strategies_all_cli_choices_resolve_to_one():
    for name in _STRATEGY_CLI_CHOICES:
        if name == "all":
            assert len(_load_strategies(name)) > 1
        else:
            assert len(_load_strategies(name)) == 1, name
