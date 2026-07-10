"""Tests for CLI dataset maps (split= not variant=)."""

from fugu.cli.collect import _DATASET_MAP as COLLECT_MAP
from fugu.cli.eval import _DATASET_MAP as EVAL_MAP


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
