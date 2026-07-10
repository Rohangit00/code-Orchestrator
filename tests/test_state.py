"""Tests for PlannerState and Transition serialisation."""

from fugu.core.actions import PlannerAction
from fugu.core.state import (
    CompileStatus,
    Metadata,
    PlannerState,
    TestResults,
    Transition,
)


def test_to_prompt_contains_tags(sample_state: PlannerState):
    prompt = sample_state.to_prompt()
    assert "<|task|>" in prompt
    assert "Fix the bug" in prompt
    assert "<|tests|>" in prompt
    assert "<|step|>0/10" in prompt


def test_test_results_pass_rate():
    tr = TestResults(passed=3, failed=1, errors=0)
    assert tr.total == 4
    assert tr.pass_rate == 0.75


def test_transition_roundtrip(sample_transition: Transition):
    data = sample_transition.to_dict()
    restored = Transition.from_dict(data)
    assert restored.action is PlannerAction.CALL_QWEN
    assert restored.reward == 0.5
    assert restored.done is False
    assert restored.state.task_description == sample_transition.state.task_description
    assert restored.metadata.task_id == "t1"


def test_metadata_defaults():
    m = Metadata()
    assert m.worker_name == ""
    assert m.compile_success is True
