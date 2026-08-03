"""Train/infer prompt parity (issue #9)."""

from fugu.core.actions import PlannerAction
from fugu.core.state import PlannerState, Transition
from fugu.planner.prompts import SYSTEM_PROMPT, build_planner_prompt
from fugu.planner.tokenizer import StateTokenizer


def test_build_planner_prompt_includes_system_and_state():
    state = PlannerState(task_description="hello task")
    prompt = build_planner_prompt(state, tokenizer=None)
    assert "coding orchestrator" in prompt.lower() or "CALL_QWEN" in prompt
    assert "hello task" in prompt
    assert SYSTEM_PROMPT.split("\n")[0] in prompt or "CALL_QWEN" in prompt
    assert "CALL_QWEN" in prompt
    assert "CALL_ORNITH" in prompt
    # Gemma disabled — must not appear in planner prompt
    assert "CALL_GEMMA" not in prompt
    assert "CALL_GEMMA" not in SYSTEM_PROMPT


def test_tokenizer_training_uses_shared_builder():
    state = PlannerState(task_description="shared prompt task")
    tr = Transition(
        state=state,
        action=PlannerAction.RUN_TESTS,
        reward=0.0,
        next_state=None,
        done=False,
    )
    tok = StateTokenizer()
    pair = tok.format_for_training(tr, tokenizer=None)
    direct = build_planner_prompt(state, tokenizer=None)
    assert pair["prompt"] == direct
    assert pair["completion"] == "RUN_TESTS"
