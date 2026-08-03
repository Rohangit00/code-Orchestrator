"""Shared planner prompt construction for training and inference.

Both :class:`~fugu.training.data.TransitionDataset` and
:class:`~fugu.planner.model.PlannerModel` must use the same formatting so
that supervised fine-tuning matches deployment-time inputs.
"""

from __future__ import annotations

from typing import Any

from fugu.core.state import PlannerState

# System prompt template used to frame the action-selection task.
# Two active workers: CALL_QWEN (strong/expensive) and CALL_ORNITH (cheap).
# CALL_GEMMA is disabled and omitted so the planner does not learn it.
SYSTEM_PROMPT = (
    "You are a coding orchestrator that manages multiple AI coding workers. "
    "Your job is to choose the best next action to solve a coding task.\n\n"
    "Available actions:\n"
    "  CALL_QWEN   — Strong / expensive coding model. Use when the cheap "
    "worker fails or the problem looks hard.\n"
    "  CALL_ORNITH — Cheap / fast coding model. Prefer as the default first try.\n"
    "  RUN_TESTS   — Run the test suite without modifying code.\n"
    "  VERIFY      — Run a full verification (compile check + tests).\n"
    "  RETRY       — Re-call the last worker with error context.\n"
    "  STOP        — Stop the episode and submit the current solution.\n\n"
    "Prefer CALL_ORNITH when it is enough; escalate to CALL_QWEN when needed. "
    "Respond with ONLY the action name."
)


def build_planner_prompt(
    state: PlannerState,
    *,
    tokenizer: Any | None = None,
    add_generation_prompt: bool = True,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Build the full planner prompt for train and inference.

    Parameters
    ----------
    state:
        Current planner observation.
    tokenizer:
        Optional HuggingFace tokenizer. When provided and it supports
        ``apply_chat_template``, that template is used.
    add_generation_prompt:
        Passed to ``apply_chat_template`` when available.
    system_prompt:
        Override for the system instruction (defaults to :data:`SYSTEM_PROMPT`).
    """
    state_text = state.to_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state_text},
    ]

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass

    suffix = "\n\nAction:" if add_generation_prompt else ""
    return f"{system_prompt}\n\n{state_text}{suffix}"
