"""Shared planner prompt construction for training and inference.

Both :class:`~fugu.training.data.TransitionDataset` and
:class:`~fugu.planner.model.PlannerModel` must use the same formatting so
that supervised fine-tuning matches deployment-time inputs.
"""

from __future__ import annotations

from typing import Any

from fugu.core.state import PlannerState

# System prompt template used to frame the action-selection task.
SYSTEM_PROMPT = (
    "You are a coding orchestrator that manages multiple AI coding workers. "
    "Your job is to choose the best next action to solve a coding task.\n\n"
    "Available actions:\n"
    "  CALL_QWEN  — Dispatch the task to the Qwen coding model.\n"
    "  CALL_GEMMA — Dispatch the task to the Gemma coding model.\n"
    "  CALL_ORNITH — Dispatch the task to the Ornith coding model.\n"
    "  RUN_TESTS  — Run the test suite without modifying code.\n"
    "  VERIFY     — Run a full verification (compile check + tests).\n"
    "  RETRY      — Re-call the last worker with error context.\n"
    "  STOP       — Stop the episode and submit the current solution.\n\n"
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
