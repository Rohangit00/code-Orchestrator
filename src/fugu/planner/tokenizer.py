"""State tokenizer — converts states and transitions into training text.

Provides the text-level interface between the MDP representation
(:class:`~fugu.core.state.PlannerState`, :class:`~fugu.core.state.Transition`)
and the token-level representation consumed by the planner LM during
supervised fine-tuning.

Prompt construction is shared with inference via
:func:`~fugu.planner.prompts.build_planner_prompt`.
"""

from __future__ import annotations

from typing import Any

from fugu.core.actions import ACTION_NAMES, PlannerAction
from fugu.core.state import PlannerState, Transition
from fugu.planner.prompts import build_planner_prompt


class StateTokenizer:
    """Converts :class:`PlannerState` / :class:`Transition` into training-ready text.

    The class is intentionally lightweight — HuggingFace tokenisation is
    handled downstream by :class:`~fugu.training.data.TransitionDataset`.
    """

    SPECIAL_TOKENS: list[str] = [
        "<|task|>",
        "<|/task|>",
        "<|repo|>",
        "<|/repo|>",
        "<|history|>",
        "<|/history|>",
        "<|tests|>",
        "<|/tests|>",
        "<|compile|>",
        "<|/compile|>",
        "<|step|>",
        "<|/step|>",
    ]

    def format_state(
        self,
        state: PlannerState,
        *,
        tokenizer: Any | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Serialize *state* with the shared train/infer prompt builder."""
        return build_planner_prompt(
            state,
            tokenizer=tokenizer,
            add_generation_prompt=add_generation_prompt,
        )

    def format_for_training(
        self,
        transition: Transition,
        *,
        tokenizer: Any | None = None,
    ) -> dict[str, str]:
        """Convert a single transition into a ``{prompt, completion}`` pair.

        Uses the same prompt formatting as :meth:`PlannerModel.predict`.
        """
        prompt = self.format_state(
            transition.state,
            tokenizer=tokenizer,
            add_generation_prompt=True,
        )
        completion = ACTION_NAMES[transition.action]
        return {"prompt": prompt, "completion": completion}

    def format_batch(
        self,
        transitions: list[Transition],
        *,
        tokenizer: Any | None = None,
    ) -> list[dict[str, str]]:
        """Format a list of transitions for training."""
        return [
            self.format_for_training(t, tokenizer=tokenizer) for t in transitions
        ]

    @staticmethod
    def get_action_labels() -> list[str]:
        """Return the ordered list of human-readable action name strings."""
        return [ACTION_NAMES[action] for action in PlannerAction]
