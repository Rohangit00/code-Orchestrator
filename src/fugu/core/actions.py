"""Planner action space for the Fugu orchestrator MDP."""

from __future__ import annotations

from enum import IntEnum


class PlannerAction(IntEnum):
    """Discrete action space for the planner.

    Each action represents a high-level decision the planner can make
    at each step of the orchestration process.
    """

    CALL_QWEN = 0
    CALL_GEMMA = 1
    CALL_ORNITH = 2
    RUN_TESTS = 3
    VERIFY = 4
    RETRY = 5
    STOP = 6

    @property
    def is_worker_call(self) -> bool:
        """Return True if this action dispatches work to a coding model."""
        return self in WORKER_ACTIONS

    @classmethod
    def from_string(cls, s: str) -> PlannerAction:
        """Parse a planner action from its string name (case-insensitive).

        Args:
            s: Action name such as ``"CALL_QWEN"`` or ``"call_qwen"``.

        Returns:
            The corresponding ``PlannerAction`` member.

        Raises:
            ValueError: If *s* does not match any known action name.
        """
        normalized = s.strip().upper()
        try:
            return cls[normalized]
        except KeyError:
            valid = ", ".join(cls.__members__)
            raise ValueError(
                f"Unknown action {s!r}. Valid actions: {valid}"
            ) from None


# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

WORKER_ACTIONS: frozenset[PlannerAction] = frozenset({
    PlannerAction.CALL_QWEN,
    PlannerAction.CALL_GEMMA,
    PlannerAction.CALL_ORNITH,
})

ACTION_NAMES: dict[PlannerAction, str] = {action: action.name for action in PlannerAction}

NUM_ACTIONS: int = len(PlannerAction)
