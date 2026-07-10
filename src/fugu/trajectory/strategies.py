"""Fixed orchestration strategies for trajectory collection.

Each strategy defines a deterministic policy over the :class:`PlannerAction`
space.  They are used by the :class:`TrajectoryCollector` to generate
behavioural-cloning trajectories without a trained planner model.

``RETRY`` is a **complete** re-call of the last worker (env applies patch and
runs tests). Strategies must not emit ``RETRY`` followed by another
``CALL_*`` of the same worker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fugu.core.actions import PlannerAction
from fugu.core.state import PlannerState


def _tests_fully_pass(state: PlannerState) -> bool:
    tr = state.test_results
    return (
        tr is not None
        and tr.total > 0
        and tr.failed == 0
        and tr.errors == 0
    )


class BaseStrategy(ABC):
    """Abstract base class for trajectory-collection strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this strategy."""
        ...

    @abstractmethod
    def select_action(self, state: PlannerState, step: int) -> PlannerAction:
        """Choose the next action given the current planner state and step index."""
        ...


class SingleWorkerStrategy(BaseStrategy):
    """Call one worker; retry on failure without double worker calls.

    Pattern (env runs tests after worker / RETRY)::

        CALL_X -> (STOP if solved, else RETRY)* -> STOP after max_retries

    Explicit ``RUN_TESTS`` is optional; the environment already evaluates
    after worker patches. We keep a single ``RUN_TESTS`` only when the first
    worker step did not produce test results (edge case).

    Parameters
    ----------
    worker_action:
        The worker action to dispatch.
    max_retries:
        Maximum number of ``RETRY`` actions after the initial call.
    """

    def __init__(
        self,
        worker_action: PlannerAction = PlannerAction.CALL_QWEN,
        max_retries: int = 2,
    ) -> None:
        if not worker_action.is_worker_call:
            raise ValueError(
                f"{worker_action!r} is not a worker action. "
                "Must be one of CALL_QWEN, CALL_GEMMA, CALL_ORNITH."
            )
        self._worker_action = worker_action
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return f"single_worker_{self._worker_action.name.lower()}"

    def select_action(self, state: PlannerState, step: int) -> PlannerAction:
        if _tests_fully_pass(state):
            return PlannerAction.STOP

        if step == 0:
            return self._worker_action

        # After initial call: RETRY up to max_retries, then STOP.
        # step 1 → first RETRY, …, step max_retries → last RETRY,
        # step max_retries+1 → STOP
        retry_index = step - 1
        if retry_index < self._max_retries:
            return PlannerAction.RETRY
        return PlannerAction.STOP


class RoundRobinStrategy(BaseStrategy):
    """Cycles through workers; env evaluates after each worker call.

    Pattern::

        CALL_QWEN -> CALL_GEMMA -> CALL_ORNITH -> STOP

    Early ``STOP`` if tests already fully pass (env may have auto-terminated).
    """

    _WORKERS = [
        PlannerAction.CALL_QWEN,
        PlannerAction.CALL_GEMMA,
        PlannerAction.CALL_ORNITH,
    ]

    @property
    def name(self) -> str:
        return "round_robin"

    def select_action(self, state: PlannerState, step: int) -> PlannerAction:
        if _tests_fully_pass(state):
            return PlannerAction.STOP

        if step < len(self._WORKERS):
            return self._WORKERS[step]
        return PlannerAction.STOP


class RetryOnFailStrategy(BaseStrategy):
    """Primary worker with RETRY, then fallback workers (no double CALL).

    Pattern::

        CALL_PRIMARY
        -> RETRY (up to max_retries) if still failing
        -> CALL_FALLBACK_0, CALL_FALLBACK_1, …
        -> STOP

    ``RETRY`` fully re-invokes the last worker inside the environment.
    """

    def __init__(
        self,
        primary_action: PlannerAction = PlannerAction.CALL_QWEN,
        max_retries: int = 2,
        fallback_actions: list[PlannerAction] | None = None,
    ) -> None:
        if not primary_action.is_worker_call:
            raise ValueError(f"{primary_action!r} is not a worker action.")
        self._primary = primary_action
        self._max_retries = max_retries
        self._fallbacks = fallback_actions or [
            a
            for a in [PlannerAction.CALL_GEMMA, PlannerAction.CALL_ORNITH]
            if a != primary_action
        ]

    @property
    def name(self) -> str:
        return f"retry_on_fail_{self._primary.name.lower()}"

    def select_action(self, state: PlannerState, step: int) -> PlannerAction:
        if _tests_fully_pass(state):
            return PlannerAction.STOP

        plan = self._build_plan()
        if step < len(plan):
            return plan[step]
        return PlannerAction.STOP

    def _build_plan(self) -> list[PlannerAction]:
        plan: list[PlannerAction] = [self._primary]
        for _ in range(self._max_retries):
            plan.append(PlannerAction.RETRY)
        for fb in self._fallbacks:
            plan.append(fb)
        return plan


class VerifyFirstStrategy(BaseStrategy):
    """Baseline verify, one worker, optional verify, stop.

    Pattern::

        VERIFY -> CALL_X -> VERIFY -> STOP

    (Env may auto-stop after CALL_X if tests pass.)
    """

    def __init__(
        self,
        worker_action: PlannerAction = PlannerAction.CALL_QWEN,
    ) -> None:
        if not worker_action.is_worker_call:
            raise ValueError(f"{worker_action!r} is not a worker action.")
        self._worker_action = worker_action

    @property
    def name(self) -> str:
        return f"verify_first_{self._worker_action.name.lower()}"

    def select_action(self, state: PlannerState, step: int) -> PlannerAction:
        if step > 0 and _tests_fully_pass(state):
            return PlannerAction.STOP

        sequence = [
            PlannerAction.VERIFY,
            self._worker_action,
            PlannerAction.VERIFY,
            PlannerAction.STOP,
        ]
        if step < len(sequence):
            return sequence[step]
        return PlannerAction.STOP


ALL_STRATEGIES: list[BaseStrategy] = [
    SingleWorkerStrategy(PlannerAction.CALL_QWEN, max_retries=2),
    SingleWorkerStrategy(PlannerAction.CALL_GEMMA, max_retries=2),
    SingleWorkerStrategy(PlannerAction.CALL_ORNITH, max_retries=2),
    RoundRobinStrategy(),
    RetryOnFailStrategy(
        primary_action=PlannerAction.CALL_QWEN,
        max_retries=2,
        fallback_actions=[PlannerAction.CALL_GEMMA, PlannerAction.CALL_ORNITH],
    ),
    VerifyFirstStrategy(PlannerAction.CALL_QWEN),
]
