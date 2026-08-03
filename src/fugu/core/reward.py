"""Reward calculation for the Fugu orchestrator MDP.

The reward signal combines multiple objectives:

* **Test improvement** – primary signal; measures delta in pass rate.
* **Compile success** – bonus for maintaining compilability.
* **Worker-aware cost** – penalises token usage scaled by which worker
  was billed (Ornith cheap, Qwen expensive). ``CALL_GEMMA`` is disabled
  and does not participate in the cost model.
* **Latency** – optional; default weight is 0 (disabled) for the primary
  cost-only experiment; set ``w_latency > 0`` for a later latency study.
* **Retry penalty** – discourages the planner from looping.
* **Terminal bonus / penalty** – awarded on episode completion.
"""

from __future__ import annotations

from dataclasses import dataclass

from fugu.core.actions import PlannerAction, WORKER_ACTIONS
from fugu.core.state import CompileStatus, Metadata, TestResults, Transition


@dataclass(slots=True)
class RewardConfig:
    """Tunable weights and thresholds for reward computation.

    Cost is **worker-dependent**: normalised tokens are multiplied by a
    per-worker coefficient so the same token budget costs more for Qwen
    than for Ornith (two active workers).  Latency is a separate optional
    term (``w_latency=0`` by default for the cost-only experiment).
    """

    # Component weights
    w_tests: float = 1.0
    w_compile: float = 0.3
    w_cost: float = 0.25
    # Primary experiment: cost only. Re-enable for a latency ablation.
    w_latency: float = 0.0
    w_retry: float = 0.1

    # Relative inference cost multipliers (Ornith = 1.0 baseline).
    # Active workers: Ornith (cheap) and Qwen (strong/expensive).
    cost_ornith: float = 1.0
    cost_qwen: float = 3.5
    # Non-LLM actions, disabled workers (Gemma), and unknown.
    cost_other: float = 0.0

    # Terminal bonuses / penalties
    terminal_bonus: float = 2.0
    budget_penalty: float = 1.0

    # Discount factor for episode-level return
    gamma: float = 0.99

    # Normalisation constants (used to scale raw values into ~[0, 1])
    max_tokens_per_step: int = 8192
    max_latency_ms: float = 30_000.0


class RewardCalculator:
    """Stateless reward calculator for MDP transitions.

    Instantiate once with a ``RewardConfig`` and call ``compute`` per step
    or ``compute_episode_reward`` over a full trajectory.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.cfg = config or RewardConfig()

    # -----------------------------------------------------------------
    # Per-step reward
    # -----------------------------------------------------------------

    def compute(
        self,
        action: PlannerAction,
        tests_before: TestResults | None,
        tests_after: TestResults | None,
        compile_status: CompileStatus | None,
        tokens_used: int,
        latency_ms: float,
        is_terminal: bool,
        all_tests_passed: bool,
        budget_exhausted: bool,
        *,
        billed_worker: PlannerAction | None = None,
    ) -> float:
        """Compute the scalar reward for a single environment step.

        Args:
            action: The planner action that was taken.
            tests_before: Test results *before* the action (may be ``None`` on
                the first step).
            tests_after: Test results *after* the action.
            compile_status: Compilation status after the action.
            tokens_used: Number of tokens consumed by this step.
            latency_ms: Wall-clock latency of the worker call in milliseconds.
                Only applied when ``cfg.w_latency > 0``.
            is_terminal: Whether this step ends the episode.
            all_tests_passed: Whether every test in the suite now passes.
            budget_exhausted: Whether the token / step budget is exhausted.
            billed_worker: Worker action whose cost multiplier applies.
                For ``CALL_*`` this is usually the action itself.  For
                ``RETRY``, pass the last worker that was re-invoked.  If
                omitted, inferred from *action* when it is a worker call.

        Returns:
            A scalar reward value.
        """
        reward = 0.0

        # 1. Test improvement -------------------------------------------
        reward += self.cfg.w_tests * self._test_improvement(tests_before, tests_after)

        # 2. Compile status ---------------------------------------------
        if compile_status is not None:
            reward += self.cfg.w_compile * (1.0 if compile_status.success else -1.0)

        # 3. Worker-aware token cost ------------------------------------
        mult = self.worker_cost_multiplier(action, billed_worker=billed_worker)
        if tokens_used > 0 and mult > 0.0 and self.cfg.w_cost != 0.0:
            denom = max(self.cfg.max_tokens_per_step, 1)
            normalized_cost = min(tokens_used / denom, 1.0)
            reward -= self.cfg.w_cost * mult * normalized_cost

        # 4. Latency (optional; default weight 0) -----------------------
        if self.cfg.w_latency != 0.0 and latency_ms > 0.0:
            denom_l = max(self.cfg.max_latency_ms, 1e-6)
            normalized_latency = min(latency_ms / denom_l, 1.0)
            reward -= self.cfg.w_latency * normalized_latency

        # 5. Retry penalty ----------------------------------------------
        if action is PlannerAction.RETRY:
            reward -= self.cfg.w_retry

        # 6. Terminal bonus / penalty -----------------------------------
        if is_terminal:
            if all_tests_passed:
                reward += self.cfg.terminal_bonus
            if budget_exhausted and not all_tests_passed:
                reward -= self.cfg.budget_penalty

        return reward

    def worker_cost_multiplier(
        self,
        action: PlannerAction,
        *,
        billed_worker: PlannerAction | None = None,
    ) -> float:
        """Return the cost coefficient for the worker billed on this step.

        * ``CALL_ORNITH`` / Ornith RETRY → ``cost_ornith``
        * ``CALL_QWEN`` / Qwen RETRY → ``cost_qwen``
        * Non-worker / disabled (Gemma) → ``cost_other`` (default 0)
        """
        worker = billed_worker
        if worker is None:
            if action in WORKER_ACTIONS:
                worker = action
            else:
                return self.cfg.cost_other

        if worker is PlannerAction.CALL_ORNITH:
            return self.cfg.cost_ornith
        if worker is PlannerAction.CALL_QWEN:
            return self.cfg.cost_qwen
        # CALL_GEMMA (disabled) and anything else
        return self.cfg.cost_other

    # -----------------------------------------------------------------
    # Episode-level discounted return
    # -----------------------------------------------------------------

    def compute_episode_reward(self, transitions: list[Transition]) -> float:
        """Compute the discounted cumulative return for a full episode.

        Uses the rewards already stored on each ``Transition`` and applies
        the configured discount factor *gamma*.

        Args:
            transitions: Ordered list of transitions forming one episode.

        Returns:
            The discounted return G_0 = r_0 + γ r_1 + γ² r_2 + ….
        """
        g = 0.0
        discount = 1.0
        for t in transitions:
            g += discount * t.reward
            discount *= self.cfg.gamma
        return g

    # -----------------------------------------------------------------
    # Convenience: compute reward from Metadata
    # -----------------------------------------------------------------

    def compute_from_metadata(
        self,
        action: PlannerAction,
        metadata: Metadata,
        is_terminal: bool,
        all_tests_passed: bool,
        budget_exhausted: bool,
        *,
        billed_worker: PlannerAction | None = None,
    ) -> float:
        """Thin wrapper around ``compute`` that unpacks a ``Metadata`` object.

        If *billed_worker* is omitted, tries to map ``metadata.worker_name``
        (``qwen`` / ``ornith``) to a ``CALL_*`` action.
        """
        if billed_worker is None:
            billed_worker = _worker_action_from_name(metadata.worker_name)
        return self.compute(
            action=action,
            tests_before=metadata.tests_before,
            tests_after=metadata.tests_after,
            compile_status=(
                CompileStatus(success=metadata.compile_success)
                if metadata.compile_success is not None
                else None
            ),
            tokens_used=metadata.tokens_used,
            latency_ms=metadata.latency_ms,
            is_terminal=is_terminal,
            all_tests_passed=all_tests_passed,
            budget_exhausted=budget_exhausted,
            billed_worker=billed_worker,
        )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _test_improvement(
        before: TestResults | None,
        after: TestResults | None,
    ) -> float:
        """Delta in pass rate; positive means improvement."""
        rate_before = before.pass_rate if before is not None else 0.0
        rate_after = after.pass_rate if after is not None else 0.0
        return rate_after - rate_before


def _worker_action_from_name(name: str) -> PlannerAction | None:
    """Map a worker pool name to a CALL_* action, if known."""
    if not name:
        return None
    key = name.strip().lower()
    mapping = {
        "qwen": PlannerAction.CALL_QWEN,
        "ornith": PlannerAction.CALL_ORNITH,
        "call_qwen": PlannerAction.CALL_QWEN,
        "call_ornith": PlannerAction.CALL_ORNITH,
        # Gemma disabled: do not map to a billable worker
    }
    return mapping.get(key)
