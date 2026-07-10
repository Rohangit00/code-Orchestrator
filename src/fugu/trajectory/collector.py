"""Trajectory collector for generating behavioural-cloning data.

The :class:`TrajectoryCollector` drives a :class:`CodingEnvironment` using
a fixed :class:`BaseStrategy` to produce MDP transitions.  It supports
single-episode collection, dataset-wide sweeps, and multi-strategy runs
to build diverse training corpora.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from fugu.core.state import Transition
from fugu.trajectory.strategies import BaseStrategy

if TYPE_CHECKING:
    from fugu.buffer.replay_buffer import ReplayBuffer
    from fugu.datasets.base import BaseDataset, CodingTask
    from fugu.env.coding_env import CodingEnvironment

logger = logging.getLogger(__name__)

# Default safety limit to prevent infinite loops.
_DEFAULT_MAX_STEPS = 30


class TrajectoryCollector:
    """Collect MDP trajectories by driving a coding environment with a strategy.

    Parameters
    ----------
    env:
        The :class:`CodingEnvironment` to interact with.
    buffer:
        Optional :class:`ReplayBuffer` to store generated transitions
        incrementally during collection.
    max_steps:
        Per-episode safety limit on the number of environment steps.
    """

    def __init__(
        self,
        env: CodingEnvironment,
        buffer: ReplayBuffer | None = None,
        max_steps: int = _DEFAULT_MAX_STEPS,
    ) -> None:
        self.env = env
        self.buffer = buffer
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------

    async def collect_episode(
        self,
        task: CodingTask,
        strategy: BaseStrategy,
    ) -> list[Transition]:
        """Run a single episode and collect all transitions.

        Workflow:
            1. Reset the environment with the given task.
            2. Repeatedly call ``strategy.select_action(state, step)`` to pick
               the next action, then ``env.step(action)`` to execute it
               (including ``STOP`` so terminal reward logic runs in the env).
            3. Append the environment-produced transition
               (``env.transitions[-1]``), which already includes full metadata.
            4. Stop when the environment signals ``done`` or when
               :attr:`max_steps` is reached.
            5. If a :attr:`buffer` is attached, add all transitions.

        Args:
            task: The coding task for this episode.
            strategy: The strategy to drive action selection.

        Returns:
            Ordered list of transitions collected during the episode.
        """
        transitions: list[Transition] = []
        done = False
        step = 0
        t0 = time.monotonic()

        logger.info(
            "Starting episode: task=%s strategy=%s",
            getattr(task, "task_id", "?"),
            strategy.name,
        )

        try:
            state = self.env.reset(task)

            while not done and step < self.max_steps:
                action = strategy.select_action(state, step)

                # Always step the environment — including STOP — so terminal
                # rewards, history, and transition metadata stay authoritative.
                next_state, reward, done, _info = await self.env.step(action)

                # CodingEnvironment.step() already builds a complete Transition.
                if not self.env.transitions:
                    raise RuntimeError(
                        "Environment did not record a transition after step(); "
                        "collector cannot reconstruct transitions."
                    )
                transition = self.env.transitions[-1]
                transitions.append(transition)

                if next_state is not None:
                    state = next_state
                step += 1
        finally:
            # Honor cleanup_on_done even on exceptions / early exits.
            try:
                self.env.close()
            except Exception:
                logger.exception(
                    "Environment cleanup failed for task %s",
                    getattr(task, "task_id", "?"),
                )

        elapsed = time.monotonic() - t0

        logger.info(
            "Episode complete: task=%s strategy=%s steps=%d transitions=%d "
            "elapsed=%.2fs",
            getattr(task, "task_id", "?"),
            strategy.name,
            step,
            len(transitions),
            elapsed,
        )

        # Persist to replay buffer.
        if self.buffer is not None and transitions:
            self.buffer.add_episode(transitions)

        return transitions

    # ------------------------------------------------------------------
    # Dataset-level collection
    # ------------------------------------------------------------------

    async def collect_dataset(
        self,
        dataset: BaseDataset,
        strategy: BaseStrategy,
        max_tasks: int | None = None,
    ) -> list[list[Transition]]:
        """Collect episodes for all tasks in a dataset.

        Args:
            dataset: The dataset providing :class:`CodingTask` instances.
                Must be iterable (supports ``__iter__`` or ``__getitem__``).
            strategy: The strategy to use for every episode.
            max_tasks: Optional cap on the number of tasks to process.
                ``None`` means process all tasks.

        Returns:
            A list of episode transition lists, one per task.
        """
        all_episodes: list[list[Transition]] = []
        task_count = 0

        tasks = list(dataset)  # type: ignore[arg-type]
        if max_tasks is not None:
            tasks = tasks[:max_tasks]

        total = len(tasks)

        for i, task in enumerate(tasks):
            logger.info(
                "[%d/%d] Collecting episode for task %s with strategy %s",
                i + 1,
                total,
                getattr(task, "task_id", "?"),
                strategy.name,
            )
            try:
                episode = await self.collect_episode(task, strategy)
                all_episodes.append(episode)
                task_count += 1
            except Exception:
                logger.exception(
                    "Failed to collect episode for task %s — skipping.",
                    getattr(task, "task_id", "?"),
                )
                continue

        total_transitions = sum(len(ep) for ep in all_episodes)
        logger.info(
            "Dataset collection complete: %d/%d tasks succeeded, "
            "%d total transitions, strategy=%s",
            task_count,
            total,
            total_transitions,
            strategy.name,
        )

        return all_episodes

    # ------------------------------------------------------------------
    # Multi-strategy collection
    # ------------------------------------------------------------------

    async def collect_multi_strategy(
        self,
        dataset: BaseDataset,
        strategies: list[BaseStrategy],
        max_tasks: int | None = None,
    ) -> list[list[Transition]]:
        """Run every strategy on every task to build a diverse trajectory set.

        For each task in *dataset* (up to *max_tasks*), each strategy in
        *strategies* is applied in order.  This produces
        ``min(len(dataset), max_tasks) × len(strategies)`` episodes.

        Args:
            dataset: The dataset providing tasks.
            strategies: List of strategies to apply.
            max_tasks: Optional cap on per-strategy task count.

        Returns:
            Flat list of all episode transition lists across all
            strategy–task combinations.
        """
        all_episodes: list[list[Transition]] = []

        tasks = list(dataset)  # type: ignore[arg-type]
        if max_tasks is not None:
            tasks = tasks[:max_tasks]

        total_combos = len(tasks) * len(strategies)
        combo_idx = 0

        for strategy in strategies:
            logger.info(
                "Starting multi-strategy pass: strategy=%s tasks=%d",
                strategy.name,
                len(tasks),
            )
            for task in tasks:
                combo_idx += 1
                logger.info(
                    "[%d/%d] strategy=%s task=%s",
                    combo_idx,
                    total_combos,
                    strategy.name,
                    getattr(task, "task_id", "?"),
                )
                try:
                    episode = await self.collect_episode(task, strategy)
                    all_episodes.append(episode)
                except Exception:
                    logger.exception(
                        "Failed: strategy=%s task=%s — skipping.",
                        strategy.name,
                        getattr(task, "task_id", "?"),
                    )
                    continue

        total_transitions = sum(len(ep) for ep in all_episodes)
        logger.info(
            "Multi-strategy collection complete: %d episodes, "
            "%d total transitions across %d strategies",
            len(all_episodes),
            total_transitions,
            len(strategies),
        )

        return all_episodes
