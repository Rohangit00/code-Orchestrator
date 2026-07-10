"""Episode-level filtering for behavioural cloning datasets.

Light quality controls before SFT: drop or keep whole episodes based on
return and solve status so the planner is not trained only on failed
schedules.
"""

from __future__ import annotations

import logging
from typing import Sequence

from fugu.core.reward import RewardCalculator
from fugu.core.state import Transition

logger = logging.getLogger(__name__)


def group_episodes(transitions: Sequence[Transition]) -> list[list[Transition]]:
    """Split a flat transition list into episodes ending at ``done=True``."""
    episodes: list[list[Transition]] = []
    current: list[Transition] = []
    for t in transitions:
        current.append(t)
        if t.done:
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return episodes


def episode_return(
    episode: Sequence[Transition],
    *,
    gamma: float | None = None,
) -> float:
    """Sum (or discounted sum) of step rewards in an episode."""
    if gamma is None:
        return float(sum(t.reward for t in episode))
    calc = RewardCalculator()
    calc.cfg.gamma = gamma
    return calc.compute_episode_reward(list(episode))


def episode_solved(episode: Sequence[Transition]) -> bool:
    """True if the episode ends with all evaluated tests passing."""
    if not episode:
        return False
    last = episode[-1]
    after = last.metadata.tests_after
    if after is None:
        return False
    return after.total > 0 and after.failed == 0 and after.errors == 0


def filter_transitions(
    transitions: Sequence[Transition],
    *,
    min_return: float | None = 0.0,
    require_solved: bool = False,
    gamma: float | None = None,
) -> list[Transition]:
    """Keep transitions from episodes that meet quality criteria.

    Parameters
    ----------
    transitions:
        Flat list of transitions (one or more episodes).
    min_return:
        Minimum undiscounted (or discounted if *gamma* set) episode return.
        ``None`` disables the return filter.
    require_solved:
        If True, keep only episodes whose final tests all pass.
    gamma:
        Optional discount for return filtering.
    """
    kept: list[Transition] = []
    dropped = 0
    for ep in group_episodes(transitions):
        ret = episode_return(ep, gamma=gamma)
        solved = episode_solved(ep)
        if require_solved and not solved:
            dropped += 1
            continue
        if min_return is not None and ret < min_return:
            dropped += 1
            continue
        kept.extend(ep)

    logger.info(
        "Filtered transitions: kept=%d dropped_episodes=%d (min_return=%s, "
        "require_solved=%s)",
        len(kept),
        dropped,
        min_return,
        require_solved,
    )
    return kept
