"""Tests for light BC filtering (issue #11)."""

from fugu.core.actions import PlannerAction
from fugu.core.state import Metadata, PlannerState, TestResults, Transition
from fugu.training.filter import filter_transitions, group_episodes


def _ep(rewards: list[float], solved: bool) -> list[Transition]:
    out = []
    for i, r in enumerate(rewards):
        done = i == len(rewards) - 1
        after = (
            TestResults(passed=1, failed=0, errors=0)
            if solved and done
            else TestResults(passed=0, failed=1, errors=0)
        )
        out.append(
            Transition(
                state=PlannerState(step_number=i),
                action=PlannerAction.CALL_QWEN,
                reward=r,
                next_state=None,
                done=done,
                metadata=Metadata(tests_after=after),
            )
        )
    return out


def test_group_episodes():
    flat = _ep([1.0], False) + _ep([-1.0, 0.5], True)
    eps = group_episodes(flat)
    assert len(eps) == 2
    assert len(eps[0]) == 1
    assert len(eps[1]) == 2


def test_filter_min_return():
    flat = _ep([-2.0], False) + _ep([1.0], True)
    kept = filter_transitions(flat, min_return=0.0, require_solved=False)
    assert len(kept) == 1
    assert kept[0].reward == 1.0


def test_filter_require_solved():
    flat = _ep([5.0], False) + _ep([0.1], True)
    kept = filter_transitions(flat, min_return=None, require_solved=True)
    assert len(kept) == 1
    assert kept[0].metadata.tests_after.failed == 0
