"""Tests for ReplayBuffer persistence."""

from pathlib import Path

from fugu.buffer.replay_buffer import ReplayBuffer
from fugu.core.actions import PlannerAction
from fugu.core.state import Metadata, PlannerState, Transition


def _tr(i: int) -> Transition:
    s = PlannerState(task_description=f"t{i}", step_number=i)
    return Transition(
        state=s,
        action=PlannerAction.CALL_QWEN,
        reward=float(i),
        next_state=None,
        done=True,
        metadata=Metadata(task_id=f"id{i}"),
    )


def test_add_sample_save_load(tmp_path: Path):
    buf = ReplayBuffer(capacity=100, storage_dir=str(tmp_path / "buf"))
    for i in range(5):
        buf.add(_tr(i))
    assert len(buf) == 5
    sample = buf.sample(2)
    assert len(sample) == 2

    path = tmp_path / "buffer.bin"
    buf.save(path)

    buf2 = ReplayBuffer(capacity=100, storage_dir=str(tmp_path / "buf2"))
    buf2.load(path)
    assert len(buf2) == 5
    all_t = buf2.get_all()
    assert all_t[0].metadata.task_id == "id0"
    assert all_t[-1].reward == 4.0
