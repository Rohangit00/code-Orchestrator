"""Tests for mock workers and pool dispatch."""

import pytest

from fugu.core.actions import PlannerAction
from fugu.workers.mock import MockWorker
from fugu.workers.pool import WorkerPool


@pytest.mark.asyncio
async def test_mock_worker_returns_patch():
    w = MockWorker(name="m", patch="diff", latency_ms=0)
    resp = await w.generate("p", "ctx", [], None)
    assert resp.success
    assert resp.patch == "diff"
    assert w.call_count == 1


def test_pool_mock_dispatch():
    pool = WorkerPool.mock()
    assert pool.has_worker(PlannerAction.CALL_QWEN)
    w = pool.get_worker(PlannerAction.CALL_QWEN)
    assert "qwen" in w.name or w.name.startswith("mock")


def test_pool_rejects_non_worker_action():
    pool = WorkerPool()
    with pytest.raises(ValueError):
        pool.register(PlannerAction.STOP, MockWorker())
