"""Environment termination, transitions, and cleanup (issues #3,#4,#7,#14)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fugu.core.actions import PlannerAction
from fugu.core.reward import RewardCalculator
from fugu.core.state import CompileStatus, TestResults
from fugu.datasets.base import CodingTask
from fugu.env.coding_env import CodingEnvironment
from fugu.execution.runner import TestRunner
from fugu.repo.manager import RepoManager
from fugu.workers.base import WorkerResponse
from fugu.workers.mock import MockWorker
from fugu.workers.pool import WorkerPool


def _local_task(repo: Path) -> CodingTask:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    return CodingTask(
        task_id="local-toy",
        problem_statement="make tests pass",
        repo_url=str(repo),
        base_commit=head,
        test_command="python -m pytest --tb=no -q",
    )


@pytest.fixture
def local_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "test_mod.py").write_text(
        "from mod import VALUE\n\ndef test_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "i"], cwd=repo, check=True, capture_output=True)
    return repo


def _env_with_patch(workspace: Path, patch: str) -> tuple[CodingEnvironment, RepoManager]:
    pool = WorkerPool()
    for action, name in [
        (PlannerAction.CALL_QWEN, "qwen"),
        (PlannerAction.CALL_GEMMA, "gemma"),
        (PlannerAction.CALL_ORNITH, "ornith"),
    ]:
        pool.register(
            action,
            MockWorker(name=name, patch=patch, latency_ms=0, tokens_used=10),
        )
    repo_mgr = RepoManager(workspace_dir=str(workspace / "ws"), max_disk_mb=500)
    # Clone via file copy path: RepoManager expects git remote URL.
    # Use local path as remote (git supports it).
    runner = TestRunner(
        timeout_seconds=60,
        isolation_mode="host",
        allow_host_execution=True,
    )
    env = CodingEnvironment(
        worker_pool=pool,
        repo_manager=repo_mgr,
        test_runner=runner,
        reward_calculator=RewardCalculator(),
        max_steps=5,
        cleanup_on_done=True,
    )
    return env, repo_mgr


@pytest.mark.asyncio
async def test_worker_auto_terminates_when_tests_pass(tmp_path: Path, local_repo: Path):
    """Issue #7: solving on worker step ends episode without extra RUN_TESTS."""
    patch = "--- a/mod.py\n+++ b/mod.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
    env, repo_mgr = _env_with_patch(tmp_path, patch)
    task = _local_task(local_repo)
    task.repo_url = str(local_repo)

    # Isolate termination logic from real pytest/git-apply flakiness:
    # baseline fails; after worker patch application, tests fully pass.
    baseline = TestResults(passed=0, failed=1, errors=0, output="fail")
    solved = TestResults(passed=1, failed=0, errors=0, output="ok")
    call_count = {"n": 0}

    def fake_run_tests(repo_path, test_command=None, specific_tests=None, repo_url=None):
        call_count["n"] += 1
        return baseline if call_count["n"] == 1 else solved

    env._runner.run_tests = fake_run_tests  # type: ignore[method-assign]
    env._runner.compile_check = lambda p: CompileStatus(success=True)  # type: ignore
    env._repo.apply_patch = lambda p: True  # type: ignore
    env._repo.get_diff = lambda: patch  # type: ignore
    env._repo.get_changed_files = lambda: ["mod.py"]  # type: ignore

    try:
        state = env.reset(task)
        assert state is not None
        _next, _reward, done, info = await env.step(PlannerAction.CALL_QWEN)
        assert info["all_tests_passed"] is True
        assert done is True
        assert env.transitions[-1].done is True
        assert env.transitions[-1].action is PlannerAction.CALL_QWEN
        assert env.transitions[-1].metadata.task_id == "local-toy"
        # Terminal success should not require a separate STOP / RUN_TESTS.
        assert len(env.transitions) == 1
    finally:
        env.close()


@pytest.mark.asyncio
async def test_stop_goes_through_env(tmp_path: Path, local_repo: Path):
    """Issue #4: STOP is handled by the environment."""
    env, _ = _env_with_patch(tmp_path, "")
    task = _local_task(local_repo)
    task.repo_url = str(local_repo)
    try:
        env.reset(task)
        _, reward, done, info = await env.step(PlannerAction.STOP)
        assert done is True
        assert env.transitions[-1].action is PlannerAction.STOP
        assert "metadata" in info
    finally:
        env.close()


@pytest.mark.asyncio
async def test_cleanup_on_close(tmp_path: Path, local_repo: Path):
    """Issue #14 basic: close removes workspace repo."""
    env, repo_mgr = _env_with_patch(tmp_path, "")
    task = _local_task(local_repo)
    task.repo_url = str(local_repo)
    env.reset(task)
    assert repo_mgr.current_path is not None
    path = repo_mgr.current_path
    assert path.exists()
    env.close()
    assert repo_mgr.current_path is None
    assert not path.exists()


def test_reset_without_repo_url_uses_standalone(tmp_path: Path):
    """No repo_url → standalone workspace (LiveCodeBench / contest path)."""
    env, _ = _env_with_patch(tmp_path, "")
    task = CodingTask(
        task_id="no-repo",
        problem_statement="x",
        repo_url=None,
        metadata={
            "workspace_files": {
                "solution.py": "x = 1\n",
                "test_solution.py": "def test_ok():\n    assert True\n",
            }
        },
        test_command="python -m pytest test_solution.py -q",
    )
    state = env.reset(task)
    assert state is not None
    assert state.task_description == "x"
    env.close()
