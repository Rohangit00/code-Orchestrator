"""Shared pytest fixtures for Fugu tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fugu.core.actions import PlannerAction
from fugu.core.state import (
    CompileStatus,
    Metadata,
    PlannerState,
    TestResults,
    Transition,
)
from fugu.execution.runner import TestRunner

# Avoid pytest collecting classes whose names start with "Test".
TestResults.__test__ = False  # type: ignore[attr-defined]
TestRunner.__test__ = False  # type: ignore[attr-defined]


@pytest.fixture
def sample_state() -> PlannerState:
    return PlannerState(
        task_description="Fix the bug in foo()",
        repo_context="foo.py\nbar.py",
        history=[],
        test_results=TestResults(passed=1, failed=1, errors=0),
        compile_status=CompileStatus(success=True),
        current_patch="",
        step_number=0,
        max_steps=10,
        remaining_budget=1.0,
    )


@pytest.fixture
def sample_transition(sample_state: PlannerState) -> Transition:
    return Transition(
        state=sample_state,
        action=PlannerAction.CALL_QWEN,
        reward=0.5,
        next_state=sample_state,
        done=False,
        metadata=Metadata(task_id="t1", episode_step=1),
    )


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a tiny git repo with a failing test (pytest)."""
    repo = tmp_path / "toy_repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "test_mod.py").write_text(
        "from mod import add\n\n"
        "def test_add_ok():\n"
        "    assert add(1, 1) == 2\n\n"
        "def test_add_fail():\n"
        "    assert add(1, 1) == 3\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    # HEAD commit hash for local clone-by-path style usage
    return repo
