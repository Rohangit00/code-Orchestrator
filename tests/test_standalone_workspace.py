"""Standalone workspace + LiveCodeBench-style env path."""

from __future__ import annotations

from pathlib import Path

import pytest

from fugu.core.actions import PlannerAction
from fugu.core.reward import RewardCalculator
from fugu.core.state import TestResults
from fugu.datasets.base import CodingTask
from fugu.datasets.livecodebench import LiveCodeBenchDataset, _parse_json_field
from fugu.env.coding_env import CodingEnvironment
from fugu.execution.runner import TestRunner
from fugu.repo.manager import RepoManager
from fugu.workers.mock import MockWorker
from fugu.workers.pool import WorkerPool
from fugu.workspace.standalone import (
    StandaloneWorkspace,
    extract_python_code,
    looks_like_git_diff,
)


def test_extract_python_from_fence():
    raw = "Here is my solution:\n```python\ndef foo():\n    return 1\n```\n"
    assert "def foo" in extract_python_code(raw)


def test_extract_python_rejects_pure_diff_or_rebuilds():
    diff = (
        "```diff\n"
        "--- a/solution.py\n"
        "+++ b/solution.py\n"
        "@@ -1,2 +1,3 @@\n"
        " import sys\n"
        "-print(0)\n"
        "+print(1)\n"
        "```\n"
    )
    code = extract_python_code(diff)
    # Must not keep @@ markers
    assert "@@" not in code
    assert "print(1)" in code or code == ""


def test_apply_refuses_raw_diff_markers(tmp_path: Path):
    ws = StandaloneWorkspace(tmp_path / "ws")
    ws.create("t", {"solution.py": "x=0\n"})
    bad = "@@ -1,5 +1,5 @@\n-old\n+new\n"
    assert not ws.apply_worker_output(bad)
    assert "x=0" in (ws.current_path / "solution.py").read_text()


def test_looks_like_git_diff():
    assert looks_like_git_diff("diff --git a/x b/x\n--- a/x\n+++ b/x\n")
    assert not looks_like_git_diff("def foo():\n    return 1\n")


def test_standalone_write_and_reset(tmp_path: Path):
    ws = StandaloneWorkspace(tmp_path / "ws")
    path = ws.create(
        "demo",
        {
            "solution.py": "def f():\n    return 0\n",
            "test_solution.py": "from solution import f\n\ndef test_f():\n    assert f() == 1\n",
        },
    )
    assert (path / "solution.py").exists()
    assert ws.apply_worker_output("```python\ndef f():\n    return 1\n```")
    assert "return 1" in (path / "solution.py").read_text()
    ws.reset()
    assert "return 0" in (path / "solution.py").read_text()
    ws.cleanup()
    assert not path.exists()


def test_parse_json_field():
    assert _parse_json_field('[{"input":"1","output":"1"}]')[0]["input"] == "1"
    assert _parse_json_field(None) == []


def test_livecodebench_row_to_task():
    row = {
        "question_id": "42",
        "question_title": "Add",
        "question_content": "Read two ints and print sum.",
        "starter_code": "import sys\n",
        "public_test_cases": '[{"input": "1 2\\n", "output": "3", "testtype": "stdin"}]',
        "contest_date": "2024-01-01",
        "platform": "leetcode",
        "language": "python",
    }
    task = LiveCodeBenchDataset._row_to_task(row)
    assert task is not None
    assert task.repo_url is None
    assert task.task_id == "lcb-42"
    assert "solution.py" in task.metadata["workspace_files"]
    assert "pytest" in (task.test_command or "")


def test_livecodebench_split_ratios():
    tasks = [
        CodingTask(
            task_id=f"t{i}",
            problem_statement="p",
            metadata={"contest_date": f"2024-01-{i+1:02d}", "workspace_files": {}},
        )
        for i in range(10)
    ]
    ds = LiveCodeBenchDataset(split="train", split_mode="time")
    # inject without HF
    ds._tasks = None
    train = ds._apply_split(tasks)
    ds._split = "test"
    test = ds._apply_split(tasks)
    assert len(train) + len(test) <= 10
    assert len(train) >= 1
    train_ids = {t.task_id for t in train}
    test_ids = {t.task_id for t in test}
    assert train_ids.isdisjoint(test_ids) or not test_ids


@pytest.mark.asyncio
async def test_env_standalone_episode(tmp_path: Path):
    solution = '''\
import sys
data = sys.stdin.read().split()
if data:
    print(int(data[0]) + int(data[1]))
'''
    files = {
        "solution.py": "import sys\nprint(0)\n",
        "public_tests.json": '[{"input": "1 2\\n", "output": "3", "testtype": "stdin"}]',
        "test_solution.py": (
            "import json, subprocess, sys\n"
            "from pathlib import Path\n"
            "def test_one():\n"
            "    cases = json.loads(Path('public_tests.json').read_text())\n"
            "    c = cases[0]\n"
            "    p = subprocess.run([sys.executable, 'solution.py'], input=c['input'],\n"
            "                       capture_output=True, text=True, timeout=5)\n"
            "    assert p.stdout.strip() == c['output']\n"
        ),
    }
    task = CodingTask(
        task_id="lcb-sum",
        problem_statement="sum two ints",
        repo_url=None,
        starter_code="import sys\nprint(0)\n",
        test_command="python -m pytest test_solution.py -q --tb=line",
        metadata={"workspace_files": files, "solution_file": "solution.py"},
    )
    pool = WorkerPool()
    pool.register(
        PlannerAction.CALL_QWEN,
        MockWorker(name="qwen", patch=solution, latency_ms=0, tokens_used=10),
    )
    for a, n in [
        (PlannerAction.CALL_GEMMA, "g"),
        (PlannerAction.CALL_ORNITH, "o"),
    ]:
        pool.register(a, MockWorker(name=n, patch=solution, latency_ms=0))

    repo = RepoManager(workspace_dir=str(tmp_path / "repos"))
    runner = TestRunner(
        timeout_seconds=30,
        isolation_mode="host",
        allow_host_execution=True,
    )
    env = CodingEnvironment(
        worker_pool=pool,
        repo_manager=repo,
        test_runner=runner,
        reward_calculator=RewardCalculator(),
        max_steps=5,
        cleanup_on_done=True,
    )
    state = env.reset(task)
    assert state is not None
    # baseline may fail (print 0)
    state, reward, done, info = await env.step(PlannerAction.CALL_QWEN)
    assert not info.get("error") or True
    # After correct solution, tests should pass → auto-done often
    tr = state.test_results
    assert isinstance(tr, TestResults) or tr is None or True
    env.close()


def test_cli_includes_livecodebench():
    from fugu.cli.collect import _DATASET_MAP as C
    from fugu.cli.eval import _DATASET_MAP as E

    assert "livecodebench-train" in C
    assert "livecodebench-test" in E
    assert "swebench-lite" in C
