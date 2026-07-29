"""Dual-mode LiveCodeBench public-test harness (stdin + functional)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fugu.datasets.livecodebench import LiveCodeBenchDataset, _PYTEST_HARNESS


def _write_workspace(root: Path, solution: str, cases: list, meta: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "solution.py").write_text(solution, encoding="utf-8")
    (root / "public_tests.json").write_text(
        json.dumps(cases), encoding="utf-8"
    )
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (root / "test_solution.py").write_text(_PYTEST_HARNESS, encoding="utf-8")


def test_stdin_harness_pass(tmp_path: Path):
    sol = "import sys\ndata=sys.stdin.read().split()\nprint(int(data[0])+int(data[1]))\n"
    cases = [{"input": "1 2\n", "output": "3", "testtype": "stdin"}]
    _write_workspace(tmp_path, sol, cases, {})
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "test_solution.py", "-q", "--tb=line"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_functional_harness_pass(tmp_path: Path):
    sol = (
        "from typing import List\n\n"
        "class Solution:\n"
        "    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:\n"
        "        n = len(nums)\n"
        "        return [len(set(nums[:i+1])) - len(set(nums[i+1:])) for i in range(n)]\n"
    )
    cases = [
        {
            "input": "[1, 2, 3, 4, 5]",
            "output": "[-3, -1, 1, 3, 5]",
            "testtype": "functional",
        }
    ]
    _write_workspace(
        tmp_path, sol, cases, {"func_name": "distinctDifferenceArray"}
    )
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "test_solution.py", "-q", "--tb=line"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_functional_multi_arg(tmp_path: Path):
    sol = (
        "from typing import List\n\n"
        "class Solution:\n"
        "    def maximumOr(self, nums: List[int], k: int) -> int:\n"
        "        return 30\n"
    )
    cases = [
        {
            "input": "[12, 9]\n1",
            "output": "30",
            "testtype": "functional",
        }
    ]
    _write_workspace(tmp_path, sol, cases, {"func_name": "maximumOr"})
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "test_solution.py", "-q", "--tb=line"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_row_to_task_includes_meta_func_name():
    row = {
        "question_id": "2727",
        "question_title": "t",
        "question_content": "c",
        "starter_code": (
            "class Solution:\n"
            "    def countSeniors(self, details: List[str]) -> int:\n"
            "        "
        ),
        "public_test_cases": json.dumps(
            [
                {
                    "input": '["a"]',
                    "output": "0",
                    "testtype": "functional",
                }
            ]
        ),
        "metadata": json.dumps({"func_name": "countSeniors"}),
        "platform": "leetcode",
        "contest_date": "2024-01-01",
    }
    task = LiveCodeBenchDataset._row_to_task(row)
    assert task is not None
    assert "meta.json" in task.metadata["workspace_files"]
    meta = json.loads(task.metadata["workspace_files"]["meta.json"])
    assert meta["func_name"] == "countSeniors"
    assert "from typing import List" in task.metadata["workspace_files"]["solution.py"]
