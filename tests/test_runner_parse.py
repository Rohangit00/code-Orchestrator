"""Tests for TestRunner parsing and command selection (issue #6)."""

from pathlib import Path

import pytest

from fugu.execution.runner import IsolationError, TestRunner


def test_parse_uses_last_summary_not_first_heading():
    runner = TestRunner()
    output = """
============================= test session starts ==============================
collected 4 items

test_a.py::test_one PASSED
test_a.py::test_two FAILED
test_a.py::test_three PASSED
test_a.py::test_four ERROR

=========================== short test summary info ============================
FAILED test_a.py::test_two
ERROR test_a.py::test_four
==================== 2 passed, 1 failed, 1 error in 0.12s ======================
"""
    results = runner._parse_pytest_output(output)
    assert results.passed == 2
    assert results.failed == 1
    assert results.errors == 1


def test_detect_command_no_x_flag(tmp_path: Path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    cmd = TestRunner._detect_test_command(tmp_path)
    assert "pytest" in cmd
    assert "-x" not in cmd.split()


def test_authoritative_command_not_overwritten(tmp_path: Path, monkeypatch):
    """task.test_command must be used as-is (no silent rewrite to pytest)."""
    runner = TestRunner(timeout_seconds=5, allow_host_execution=True)
    called: dict = {}

    def fake_run(*args, **kwargs):
        called["cmd"] = args[0] if args else kwargs.get("args")
        class R:
            stdout = "Ran 1 test in 0.001s\nOK\n"
            stderr = ""
            returncode = 0
        return R()

    monkeypatch.setattr("fugu.execution.runner.subprocess.run", fake_run)
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

    harness_cmd = "./tests/run_tests.sh --instance django__django-123"
    runner.run_tests(tmp_path, test_command=harness_cmd)
    assert called["cmd"] == harness_cmd


def test_specific_tests_ignored_for_non_pytest(tmp_path: Path, monkeypatch):
    runner = TestRunner(timeout_seconds=5, allow_host_execution=True)
    called: dict = {}

    def fake_run(*args, **kwargs):
        called["cmd"] = args[0]
        class R:
            stdout = "OK"
            stderr = ""
            returncode = 0
        return R()

    monkeypatch.setattr("fugu.execution.runner.subprocess.run", fake_run)
    runner.run_tests(
        tmp_path,
        test_command="./harness.sh",
        specific_tests=["tests/test_a.py::test_x"],
    )
    assert "test_x" not in called["cmd"]
    assert called["cmd"] == "./harness.sh"


def test_isolation_blocks_untrusted_remote():
    runner = TestRunner(allow_host_execution=False, isolation_mode="host")
    with pytest.raises(IsolationError):
        runner.run_tests(
            Path("/tmp"),
            test_command="true",
            repo_url="https://github.com/django/django.git",
        )


def test_isolation_allows_local_when_host_disabled(tmp_path: Path, monkeypatch):
    runner = TestRunner(allow_host_execution=False, isolation_mode="host")

    def fake_run(*args, **kwargs):
        class R:
            stdout = "Ran 0 tests in 0.0s\nOK\n"
            stderr = ""
            returncode = 0
        return R()

    monkeypatch.setattr("fugu.execution.runner.subprocess.run", fake_run)
    # Local path URL should not be treated as untrusted remote
    runner.run_tests(tmp_path, test_command="true", repo_url=str(tmp_path))
