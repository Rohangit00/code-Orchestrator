"""Tests for Docker isolation executor (issue #12)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fugu.execution.runner import IsolationError, TestRunner


def test_build_docker_run_argv_includes_security_and_mount(tmp_path: Path):
    runner = TestRunner(
        isolation_mode="docker",
        docker_image="python:3.11-slim",
        docker_network="none",
        docker_memory="2g",
        docker_cpus="1",
        docker_workdir="/workspace",
        docker_user="1000:1000",
        docker_extra_args=["--read-only"],
    )
    argv = runner.build_docker_run_argv(
        tmp_path, "python -m pytest -q", container_name="fugu-test-abc"
    )

    assert argv[0:3] == ["docker", "run", "--rm"]
    assert "--name" in argv and "fugu-test-abc" in argv
    assert "--network" in argv and "none" in argv
    assert "--memory" in argv and "2g" in argv
    assert "--cpus" in argv and "1" in argv
    assert "--cap-drop" in argv and "ALL" in argv
    assert "--security-opt" in argv and "no-new-privileges" in argv
    assert "--user" in argv and "1000:1000" in argv
    assert "--read-only" in argv
    assert "python:3.11-slim" in argv

    mount = f"{tmp_path.resolve()}:/workspace:rw"
    assert mount in argv
    assert "-w" in argv and "/workspace" in argv
    # Inner command via shell
    assert argv[-3:] == ["sh", "-lc", "python -m pytest -q"]


def test_docker_run_tests_uses_docker_and_parses(tmp_path: Path, monkeypatch):
    runner = TestRunner(
        isolation_mode="docker",
        allow_host_execution=False,
        timeout_seconds=30,
        docker_image="python:3.11-slim",
    )
    called: dict = {}

    def fake_run(argv, **kwargs):
        called["argv"] = argv
        # Must not use host shell for docker binary
        assert kwargs.get("shell") in (None, False)
        out = (
            "==================== 2 passed, 1 failed in 0.05s ======================\n"
        )
        m = MagicMock()
        m.stdout = out
        m.stderr = ""
        m.returncode = 1
        return m

    monkeypatch.setattr("fugu.execution.runner.subprocess.run", fake_run)

    results = runner.run_tests(
        tmp_path,
        test_command="python -m pytest --tb=no -q",
        repo_url="https://github.com/django/django.git",
    )

    assert called["argv"][0] == "docker"
    assert "python:3.11-slim" in called["argv"]
    assert results.passed == 2
    assert results.failed == 1


def test_docker_mode_allows_untrusted_remote_without_host_gate(
    tmp_path: Path, monkeypatch
):
    """docker mode must not raise IsolationError for https remotes."""
    runner = TestRunner(isolation_mode="docker", allow_host_execution=False)

    def fake_run(argv, **kwargs):
        m = MagicMock()
        m.stdout = "Ran 0 tests in 0.0s\nOK\n"
        m.stderr = ""
        m.returncode = 0
        return m

    monkeypatch.setattr("fugu.execution.runner.subprocess.run", fake_run)
    runner.run_tests(
        tmp_path,
        test_command="true",
        repo_url="https://github.com/example/repo.git",
    )


def test_host_mode_still_blocks_untrusted():
    runner = TestRunner(isolation_mode="host", allow_host_execution=False)
    with pytest.raises(IsolationError):
        runner.run_tests(
            Path("/tmp"),
            test_command="true",
            repo_url="https://github.com/django/django.git",
        )


def test_compile_check_docker_uses_docker(tmp_path: Path, monkeypatch):
    runner = TestRunner(isolation_mode="docker", docker_image="python:3.11-slim")
    called: dict = {}

    def fake_run(argv, **kwargs):
        called["argv"] = argv
        m = MagicMock()
        m.stdout = ""
        m.stderr = ""
        m.returncode = 0
        return m

    monkeypatch.setattr("fugu.execution.runner.subprocess.run", fake_run)
    status = runner.compile_check(tmp_path)
    assert status.success
    assert called["argv"][0] == "docker"
    assert "python:3.11-slim" in called["argv"]
    # compile command present in sh -lc payload
    assert any("compileall" in str(x) for x in called["argv"])


def test_docker_missing_binary_raises_clear_error(tmp_path: Path, monkeypatch):
    runner = TestRunner(isolation_mode="docker")

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr("fugu.execution.runner.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="Docker binary not found"):
        runner.run_tests(tmp_path, test_command="true")
