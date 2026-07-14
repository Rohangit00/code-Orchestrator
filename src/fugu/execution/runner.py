"""Test execution and compilation checking.

Runs tests inside a repository. The task-provided command (or a benchmark
native harness command) is authoritative. Pytest auto-detection is only a
fallback for local/mock fixtures.

When ``isolation_mode="docker"``, tests and compile checks run inside a
container (Fugu sandbox). This is **not** the official SWE-bench evaluation
harness; it only prevents untrusted test code from executing on the host.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path

from fugu.core.state import CompileStatus, TestResults

logger = logging.getLogger(__name__)

# Fallback only — not used when task.test_command is set.
_DEFAULT_PYTEST_CMD = "python -m pytest --tb=no -q"
_DEFAULT_UNITTEST_CMD = "python -m unittest discover -s . -v"


class IsolationError(RuntimeError):
    """Raised when untrusted host execution is not allowed."""


class TestRunner:
    """Executes tests in a repository and captures results.

    Parameters
    ----------
    timeout_seconds:
        Wall-clock timeout for a single test invocation.
    isolation_mode:
        ``"host"`` (default) or ``"docker"``. Docker is required for real
        third-party SWE-bench once ``allow_host_execution`` is false.
    allow_host_execution:
        When ``False``, refuse to run tests on the host for untrusted
        remote repositories. Mock / local paths may still use the host
        when this is ``True``.
    docker_image:
        Image used when ``isolation_mode="docker"``.
    docker_network:
        Docker network mode (default ``"none"`` — no outbound access).
    docker_memory / docker_cpus:
        Resource limits passed to ``docker run``.
    docker_workdir:
        Absolute path inside the container where the repo is mounted.
    docker_user:
        Optional ``uid:gid`` (or name) for ``--user``. Empty = image default.
    docker_extra_args:
        Extra argv tokens inserted into ``docker run`` (escape hatch).
    """

    def __init__(
        self,
        timeout_seconds: int = 300,
        isolation_mode: str = "host",
        allow_host_execution: bool = True,
        docker_image: str = "python:3.11-slim",
        docker_network: str = "none",
        docker_memory: str = "4g",
        docker_cpus: str = "2",
        docker_workdir: str = "/workspace",
        docker_user: str = "",
        docker_extra_args: list[str] | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        mode = isolation_mode.lower().strip()
        if mode not in {"host", "docker"}:
            raise ValueError(
                f"isolation_mode must be 'host' or 'docker', got {isolation_mode!r}"
            )
        self.isolation_mode = mode
        self.allow_host_execution = allow_host_execution
        self.docker_image = docker_image
        self.docker_network = docker_network
        self.docker_memory = docker_memory
        self.docker_cpus = docker_cpus
        self.docker_workdir = docker_workdir
        self.docker_user = docker_user
        self.docker_extra_args = list(docker_extra_args or [])

    # -- public API ----------------------------------------------------------

    def run_tests(
        self,
        repo_path: Path,
        test_command: str | None = None,
        specific_tests: list[str] | None = None,
        *,
        repo_url: str | None = None,
    ) -> TestResults:
        """Run tests and return structured results.

        Command selection priority:

        1. *test_command* from the task / native harness (authoritative).
        2. Pytest/unittest auto-detection as a **fallback only**.

        *specific_tests* are only appended when the effective command is
        pytest-based (never blindly appended to arbitrary harness commands).
        """
        self._enforce_isolation(repo_url=repo_url, repo_path=repo_path)

        used_fallback = False
        if test_command is None or not str(test_command).strip():
            test_command = self._detect_test_command(repo_path)
            used_fallback = True
            logger.info(
                "No task test_command; using auto-detected fallback: %s",
                test_command,
            )
        else:
            test_command = str(test_command).strip()

        if specific_tests:
            if self._is_pytest_command(test_command):
                test_command = test_command + " " + " ".join(specific_tests)
            else:
                logger.warning(
                    "Ignoring specific_tests for non-pytest command %r "
                    "(SWE-bench harnesses often use project-specific runners).",
                    test_command,
                )

        if used_fallback:
            logger.debug("Fallback test command: %s", test_command)
        else:
            logger.info("Running authoritative test command: %s", test_command)

        logger.info("Running tests: %s (cwd=%s)", test_command, repo_path)

        if self.isolation_mode == "docker":
            return self._run_tests_docker(repo_path, test_command)

        return self._run_tests_host(repo_path, test_command)

    def compile_check(self, repo_path: Path) -> CompileStatus:
        """Run ``python -m compileall -q .`` and return a :class:`CompileStatus`."""
        compile_cmd = "python -m compileall -q ."
        if self.isolation_mode == "docker":
            return self._compile_check_docker(repo_path, compile_cmd)
        return self._compile_check_host(repo_path, compile_cmd)

    def _compile_check_host(
        self, repo_path: Path, compile_cmd: str
    ) -> CompileStatus:
        try:
            proc = subprocess.run(
                ["python", "-m", "compileall", "-q", "."],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode == 0:
                return CompileStatus(success=True, error_message="")
            error_msg = (proc.stdout + "\n" + proc.stderr).strip()
            return CompileStatus(success=False, error_message=error_msg)
        except subprocess.TimeoutExpired:
            return CompileStatus(
                success=False,
                error_message="Compile check timed out after 120s",
            )
        except OSError as exc:
            return CompileStatus(success=False, error_message=str(exc))

    def _compile_check_docker(
        self, repo_path: Path, compile_cmd: str
    ) -> CompileStatus:
        container = self._docker_container_name("compile")
        argv = self.build_docker_run_argv(
            repo_path, compile_cmd, container_name=container
        )
        start = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=min(120, self.timeout_seconds),
            )
        except subprocess.TimeoutExpired:
            self._docker_kill(container)
            return CompileStatus(
                success=False,
                error_message="Compile check timed out (docker)",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Docker binary not found. Install Docker and ensure `docker` "
                "is on PATH, or set isolation_mode=host for trusted local repos."
            ) from exc
        except OSError as exc:
            return CompileStatus(success=False, error_message=str(exc))

        duration_ms = (time.monotonic() - start) * 1000
        combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        logger.debug(
            "Docker compile finished in %.0f ms rc=%s", duration_ms, proc.returncode
        )
        if proc.returncode == 0:
            return CompileStatus(success=True, error_message="")
        return CompileStatus(success=False, error_message=combined)

    # -- isolation -----------------------------------------------------------

    def _enforce_isolation(
        self,
        *,
        repo_url: str | None,
        repo_path: Path,
    ) -> None:
        """Gate untrusted third-party execution on the host."""
        if self.isolation_mode == "docker":
            return
        if self.allow_host_execution:
            return
        if not self._is_untrusted_remote(repo_url, repo_path):
            return
        raise IsolationError(
            "Host execution of untrusted third-party repository tests is "
            "disabled. Set env.allow_host_execution=true only for trusted "
            "local/mock workloads, or set isolation_mode=docker for real "
            "SWE-bench collection (Fugu container sandbox)."
        )

    @staticmethod
    def _is_untrusted_remote(repo_url: str | None, repo_path: Path) -> bool:
        if repo_url:
            url = repo_url.strip().lower()
            if url.startswith("file:") or url.startswith("/"):
                return False
            if url.startswith("http://") or url.startswith("https://") or url.startswith(
                "git@"
            ):
                return True
        # Local path-only fixtures are trusted for mock development.
        return False

    # -- docker executor -----------------------------------------------------

    @staticmethod
    def _docker_container_name(kind: str = "test") -> str:
        return f"fugu-{kind}-{uuid.uuid4().hex[:12]}"

    def build_docker_run_argv(
        self,
        repo_path: Path,
        command: str,
        *,
        container_name: str | None = None,
    ) -> list[str]:
        """Build ``docker run`` argv (public for unit tests)."""
        host_path = str(Path(repo_path).resolve())
        workdir = self.docker_workdir or "/workspace"
        name = container_name or self._docker_container_name("test")

        argv: list[str] = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            self.docker_network or "none",
            "--memory",
            self.docker_memory or "4g",
            "--cpus",
            self.docker_cpus or "2",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "-v",
            f"{host_path}:{workdir}:rw",
            "-w",
            workdir,
        ]
        if self.docker_user:
            argv.extend(["--user", self.docker_user])
        if self.docker_extra_args:
            argv.extend(self.docker_extra_args)
        argv.append(self.docker_image)
        # Shell form so task harness commands (pipes, env, etc.) work as on host.
        argv.extend(["sh", "-lc", command])
        return argv

    def _docker_kill(self, container_name: str) -> None:
        try:
            subprocess.run(
                ["docker", "kill", container_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            logger.debug("docker kill %s failed or not needed", container_name)

    def _run_tests_docker(self, repo_path: Path, test_command: str) -> TestResults:
        container = self._docker_container_name("test")
        argv = self.build_docker_run_argv(
            repo_path, test_command, container_name=container
        )
        logger.info(
            "Running tests in docker: image=%s cmd=%s mount=%s",
            self.docker_image,
            test_command,
            Path(repo_path).resolve(),
        )
        logger.debug("docker argv: %s", shlex.join(argv))

        start = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            self._docker_kill(container)
            duration_ms = (time.monotonic() - start) * 1000
            output = (exc.stdout or "") + "\n" + (exc.stderr or "")
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            logger.warning(
                "Docker tests timed out after %ds", self.timeout_seconds
            )
            return TestResults(
                passed=0,
                failed=0,
                errors=0,
                output=f"TIMEOUT after {self.timeout_seconds}s (docker)\n{output}",
                duration_ms=duration_ms,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Docker binary not found. Install Docker and ensure `docker` "
                "is on PATH when isolation_mode=docker."
            ) from exc

        duration_ms = (time.monotonic() - start) * 1000
        return self._results_from_output(test_command, combined_output, duration_ms)

    def _run_tests_host(self, repo_path: Path, test_command: str) -> TestResults:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                test_command,
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            duration_ms = (time.monotonic() - start) * 1000
            output = (exc.stdout or "") + "\n" + (exc.stderr or "")
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            logger.warning("Tests timed out after %ds", self.timeout_seconds)
            return TestResults(
                passed=0,
                failed=0,
                errors=0,
                output=f"TIMEOUT after {self.timeout_seconds}s\n{output}",
                duration_ms=duration_ms,
            )

        duration_ms = (time.monotonic() - start) * 1000
        return self._results_from_output(test_command, combined_output, duration_ms)

    def _results_from_output(
        self,
        test_command: str,
        combined_output: str,
        duration_ms: float,
    ) -> TestResults:
        if self._is_pytest_command(test_command) or "====" in combined_output:
            results = self._parse_pytest_output(combined_output)
        else:
            results = self._parse_unittest_output(combined_output)

        results.duration_ms = duration_ms
        results.output = combined_output
        return results

    # -- parsers -------------------------------------------------------------

    def _parse_pytest_output(self, output: str) -> TestResults:
        """Parse pytest summary for passed/failed/error counts.

        Uses the **last** summary-like block that contains result counts,
        not the first ``====`` heading.
        """
        passed = failed = errors = 0

        summary_text = self._last_pytest_summary(output)
        if summary_text:
            for m in re.finditer(
                r"(\d+)\s+(passed|failed|errors?|error)\b",
                summary_text,
                flags=re.IGNORECASE,
            ):
                count = int(m.group(1))
                category = m.group(2).lower()
                if category == "passed":
                    passed = count
                elif category == "failed":
                    failed = count
                elif category in {"error", "errors"}:
                    errors = count

        test_names_passed: list[str] = []
        test_names_failed: list[str] = []
        for m in re.finditer(
            r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR)\s*$",
            output,
            re.MULTILINE,
        ):
            name, status = m.group(1), m.group(2)
            if status == "PASSED":
                test_names_passed.append(name)
            else:
                test_names_failed.append(name)

        return TestResults(
            passed=passed,
            failed=failed,
            errors=errors,
            output="",
            test_names_passed=test_names_passed,
            test_names_failed=test_names_failed,
        )

    @staticmethod
    def _last_pytest_summary(output: str) -> str | None:
        """Return text of the last pytest result summary block, if any."""
        # Prefer lines like: ===== 3 passed, 1 failed in 1.23s =====
        candidates = list(
            re.finditer(r"=+\s*([^=\n]*?\d+\s+(?:passed|failed|errors?|error)[^=\n]*?)\s*=+", output, re.IGNORECASE)
        )
        if candidates:
            return candidates[-1].group(1)

        # Fallback: last line containing result counts without requiring ===
        lines = [
            ln
            for ln in output.splitlines()
            if re.search(r"\d+\s+(passed|failed|errors?|error)\b", ln, re.I)
        ]
        if lines:
            return lines[-1]
        return None

    def _parse_unittest_output(self, output: str) -> TestResults:
        """Parse unittest output for totals and failures/errors."""
        passed = failed = errors = 0
        total = 0

        ran_match = re.search(r"Ran\s+(\d+)\s+test", output)
        if ran_match:
            total = int(ran_match.group(1))

        fail_match = re.search(r"FAILED\s*\(([^)]+)\)", output)
        if fail_match:
            detail = fail_match.group(1)
            f_match = re.search(r"failures=(\d+)", detail)
            e_match = re.search(r"errors=(\d+)", detail)
            if f_match:
                failed = int(f_match.group(1))
            if e_match:
                errors = int(e_match.group(1))
            passed = max(0, total - failed - errors)
        elif re.search(r"\bOK\b", output) and total > 0:
            passed = total

        test_names_passed: list[str] = []
        test_names_failed: list[str] = []
        for m in re.finditer(
            r"^(test\S+)\s+\(([^)]+)\)\s+\.\.\.\s+(ok|FAIL|ERROR)\s*$",
            output,
            re.MULTILINE,
        ):
            test_name = f"{m.group(2)}.{m.group(1)}"
            status = m.group(3)
            if status == "ok":
                test_names_passed.append(test_name)
            else:
                test_names_failed.append(test_name)

        return TestResults(
            passed=passed,
            failed=failed,
            errors=errors,
            output="",
            test_names_passed=test_names_passed,
            test_names_failed=test_names_failed,
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _is_pytest_command(test_command: str) -> bool:
        cmd = test_command.lower()
        return "pytest" in cmd

    @staticmethod
    def _detect_test_command(repo_path: Path) -> str:
        """Auto-detect a fallback test command (never includes ``-x``)."""
        if (repo_path / "pytest.ini").exists():
            return _DEFAULT_PYTEST_CMD
        if (repo_path / "conftest.py").exists():
            return _DEFAULT_PYTEST_CMD

        pyproject = repo_path / "pyproject.toml"
        if pyproject.exists():
            try:
                text = pyproject.read_text(encoding="utf-8")
                if "[tool.pytest" in text:
                    return _DEFAULT_PYTEST_CMD
            except OSError:
                pass

        setup_cfg = repo_path / "setup.cfg"
        if setup_cfg.exists():
            try:
                text = setup_cfg.read_text(encoding="utf-8")
                if "[tool:pytest]" in text:
                    return _DEFAULT_PYTEST_CMD
            except OSError:
                pass

        test_files = list(repo_path.rglob("test_*.py"))
        if test_files:
            return _DEFAULT_PYTEST_CMD

        return _DEFAULT_UNITTEST_CMD
