"""Gymnasium-style coding environment for the Fugu orchestrator MDP.

Supports:

* **Repo tasks** (SWE-bench): clone, git patch, tests.
* **Standalone tasks** (LiveCodeBench / HumanEval-style): local workspace,
  write Python solution files, run pytest.

Typical usage::

    env = CodingEnvironment(pool, repo_mgr, runner)
    state = env.reset(task)
    while True:
        action = planner.select(state)
        state, reward, done, info = await env.step(action)
        if done:
            break
    env.close()
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fugu.core.actions import PlannerAction
from fugu.core.reward import RewardCalculator
from fugu.core.state import (
    CompileStatus,
    HistoryEntry,
    Metadata,
    PlannerState,
    TestResults,
    Transition,
)
from fugu.workers.pool import WorkerPool
from fugu.workspace.standalone import StandaloneWorkspace

if TYPE_CHECKING:
    from fugu.datasets.base import CodingTask
    from fugu.execution.runner import TestRunner
    from fugu.repo.manager import RepoManager

logger = logging.getLogger(__name__)


class CodingEnvironment:
    """Gymnasium-style environment for coding episodes.

    Parameters:
        worker_pool: Registry of worker LLMs keyed by :class:`PlannerAction`.
        repo_manager: Handles cloning, patching, and resetting git repos.
        test_runner: Executes test suites and compile checks.
        reward_calculator: Optional custom reward calculator; a default is
            created if ``None``.
        max_steps: Maximum number of steps before the episode terminates.
        standalone_workspace: Optional manager for non-git tasks; created
            under ``repo_manager.workspace_dir / "standalone"`` if omitted.
    """

    def __init__(
        self,
        worker_pool: WorkerPool,
        repo_manager: RepoManager,
        test_runner: TestRunner,
        reward_calculator: RewardCalculator | None = None,
        max_steps: int = 10,
        cleanup_on_done: bool = True,
        standalone_workspace: StandaloneWorkspace | None = None,
    ) -> None:
        self._pool = worker_pool
        self._repo = repo_manager
        self._runner = test_runner
        self._reward = reward_calculator or RewardCalculator()
        self._max_steps = max_steps
        self._cleanup_on_done = cleanup_on_done
        self._standalone = standalone_workspace or StandaloneWorkspace(
            Path(repo_manager.workspace_dir) / "standalone"
        )

        # Episode state — populated by reset()
        self._task: CodingTask | None = None
        self._state: PlannerState | None = None
        self._transitions: list[Transition] = []
        self._done: bool = True
        self._mode: str = "repo"  # "repo" | "standalone"

        # Track last worker action for RETRY
        self._last_worker_action: PlannerAction | None = None
        self._last_worker_error: str = ""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _work_path(self) -> Path | None:
        if self._mode == "standalone":
            return self._standalone.current_path
        return self._repo.current_path

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, task: CodingTask) -> PlannerState:
        """Start a new episode (repo-backed or standalone workspace)."""
        self._task = task
        self._transitions = []
        self._done = False
        self._last_worker_action = None
        self._last_worker_error = ""

        # Prefer standalone when no git remote (LCB / HumanEval-style).
        if not task.repo_url:
            return self._reset_standalone(task)
        return self._reset_repo(task)

    def _reset_repo(self, task: CodingTask) -> PlannerState:
        self._mode = "repo"
        self._standalone.cleanup()

        repo_path = self._repo.clone(task.repo_url, task.base_commit)  # type: ignore[arg-type]

        if task.test_patch and repo_path:
            applied = self._repo.apply_test_patch(task.test_patch)
            if not applied:
                logger.warning(
                    "Failed to apply test patch for task %s", task.task_id
                )

        baseline_tests: TestResults | None = None
        baseline_compile: CompileStatus | None = None
        if repo_path:
            baseline_compile = self._runner.compile_check(repo_path)
            baseline_tests = self._runner.run_tests(
                repo_path,
                test_command=task.test_command,
                repo_url=task.repo_url,
            )

        repo_context = ""
        if repo_path:
            try:
                from fugu.repo.context import RepoContext

                ctx = RepoContext(repo_path)
                error_output = (
                    baseline_tests.output if baseline_tests else ""
                )
                repo_context = ctx.get_summary(
                    error_output=error_output,
                    task_description=task.problem_statement,
                )
            except Exception as exc:
                logger.debug("Could not build repo context: %s", exc)
                repo_context = ""

        return self._finish_reset(task, baseline_tests, baseline_compile, repo_context)

    def _reset_standalone(self, task: CodingTask) -> PlannerState:
        self._mode = "standalone"
        try:
            self._repo.cleanup()
        except Exception:
            pass

        files = dict(task.metadata.get("workspace_files") or {})
        if not files:
            # Minimal fallback: starter + empty tests
            starter = task.starter_code or "# Write your solution\n"
            files = {
                "solution.py": starter,
                "test_solution.py": (
                    "def test_placeholder():\n"
                    "    # No public tests provided on task\n"
                    "    assert True\n"
                ),
            }
        solution_name = str(task.metadata.get("solution_file") or "solution.py")
        path = self._standalone.create(
            task.task_id, files, solution_name=solution_name
        )

        baseline_compile = self._runner.compile_check(path)
        # Standalone tasks are local paths — not untrusted remotes.
        baseline_tests = self._runner.run_tests(
            path,
            test_command=task.test_command or "python -m pytest -q --tb=no",
            repo_url=str(path),
        )
        repo_context = self._standalone.file_tree_summary()
        if task.starter_code:
            repo_context += f"\n\n[Starter]\n{task.starter_code[:1500]}"

        return self._finish_reset(task, baseline_tests, baseline_compile, repo_context)

    def _finish_reset(
        self,
        task: CodingTask,
        baseline_tests: TestResults | None,
        baseline_compile: CompileStatus | None,
        repo_context: str,
    ) -> PlannerState:
        self._state = PlannerState(
            task_description=task.problem_statement,
            repo_context=repo_context,
            history=[],
            test_results=baseline_tests,
            compile_status=baseline_compile,
            current_patch="",
            step_number=0,
            max_steps=self._max_steps,
            remaining_budget=1.0,
        )
        logger.info(
            "Episode reset for task %s mode=%s — baseline tests: %s",
            task.task_id,
            self._mode,
            baseline_tests.summary() if baseline_tests else "none",
        )
        return copy.deepcopy(self._state)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    async def step(
        self, action: PlannerAction
    ) -> tuple[PlannerState, float, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: The planner's chosen action.

        Returns:
            A 4-tuple ``(next_state, reward, done, info)`` following the
            Gymnasium convention.

        Raises:
            RuntimeError: If the environment has not been reset or the
                episode is already done.
        """
        if self._state is None or self._done:
            raise RuntimeError(
                "Environment must be reset before stepping. "
                "Call env.reset(task) first."
            )

        assert self._task is not None

        prev_state = copy.deepcopy(self._state)
        tests_before = copy.deepcopy(self._state.test_results)
        step_start = time.monotonic()

        # Dispatch based on action type
        patch_applied = ""
        worker_name: str | None = None
        tokens_used = 0
        latency_ms = 0.0
        tests_after = tests_before
        compile_status = self._state.compile_status
        error_msg = ""
        # True when this step executed a fresh test evaluation.
        tests_were_evaluated = False

        if action.is_worker_call:
            # ----------------------------------------------------------
            # Worker call: CALL_QWEN, CALL_GEMMA, CALL_ORNITH
            # ----------------------------------------------------------
            worker = self._pool.get_worker(action)
            worker_name = worker.name

            # Build history dicts for the worker prompt
            history_dicts = self._build_worker_history()
            test_dict = self._test_results_to_dict(tests_before)

            response = await worker.generate(
                prompt=self._task.problem_statement,
                repository_context=self._state.repo_context,
                history=history_dicts,
                test_results=test_dict,
                code_format=self._worker_code_format(),
            )

            tokens_used = response.tokens_used
            latency_ms = response.latency_ms
            patch_applied = response.patch or response.raw_output

            if response.success and (response.patch or response.raw_output):
                applied, err = self._apply_worker_content(
                    response.patch or response.raw_output
                )
                if applied:
                    work = self._work_path()
                    assert work is not None
                    self._state.current_patch = (
                        self._repo.get_diff()
                        if self._mode == "repo"
                        else (response.patch or "")[:2000]
                    )
                    compile_status = self._runner.compile_check(work)
                    tests_after = self._run_task_tests(work)
                    tests_were_evaluated = True
                else:
                    error_msg = err or "Worker output failed to apply"
                    logger.warning(
                        "Output from %s failed to apply: %s", worker_name, error_msg
                    )
            else:
                error_msg = response.error or "Worker generation failed"
                logger.warning(
                    "Worker %s failed: %s", worker_name, error_msg
                )

            # Track for RETRY
            self._last_worker_action = action
            self._last_worker_error = error_msg

        elif action is PlannerAction.RUN_TESTS:
            work = self._work_path()
            if work is not None:
                tests_after = self._run_task_tests(work)
                tests_were_evaluated = True
            latency_ms = (time.monotonic() - step_start) * 1000.0

        elif action is PlannerAction.VERIFY:
            work = self._work_path()
            if work is not None:
                compile_status = self._runner.compile_check(work)
                tests_after = self._run_task_tests(work)
                tests_were_evaluated = True
            latency_ms = (time.monotonic() - step_start) * 1000.0

        elif action is PlannerAction.RETRY:
            if self._last_worker_action is not None:
                worker = self._pool.get_worker(self._last_worker_action)
                worker_name = worker.name

                history_dicts = self._build_worker_history()
                if self._last_worker_error:
                    history_dicts.append(
                        {
                            "action": "RETRY",
                            "error": self._last_worker_error,
                            "test_output": (
                                tests_before.output if tests_before else ""
                            ),
                        }
                    )

                test_dict = self._test_results_to_dict(tests_before)

                response = await worker.generate(
                    prompt=self._task.problem_statement,
                    repository_context=self._state.repo_context,
                    history=history_dicts,
                    test_results=test_dict,
                    code_format=self._worker_code_format(),
                )

                tokens_used = response.tokens_used
                latency_ms = response.latency_ms
                patch_applied = response.patch or response.raw_output

                if response.success and (response.patch or response.raw_output):
                    self._reset_workspace_for_retry()
                    applied, err = self._apply_worker_content(
                        response.patch or response.raw_output
                    )
                    if applied:
                        work = self._work_path()
                        assert work is not None
                        self._state.current_patch = (
                            self._repo.get_diff()
                            if self._mode == "repo"
                            else (response.patch or "")[:2000]
                        )
                        compile_status = self._runner.compile_check(work)
                        tests_after = self._run_task_tests(work)
                        tests_were_evaluated = True
                    else:
                        error_msg = err or "Retry output failed to apply"
                else:
                    error_msg = response.error or "Retry generation failed"

                self._last_worker_error = error_msg
            else:
                logger.warning("RETRY called with no previous worker action")
                latency_ms = 0.0

        elif action is PlannerAction.STOP:
            self._done = True
            latency_ms = 0.0

        # If latency was not set explicitly from the worker response,
        # compute from wall clock.
        if latency_ms == 0.0 and action not in (
            PlannerAction.STOP,
        ):
            latency_ms = (time.monotonic() - step_start) * 1000.0

        # ------------------------------------------------------------------
        # Update state
        # ------------------------------------------------------------------
        self._state.step_number += 1
        self._state.test_results = tests_after
        self._state.compile_status = compile_status
        self._state.remaining_budget = max(
            0.0, 1.0 - self._state.step_number / self._max_steps
        )

        # Changed files
        files_modified: list[str] = []
        if self._mode == "repo" and self._repo.current_path:
            try:
                files_modified = self._repo.get_changed_files()
            except Exception:
                pass
        elif self._mode == "standalone" and self._standalone.current_path:
            files_modified = [self._standalone._solution_name]

        # Build history entry
        history_entry = HistoryEntry(
            step=self._state.step_number,
            action=action,
            worker_used=worker_name,
            tests_before=tests_before,
            tests_after=tests_after,
            compile_status=compile_status,
            patch_applied=patch_applied,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )
        self._state.history.append(history_entry)

        # ------------------------------------------------------------------
        # Determine termination
        # ------------------------------------------------------------------
        all_tests_passed = (
            tests_after is not None
            and tests_after.total > 0
            and tests_after.failed == 0
            and tests_after.errors == 0
        )
        # Compile is OK if not checked this episode or last check succeeded.
        compile_ok = compile_status is None or compile_status.success

        # Auto-terminate after ANY action that produces a fresh passing
        # evaluation (including worker / RETRY steps that run tests).
        if tests_were_evaluated and all_tests_passed and compile_ok:
            self._done = True

        budget_exhausted = self._state.step_number >= self._max_steps
        if budget_exhausted:
            self._done = True

        is_terminal = self._done

        # ------------------------------------------------------------------
        # Compute reward
        # ------------------------------------------------------------------
        reward = self._reward.compute(
            action=action,
            tests_before=tests_before,
            tests_after=tests_after,
            compile_status=compile_status,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            is_terminal=is_terminal,
            all_tests_passed=all_tests_passed,
            budget_exhausted=budget_exhausted,
        )

        # ------------------------------------------------------------------
        # Build transition
        # ------------------------------------------------------------------
        metadata = Metadata(
            worker_name=worker_name or "",
            tests_before=tests_before,
            tests_after=tests_after,
            compile_success=(
                compile_status.success if compile_status else True
            ),
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            files_modified=files_modified,
            patch=patch_applied,
            task_id=self._task.task_id,
            episode_step=self._state.step_number,
        )

        next_state = copy.deepcopy(self._state) if not is_terminal else None

        transition = Transition(
            state=prev_state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=is_terminal,
            metadata=metadata,
        )
        self._transitions.append(transition)

        info: dict[str, Any] = {
            "worker_name": worker_name,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "patch_applied": bool(patch_applied),
            "all_tests_passed": all_tests_passed,
            "budget_exhausted": budget_exhausted,
            "error": error_msg,
            "files_modified": files_modified,
            # Full transition (authoritative); collectors should prefer this
            # or env.transitions[-1] over reconstructing from return values.
            "metadata": metadata,
            "transition": transition,
        }

        logger.info(
            "Step %d: action=%s reward=%.4f done=%s tests=%s",
            self._state.step_number,
            action.name,
            reward,
            is_terminal,
            tests_after.summary() if tests_after else "none",
        )

        return copy.deepcopy(self._state), reward, is_terminal, info

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up workspace and mark the episode finished.

        Cleanup is skipped when ``cleanup_on_done`` is ``False`` (debug).
        """
        self._done = True
        if not self._cleanup_on_done:
            return
        try:
            self._repo.cleanup()
        except Exception as exc:
            logger.warning("Error during repo cleanup: %s", exc)
        try:
            self._standalone.cleanup()
        except Exception as exc:
            logger.warning("Error during standalone cleanup: %s", exc)

    def _run_task_tests(self, work: Path) -> TestResults:
        assert self._task is not None
        if self._mode == "standalone":
            repo_url = str(work)
            cmd = self._task.test_command or "python -m pytest -q --tb=no"
        else:
            repo_url = self._task.repo_url
            cmd = self._task.test_command
        return self._runner.run_tests(work, test_command=cmd, repo_url=repo_url)

    def _worker_code_format(self) -> str:
        """Standalone LCB tasks need full Python; SWE needs git diffs."""
        return "python" if self._mode == "standalone" else "diff"

    def _apply_worker_content(self, content: str) -> tuple[bool, str]:
        """Apply git patch (repo) or write Python solution (standalone)."""
        if self._mode == "standalone":
            ok = self._standalone.apply_worker_output(content)
            return (
                ok,
                ""
                if ok
                else (
                    "Failed to write Python solution (worker must output full "
                    "```python code, not a git diff)"
                ),
            )
        if not self._repo.current_path:
            return False, "No repo path"
        ok = self._repo.apply_patch(content)
        return ok, "" if ok else "Patch failed to apply"

    def _reset_workspace_for_retry(self) -> None:
        assert self._task is not None
        if self._mode == "standalone":
            self._standalone.reset()
            return
        if self._repo.current_path:
            self._repo.reset()
            if self._task.test_patch:
                self._repo.apply_test_patch(self._task.test_patch)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def transitions(self) -> list[Transition]:
        """Return all transitions recorded during this episode."""
        return list(self._transitions)

    @property
    def current_state(self) -> PlannerState | None:
        """Return the current planner state (or ``None`` if not reset)."""
        return copy.deepcopy(self._state) if self._state else None

    @property
    def is_done(self) -> bool:
        """Return ``True`` if the episode has terminated."""
        return self._done

    @property
    def episode_reward(self) -> float:
        """Compute the discounted return for the current episode."""
        return self._reward.compute_episode_reward(self._transitions)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_worker_history(self) -> list[dict[str, Any]]:
        """Convert the state's history entries into dicts for worker prompts."""
        result: list[dict[str, Any]] = []
        for entry in self._state.history:  # type: ignore[union-attr]
            d: dict[str, Any] = {"action": entry.action.name}
            if entry.tests_after and entry.tests_after.output:
                d["test_output"] = entry.tests_after.output
            if entry.patch_applied:
                d["patch"] = entry.patch_applied
            if (
                entry.compile_status
                and not entry.compile_status.success
            ):
                d["error"] = entry.compile_status.error_message
            result.append(d)
        return result

    @staticmethod
    def _test_results_to_dict(
        results: TestResults | None,
    ) -> dict[str, Any] | None:
        """Convert ``TestResults`` to a plain dict for worker prompts."""
        if results is None:
            return None
        return {
            "passed": results.passed,
            "failed": results.failed,
            "errors": results.errors,
            "output": results.output,
        }
