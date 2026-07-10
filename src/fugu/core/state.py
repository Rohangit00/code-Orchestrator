"""MDP state representations for the Fugu orchestrator.

All dataclasses use ``__slots__`` for memory efficiency and faster attribute
access.  Serialisation helpers (``to_dict`` / ``from_dict``) round-trip
through plain Python dicts so that transitions can be stored in replay
buffers without depending on ``pickle``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from fugu.core.actions import PlannerAction


# ---------------------------------------------------------------------------
# Test & compile results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TestResults:
    """Aggregated results from a test-suite execution."""

    passed: int = 0
    failed: int = 0
    errors: int = 0
    output: str = ""
    duration_ms: float = 0.0
    test_names_passed: list[str] = field(default_factory=list)
    test_names_failed: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of tests that were executed."""
        return self.passed + self.failed + self.errors

    @property
    def pass_rate(self) -> float:
        """Fraction of tests that passed (0.0–1.0). Returns 0.0 when no tests ran."""
        return self.passed / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        """One-line human-readable summary of the test results."""
        return (
            f"passed: {self.passed}, failed: {self.failed}, "
            f"errors: {self.errors}, total: {self.total}, "
            f"pass_rate: {self.pass_rate:.2%}, duration: {self.duration_ms:.0f}ms"
        )


@dataclass(slots=True)
class CompileStatus:
    """Result of a compilation / syntax check."""

    success: bool = True
    error_message: str = ""


# ---------------------------------------------------------------------------
# Episode history
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HistoryEntry:
    """A single step in the orchestration history visible to the planner."""

    step: int
    action: PlannerAction
    worker_used: str | None = None
    tests_before: TestResults | None = None
    tests_after: TestResults | None = None
    compile_status: CompileStatus | None = None
    patch_applied: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Planner observation state
# ---------------------------------------------------------------------------

_REPO_CONTEXT_MAX_CHARS = 2000
_HISTORY_WINDOW = 5


@dataclass(slots=True)
class PlannerState:
    """Full observation state fed into the planner model.

    ``to_prompt()`` serialises the state into a structured tagged string
    that can be tokenised directly by the planner LM.
    """

    task_description: str = ""
    repo_context: str = ""
    history: list[HistoryEntry] = field(default_factory=list)
    test_results: TestResults | None = None
    compile_status: CompileStatus | None = None
    current_patch: str = ""
    step_number: int = 0
    max_steps: int = 20
    remaining_budget: float = 1.0

    def to_prompt(self) -> str:
        """Serialise this state into a tagged prompt for the planner model.

        Format
        ------
        ::

            <|task|>...<|/task|>
            <|repo|>...<|/repo|>
            <|history|>...<|/history|>
            <|tests|>...<|/tests|>
            <|compile|>...<|/compile|>
            <|step|>N/M<|/step|>
        """
        parts: list[str] = []

        # Task description
        parts.append(f"<|task|>{self.task_description}<|/task|>")

        # Repository context (truncated)
        repo_ctx = self.repo_context[:_REPO_CONTEXT_MAX_CHARS]
        if len(self.repo_context) > _REPO_CONTEXT_MAX_CHARS:
            repo_ctx += "…"
        parts.append(f"<|repo|>{repo_ctx}<|/repo|>")

        # History (last N entries)
        history_lines: list[str] = []
        recent = self.history[-_HISTORY_WINDOW:]
        for entry in recent:
            line = f"step {entry.step}: {entry.action.name}"
            if entry.tests_after is not None:
                line += (
                    f" -> passed: {entry.tests_after.passed}, "
                    f"failed: {entry.tests_after.failed}"
                )
            if entry.worker_used is not None:
                line += f" (worker: {entry.worker_used})"
            if entry.tokens_used > 0:
                line += f" [tokens: {entry.tokens_used}]"
            history_lines.append(line)
        parts.append(f"<|history|>{chr(10).join(history_lines)}<|/history|>")

        # Test results
        if self.test_results is not None:
            parts.append(
                f"<|tests|>passed: {self.test_results.passed}, "
                f"failed: {self.test_results.failed}, "
                f"errors: {self.test_results.errors}<|/tests|>"
            )
        else:
            parts.append("<|tests|>no tests run yet<|/tests|>")

        # Compile status
        if self.compile_status is not None:
            if self.compile_status.success:
                parts.append("<|compile|>success<|/compile|>")
            else:
                parts.append(
                    f"<|compile|>failed: {self.compile_status.error_message}<|/compile|>"
                )
        else:
            parts.append("<|compile|>unknown<|/compile|>")

        # Step counter
        parts.append(f"<|step|>{self.step_number}/{self.max_steps}<|/step|>")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Transition metadata & MDP transition tuple
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Metadata:
    """Auxiliary information attached to a single MDP transition."""

    worker_name: str = ""
    tests_before: TestResults | None = None
    tests_after: TestResults | None = None
    compile_success: bool = True
    tokens_used: int = 0
    latency_ms: float = 0.0
    files_modified: list[str] = field(default_factory=list)
    patch: str = ""
    task_id: str = ""
    episode_step: int = 0


@dataclass(slots=True)
class Transition:
    """A single (s, a, r, s', done) transition in the MDP.

    Provides ``to_dict`` / ``from_dict`` for serialisation into replay
    buffers and dataset files.
    """

    state: PlannerState
    action: PlannerAction
    reward: float
    next_state: PlannerState | None
    done: bool
    metadata: Metadata = field(default_factory=Metadata)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict suitable for JSON / msgpack storage.

        Enum values are stored as ints; nested dataclasses are recursively
        converted via ``dataclasses.asdict``.
        """
        def _convert(obj: Any) -> Any:
            if isinstance(obj, PlannerAction):
                return int(obj)
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, list):
                return [_convert(item) for item in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj

        return {
            "state": _convert(self.state),
            "action": int(self.action),
            "reward": self.reward,
            "next_state": _convert(self.next_state) if self.next_state is not None else None,
            "done": self.done,
            "metadata": _convert(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transition:
        """Reconstruct a ``Transition`` from a plain dict.

        All nested dataclasses and enums are fully reconstructed.
        """
        state = _planner_state_from_dict(data["state"])
        next_state = (
            _planner_state_from_dict(data["next_state"])
            if data["next_state"] is not None
            else None
        )
        action = PlannerAction(data["action"])
        metadata = _metadata_from_dict(data.get("metadata", {}))
        return cls(
            state=state,
            action=action,
            reward=float(data["reward"]),
            next_state=next_state,
            done=bool(data["done"]),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Private reconstruction helpers
# ---------------------------------------------------------------------------


def _test_results_from_dict(d: dict[str, Any]) -> TestResults:
    return TestResults(
        passed=int(d.get("passed", 0)),
        failed=int(d.get("failed", 0)),
        errors=int(d.get("errors", 0)),
        output=str(d.get("output", "")),
        duration_ms=float(d.get("duration_ms", 0.0)),
        test_names_passed=list(d.get("test_names_passed", [])),
        test_names_failed=list(d.get("test_names_failed", [])),
    )


def _compile_status_from_dict(d: dict[str, Any]) -> CompileStatus:
    return CompileStatus(
        success=bool(d.get("success", True)),
        error_message=str(d.get("error_message", "")),
    )


def _history_entry_from_dict(d: dict[str, Any]) -> HistoryEntry:
    return HistoryEntry(
        step=int(d["step"]),
        action=PlannerAction(d["action"]),
        worker_used=d.get("worker_used"),
        tests_before=(
            _test_results_from_dict(d["tests_before"])
            if d.get("tests_before") is not None
            else None
        ),
        tests_after=(
            _test_results_from_dict(d["tests_after"])
            if d.get("tests_after") is not None
            else None
        ),
        compile_status=(
            _compile_status_from_dict(d["compile_status"])
            if d.get("compile_status") is not None
            else None
        ),
        patch_applied=str(d.get("patch_applied", "")),
        tokens_used=int(d.get("tokens_used", 0)),
        latency_ms=float(d.get("latency_ms", 0.0)),
    )


def _planner_state_from_dict(d: dict[str, Any]) -> PlannerState:
    return PlannerState(
        task_description=str(d.get("task_description", "")),
        repo_context=str(d.get("repo_context", "")),
        history=[_history_entry_from_dict(h) for h in d.get("history", [])],
        test_results=(
            _test_results_from_dict(d["test_results"])
            if d.get("test_results") is not None
            else None
        ),
        compile_status=(
            _compile_status_from_dict(d["compile_status"])
            if d.get("compile_status") is not None
            else None
        ),
        current_patch=str(d.get("current_patch", "")),
        step_number=int(d.get("step_number", 0)),
        max_steps=int(d.get("max_steps", 20)),
        remaining_budget=float(d.get("remaining_budget", 1.0)),
    )


def _metadata_from_dict(d: dict[str, Any]) -> Metadata:
    return Metadata(
        worker_name=str(d.get("worker_name", "")),
        tests_before=(
            _test_results_from_dict(d["tests_before"])
            if d.get("tests_before") is not None
            else None
        ),
        tests_after=(
            _test_results_from_dict(d["tests_after"])
            if d.get("tests_after") is not None
            else None
        ),
        compile_success=bool(d.get("compile_success", True)),
        tokens_used=int(d.get("tokens_used", 0)),
        latency_ms=float(d.get("latency_ms", 0.0)),
        files_modified=list(d.get("files_modified", [])),
        patch=str(d.get("patch", "")),
        task_id=str(d.get("task_id", "")),
        episode_step=int(d.get("episode_step", 0)),
    )
