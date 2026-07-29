"""Abstract base worker interface and shared response type.

Every coding worker (vLLM, mock, future providers) inherits from
:class:`BaseWorker` and implements :meth:`generate`.  The response is always
a :class:`WorkerResponse` carrying the generated patch, token usage, and
timing metadata.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------


@dataclass
class WorkerResponse:
    """Structured response returned by every coding worker.

    Attributes:
        patch: A unified git diff (``git diff`` format) representing the fix.
        explanation: Optional model reasoning extracted from the output.
        raw_output: The full, unparsed model output.
        tokens_used: Total tokens consumed (prompt + completion).
        prompt_tokens: Tokens consumed by the prompt.
        completion_tokens: Tokens consumed by the completion.
        latency_ms: Wall-clock time for the request in milliseconds.
        success: ``True`` if the generation completed without error.
        error: Human-readable error string when ``success`` is ``False``.
    """

    patch: str
    explanation: str = ""
    raw_output: str = ""
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Abstract base worker
# ---------------------------------------------------------------------------


class BaseWorker(ABC):
    """Abstract base class for all coding worker models.

    Sub-classes must implement :meth:`generate` which receives task context
    and returns a :class:`WorkerResponse`.

    Parameters:
        name: Human-readable identifier for the worker (e.g. ``"qwen"``).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        repository_context: str,
        history: list[dict],
        test_results: dict | None = None,
        *,
        code_format: str = "diff",
    ) -> WorkerResponse:
        """Generate a code fix for the given coding task.

        Args:
            prompt: The coding task / issue description.
            repository_context: Summarised repo info (file tree, relevant
                file contents).
            history: Previous attempts and their outcomes.  Each entry is a
                dict with optional keys ``action``, ``error``, ``test_output``,
                ``patch``.
            test_results: Current test pass/fail state as a dict with keys
                ``passed``, ``failed``, ``errors``, ``output``.
            code_format: ``"diff"`` for SWE-bench (unified git patch) or
                ``"python"`` for LiveCodeBench-style full-file solutions.

        Returns:
            A :class:`WorkerResponse` with the generated patch/code and metadata.
        """
        ...

    # -- Prompt helpers -----------------------------------------------------

    def _build_system_prompt(self, code_format: str = "diff") -> str:
        """System prompt for either git-diff or full-Python output."""
        fmt = (code_format or "diff").lower().strip()
        if fmt == "python":
            return (
                "You are an expert competitive programmer.  Solve the problem "
                "by writing a complete Python program.\n\n"
                "IMPORTANT:\n"
                "- Output the FULL contents of solution.py (runnable Python).\n"
                "- Wrap the code in a ```python code fence.\n"
                "- Do NOT output a git diff, unified patch, or ```diff block.\n"
                "- Do NOT use lines like @@ or --- a/ or +++ b/.\n"
                "- Include only the Python source (imports + solution).\n"
                "- Read stdin / write stdout if the problem requires it.\n"
            )
        return (
            "You are an expert software engineer.  Your task is to fix the "
            "described issue by generating a code patch.\n\n"
            "IMPORTANT:\n"
            "- Output your fix as a unified git diff (the kind produced by "
            "`git diff`).\n"
            "- Wrap the diff in a ```diff code fence.\n"
            "- Include ONLY the diff, nothing else.\n"
            "- The diff should be directly applicable with `git apply`.\n"
            "- Be minimal — only change what is necessary to fix the issue.\n"
        )

    def _build_user_prompt(
        self,
        prompt: str,
        repository_context: str,
        history: list[dict],
        test_results: dict | None,
    ) -> str:
        """Combine issue, repo context, previous attempts and test state.

        Keeps the last **3** history entries and truncates test output to
        **1 000** characters to avoid blowing up the context window.

        Returns:
            A single string suitable as the ``user`` message.
        """
        parts: list[str] = [f"## Issue\n{prompt}"]

        if repository_context:
            parts.append(f"## Repository Context\n{repository_context}")

        if history:
            parts.append("## Previous Attempts")
            for entry in history[-3:]:
                parts.append(f"- Action: {entry.get('action', 'unknown')}")
                if entry.get("error"):
                    parts.append(f"  Error: {entry['error'][:500]}")
                if entry.get("test_output"):
                    parts.append(
                        f"  Test output: {entry['test_output'][:500]}"
                    )
                if entry.get("patch"):
                    parts.append(f"  Previous patch:\n{entry['patch'][:500]}")

        if test_results:
            parts.append(
                "## Current Test Results\n"
                f"Passed: {test_results.get('passed', 0)}, "
                f"Failed: {test_results.get('failed', 0)}, "
                f"Errors: {test_results.get('errors', 0)}"
            )
            output = test_results.get("output", "")
            if output:
                parts.append(f"Test output:\n{output[:1000]}")

        return "\n\n".join(parts)

    # -- Patch extraction ---------------------------------------------------

    @staticmethod
    def _extract_patch(raw_output: str, code_format: str = "diff") -> str:
        """Extract git diff or Python source depending on *code_format*."""
        if not raw_output:
            return ""

        fmt = (code_format or "diff").lower().strip()
        if fmt == "python":
            from fugu.workspace.standalone import extract_python_code

            return extract_python_code(raw_output)

        # Prefer explicit language fences
        for lang in ("diff", "python", "py", ""):
            if lang:
                pat = rf"```{lang}\s*\n(.*?)```"
            else:
                pat = r"```\s*\n(.*?)```"
            code_block = re.search(pat, raw_output, re.DOTALL | re.IGNORECASE)
            if code_block:
                return code_block.group(1).strip()

        # Diff header lines
        lines = raw_output.split("\n")
        diff_lines: list[str] = []
        in_diff = False
        for line in lines:
            if line.startswith("diff --git") or (
                not in_diff
                and (line.startswith("--- a/") or line.startswith("+++ b/"))
            ):
                in_diff = True
            if in_diff:
                diff_lines.append(line)

        if diff_lines:
            return "\n".join(diff_lines).strip()

        return raw_output.strip()
