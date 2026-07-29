"""Local filesystem workspace for tasks without a git repository.

Used by LiveCodeBench / HumanEval-style problems: write starter + tests,
overwrite solution files from worker output, run pytest in-process or Docker.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def looks_like_git_diff(text: str) -> bool:
    """Return True if *text* looks like a unified git diff."""
    if not text or not text.strip():
        return False
    head = text.lstrip()[:200]
    return (
        head.startswith("diff --git")
        or head.startswith("--- a/")
        or head.startswith("--- /dev/null")
        or "\n@@ " in text[:2000]
    )


def extract_python_code(raw_output: str) -> str:
    """Extract Python source from model output (fenced or whole text)."""
    if not raw_output:
        return ""

    # Prefer ```python ... ``` then bare ```
    for pattern in (
        r"```(?:python|py)\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ):
        m = re.search(pattern, raw_output, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Strip leading prose if the rest looks like code
    text = raw_output.strip()
    if "def " in text or "class " in text or "import " in text:
        # Drop lines before first code-like line
        lines = text.splitlines()
        start = 0
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith(
                ("def ", "class ", "import ", "from ", "#", "#!/")
            ) or (s and not s[0].isalpha() and s[0] in ("@",)):
                start = i
                break
            if s.startswith("def ") or s.startswith("class "):
                start = i
                break
        return "\n".join(lines[start:]).strip()

    return text


class StandaloneWorkspace:
    """Create and manage a single-task directory (no git)."""

    def __init__(self, workspace_dir: str | Path = "/tmp/fugu_standalone") -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._current: Path | None = None
        self._starter_files: dict[str, str] = {}
        self._solution_name: str = "solution.py"

    @property
    def current_path(self) -> Path | None:
        return self._current

    def create(
        self,
        task_id: str,
        files: dict[str, str],
        *,
        solution_name: str = "solution.py",
    ) -> Path:
        """Materialise *files* under a fresh directory for *task_id*."""
        self.cleanup()
        safe = re.sub(r"[^\w.\-]+", "_", task_id)[:120] or "task"
        path = self.workspace_dir / safe
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

        self._starter_files = dict(files)
        self._solution_name = solution_name
        for rel, content in files.items():
            dest = path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        self._current = path
        logger.info("Standalone workspace ready: %s (%d files)", path, len(files))
        return path

    def write_solution(self, code: str, filename: str | None = None) -> bool:
        """Overwrite the solution file with *code*."""
        if self._current is None:
            return False
        name = filename or self._solution_name
        try:
            (self._current / name).write_text(code, encoding="utf-8")
            return True
        except OSError as exc:
            logger.error("Failed to write solution: %s", exc)
            return False

    def apply_worker_output(self, raw: str) -> bool:
        """Apply worker output as Python solution code."""
        code = extract_python_code(raw)
        if not code.strip():
            return False
        if looks_like_git_diff(code) and "def " not in code and "class " not in code:
            logger.warning(
                "Standalone workspace expected Python code, got diff-like text"
            )
            return False
        return self.write_solution(code)

    def reset(self) -> None:
        """Restore starter files (discard worker edits)."""
        if self._current is None:
            return
        for rel, content in self._starter_files.items():
            dest = self._current / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        logger.info("Standalone workspace reset to starter files")

    def cleanup(self) -> None:
        """Delete current task directory."""
        if self._current is not None and self._current.exists():
            shutil.rmtree(self._current, ignore_errors=True)
            logger.info("Cleaned up standalone workspace %s", self._current)
        self._current = None
        self._starter_files = {}

    def file_tree_summary(self, max_chars: int = 2000) -> str:
        """Short listing of files for planner context."""
        if self._current is None:
            return ""
        lines: list[str] = ["[Standalone workspace]"]
        for p in sorted(self._current.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self._current)
                lines.append(f"  {rel}")
        text = "\n".join(lines)
        return text[:max_chars]
