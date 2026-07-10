"""Repository context extraction for the planner and workers.

Provides file-tree generation and intelligent file content extraction by
parsing tracebacks and task descriptions for file references.
"""

from __future__ import annotations

import re
from pathlib import Path


class RepoContext:
    """Extracts and caches repository context for the planner/workers."""

    def __init__(
        self,
        repo_path: Path,
        max_tree_depth: int = 3,
        max_file_chars: int = 5000,
    ) -> None:
        self.repo_path = repo_path
        self.max_tree_depth = max_tree_depth
        self.max_file_chars = max_file_chars
        self._exclude_dirs: set[str] = {
            ".git",
            "__pycache__",
            "node_modules",
            ".tox",
            "venv",
            "env",
            ".env",
            ".eggs",
        }
        self._exclude_exts: set[str] = {
            ".pyc",
            ".pyo",
            ".pyd",
            ".so",
            ".dll",
            ".class",
            ".exe",
        }
        self._cached_tree: str | None = None

    # -- public API ----------------------------------------------------------

    def get_file_tree(self) -> str:
        """Generate a truncated file tree with box-drawing characters.

        Results are cached; call :meth:`invalidate_cache` to force
        regeneration.  The tree respects ``max_tree_depth`` and filters
        out excluded directories and file extensions.
        """
        if self._cached_tree is not None:
            return self._cached_tree

        lines: list[str] = [self.repo_path.name + "/"]
        self._build_tree(self.repo_path, "", 0, lines)
        self._cached_tree = "\n".join(lines)
        return self._cached_tree

    def get_relevant_files(
        self,
        error_output: str = "",
        task_description: str = "",
    ) -> str:
        """Extract the content of files referenced in tracebacks or the task description.

        * Parses ``File "path.py"`` patterns from Python tracebacks.
        * Finds ``.py`` file references in the task description text.
        * Each file is truncated to ``max_file_chars``.

        Returns a formatted string with one section per file.
        """
        referenced: dict[str, Path] = {}  # relative_path -> absolute Path

        # 1. Parse traceback-style references: File "some/path.py", line N
        if error_output:
            for match in re.finditer(r'File "([^"]+\.py)"', error_output):
                self._resolve_and_add(match.group(1), referenced)

        # 2. Parse general .py references from the task description
        if task_description:
            # Match tokens that look like file paths ending in .py
            for match in re.finditer(
                r"(?:^|[\s\"'`(,])([A-Za-z0-9_./-]+\.py)\b", task_description
            ):
                self._resolve_and_add(match.group(1), referenced)

        if not referenced:
            return ""

        sections: list[str] = []
        for rel_path, abs_path in sorted(referenced.items()):
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                content = "<could not read file>"

            if len(content) > self.max_file_chars:
                content = content[: self.max_file_chars] + "\n… (truncated)"

            sections.append(f"--- {rel_path} ---\n{content}")

        return "\n\n".join(sections)

    def get_summary(
        self,
        error_output: str = "",
        task_description: str = "",
    ) -> str:
        """Combined summary: file tree followed by relevant file contents."""
        parts: list[str] = []

        tree = self.get_file_tree()
        parts.append(f"[File Tree]\n{tree}")

        relevant = self.get_relevant_files(error_output, task_description)
        if relevant:
            parts.append(f"[Relevant Files]\n{relevant}")

        return "\n\n".join(parts)

    def invalidate_cache(self) -> None:
        """Clear the cached file tree so the next call regenerates it."""
        self._cached_tree = None

    # -- private helpers -----------------------------------------------------

    def _build_tree(
        self,
        directory: Path,
        prefix: str,
        depth: int,
        lines: list[str],
    ) -> None:
        """Recursively build the tree lines with box-drawing characters."""
        if depth >= self.max_tree_depth:
            return

        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
        except PermissionError:
            return

        # Filter entries
        filtered: list[Path] = []
        for entry in entries:
            if entry.is_dir() and entry.name in self._exclude_dirs:
                continue
            if entry.is_file() and entry.suffix in self._exclude_exts:
                continue
            filtered.append(entry)

        for i, entry in enumerate(filtered):
            is_last = i == len(filtered) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                self._build_tree(entry, child_prefix, depth + 1, lines)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")

    def _resolve_and_add(
        self,
        raw_path: str,
        referenced: dict[str, Path],
    ) -> None:
        """Try to resolve *raw_path* relative to the repo root and add it."""
        raw_path = raw_path.strip()
        if not raw_path:
            return

        candidate = self.repo_path / raw_path
        if candidate.is_file():
            rel = str(candidate.relative_to(self.repo_path))
            referenced.setdefault(rel, candidate)
            return

        # The path may be absolute; try to map it back into the repo
        try:
            p = Path(raw_path)
            if p.is_absolute():
                try:
                    rel = str(p.relative_to(self.repo_path))
                    if (self.repo_path / rel).is_file():
                        referenced.setdefault(rel, self.repo_path / rel)
                        return
                except ValueError:
                    pass

            # Last resort: search for the filename inside the repo
            filename = Path(raw_path).name
            for match in self.repo_path.rglob(filename):
                if match.is_file() and match.suffix not in self._exclude_exts:
                    rel = str(match.relative_to(self.repo_path))
                    referenced.setdefault(rel, match)
                    break  # take the first match only
        except (OSError, ValueError):
            pass
