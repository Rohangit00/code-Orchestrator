"""Git repository manager for coding tasks.

Provides shallow cloning, patch application, diff extraction, and cleanup
while enforcing a single-repo-at-a-time policy to stay within disk limits.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class RepoManager:
    """Manages git repositories for coding tasks with minimal disk usage.

    Enforces a single-repo-at-a-time policy to stay within disk limits.
    """

    def __init__(
        self,
        workspace_dir: str = "./workspace/repos",
        max_disk_mb: int = 8000,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.max_disk_mb = max_disk_mb
        self._current_repo: Path | None = None
        self._base_commit: str | None = None
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    # -- public API ----------------------------------------------------------

    def clone(self, repo_url: str, commit_hash: str) -> Path:
        """Clone a repo and checkout a specific commit via shallow fetch.

        Cleans up any existing repo first, then:
        ``git init`` → ``git remote add`` → ``git fetch --depth=1`` →
        ``git checkout FETCH_HEAD``

        Returns the path to the cloned repository root.
        """
        # Enforce single-repo policy
        self.cleanup()

        # Derive a directory name from the URL
        repo_name = repo_url.rstrip("/").rsplit("/", 1)[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        repo_dir = self.workspace_dir / repo_name
        repo_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._run_git(["init"], cwd=repo_dir)
            self._run_git(["remote", "add", "origin", repo_url], cwd=repo_dir)
            self._run_git(
                ["fetch", "--depth=1", "origin", commit_hash],
                cwd=repo_dir,
            )
            self._run_git(["checkout", "FETCH_HEAD"], cwd=repo_dir)
        except subprocess.CalledProcessError as exc:
            logger.error("Clone failed for %s@%s: %s", repo_url, commit_hash, exc)
            shutil.rmtree(repo_dir, ignore_errors=True)
            raise

        self._current_repo = repo_dir
        self._base_commit = commit_hash
        logger.info(
            "Cloned %s@%s → %s (%.1f MB)",
            repo_url,
            commit_hash[:10],
            repo_dir,
            self.disk_usage_mb(),
        )
        return repo_dir

    def apply_patch(self, patch: str) -> bool:
        """Apply a unified diff patch to the current repo.

        Tries ``git apply`` first; falls back to ``git apply --3way``.
        Returns ``True`` on success, ``False`` otherwise.
        """
        if self._current_repo is None:
            logger.error("No repo cloned — cannot apply patch.")
            return False

        patch_file = None
        try:
            # Write patch to a temp file inside the workspace so git can read it
            patch_file = Path(
                tempfile.mktemp(suffix=".patch", dir=self.workspace_dir)
            )
            patch_file.write_text(patch, encoding="utf-8")

            # First attempt: straight apply
            try:
                self._run_git(
                    ["apply", "--verbose", str(patch_file)],
                    cwd=self._current_repo,
                )
                logger.info("Patch applied successfully (git apply).")
                return True
            except subprocess.CalledProcessError:
                logger.warning("git apply failed, trying --3way…")

            # Second attempt: three-way merge
            try:
                self._run_git(
                    ["apply", "--3way", str(patch_file)],
                    cwd=self._current_repo,
                )
                logger.info("Patch applied successfully (git apply --3way).")
                return True
            except subprocess.CalledProcessError as exc:
                logger.error("Patch application failed: %s", exc)
                return False
        finally:
            if patch_file is not None and patch_file.exists():
                patch_file.unlink()

    def apply_test_patch(self, test_patch: str) -> bool:
        """Apply a test patch from SWE-bench."""
        return self.apply_patch(test_patch)

    def get_diff(self) -> str:
        """Return the current diff of the working tree vs HEAD."""
        if self._current_repo is None:
            return ""
        try:
            result = self._run_git(["diff", "HEAD"], cwd=self._current_repo)
            return result.stdout
        except subprocess.CalledProcessError:
            return ""

    def get_changed_files(self) -> list[str]:
        """List files changed relative to HEAD (staged + unstaged)."""
        if self._current_repo is None:
            return []
        try:
            result = self._run_git(
                ["diff", "--name-only", "HEAD"],
                cwd=self._current_repo,
            )
            return [f for f in result.stdout.strip().splitlines() if f]
        except subprocess.CalledProcessError:
            return []

    def reset(self) -> None:
        """Hard reset: discard all local changes.

        ``git checkout -- .`` followed by ``git clean -fd``.
        """
        if self._current_repo is None:
            return
        try:
            self._run_git(["checkout", "--", "."], cwd=self._current_repo)
            self._run_git(["clean", "-fd"], cwd=self._current_repo)
            logger.info("Repo reset to clean state.")
        except subprocess.CalledProcessError as exc:
            logger.error("Reset failed: %s", exc)

    def cleanup(self) -> None:
        """Delete the cloned repo directory and clear internal state."""
        if self._current_repo is not None and self._current_repo.exists():
            shutil.rmtree(self._current_repo, ignore_errors=True)
            logger.info("Cleaned up %s", self._current_repo)
        self._current_repo = None
        self._base_commit = None

    def disk_usage_mb(self) -> float:
        """Total disk usage of the workspace directory in megabytes."""
        total_bytes = 0
        if not self.workspace_dir.exists():
            return 0.0
        for entry in self.workspace_dir.rglob("*"):
            try:
                if entry.is_file():
                    total_bytes += entry.stat().st_size
            except OSError:
                pass
        return total_bytes / (1024 * 1024)

    # -- properties ----------------------------------------------------------

    @property
    def current_path(self) -> Path | None:
        """Path to the currently cloned repository, or ``None``."""
        return self._current_repo

    @property
    def base_commit(self) -> str | None:
        """The commit hash that was checked out, or ``None``."""
        return self._base_commit

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _run_git(
        args: list[str],
        cwd: Path,
        *,
        timeout: int = 300,
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command and return the completed process.

        Raises ``subprocess.CalledProcessError`` on non-zero exit.
        """
        cmd = ["git"] + args
        logger.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
