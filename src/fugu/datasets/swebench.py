"""SWE-bench dataset adapter for the Fugu orchestrator.

Supports *SWE-bench Lite*, *Full*, and *Verified* splits via the
`HuggingFace datasets <https://huggingface.co/docs/datasets/>`_
library.  The dataset is lazy-loaded on first iteration or size
query to avoid import-time network calls.

Usage::

    ds = SWEBenchDataset(split="lite")
    for task in ds:
        print(task.task_id, task.repo_url)
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

from fugu.datasets.base import BaseDataset, CodingTask

logger = logging.getLogger(__name__)

# Mapping from user-friendly split names to HuggingFace dataset paths.
_SPLIT_TO_HF: dict[str, str] = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
    "verified": "princeton-nlp/SWE-bench_Verified",
}

# GitHub base URL used to reconstruct clone URLs from repo slugs.
_GITHUB_BASE = "https://github.com"


class SWEBenchDataset(BaseDataset):
    """Adapter that wraps SWE-bench (Lite / Full / Verified).

    Parameters:
        split: One of ``"lite"``, ``"full"``, or ``"verified"``.
            Alternatively, a full HuggingFace dataset path may be
            passed directly.
        hf_split: The HuggingFace split name (e.g. ``"test"``).
            Defaults to ``"test"`` which is the standard evaluation
            split for SWE-bench.
    """

    def __init__(
        self,
        split: str = "lite",
        hf_split: str = "test",
    ) -> None:
        self._split = split
        self._hf_split = hf_split
        self._hf_dataset_path = _SPLIT_TO_HF.get(split.lower(), split)
        self._dataset = None  # lazily loaded

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the HuggingFace dataset on first access."""
        if self._dataset is not None:
            return
        from datasets import load_dataset

        logger.info(
            "Loading SWE-bench dataset '%s' (split=%s) …",
            self._hf_dataset_path,
            self._hf_split,
        )
        self._dataset = load_dataset(
            self._hf_dataset_path, split=self._hf_split
        )
        logger.info("Loaded %d instances.", len(self._dataset))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"swebench_{self._split}"

    @property
    def size(self) -> int:
        self._load()
        assert self._dataset is not None
        return len(self._dataset)

    def __iter__(self) -> Iterator[CodingTask]:
        self._load()
        assert self._dataset is not None
        for row in self._dataset:
            yield self._row_to_task(row)

    # ------------------------------------------------------------------
    # Row conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_task(row: dict) -> CodingTask:
        """Convert a single HuggingFace row to a :class:`CodingTask`.

        Field mapping:
            instance_id        → task_id
            problem_statement  → problem_statement
            repo               → repo_url  (prefixed with GitHub URL)
            base_commit        → base_commit
            test_patch         → test_patch
            patch              → gold_patch
            FAIL_TO_PASS       → fail_to_pass  (JSON-encoded string → list)
            PASS_TO_PASS       → pass_to_pass  (JSON-encoded string → list)
        """
        # Repo slug → full clone URL
        repo_slug = row.get("repo", "")
        repo_url = (
            f"{_GITHUB_BASE}/{repo_slug}.git" if repo_slug else None
        )

        # FAIL_TO_PASS / PASS_TO_PASS may be JSON-encoded strings or
        # already lists, depending on the datasets library version.
        fail_to_pass = _parse_json_list(row.get("FAIL_TO_PASS", "[]"))
        pass_to_pass = _parse_json_list(row.get("PASS_TO_PASS", "[]"))

        # Collect remaining fields as metadata
        preserved_keys = {
            "instance_id",
            "problem_statement",
            "repo",
            "base_commit",
            "test_patch",
            "patch",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
        }
        extra_metadata = {
            k: v for k, v in row.items() if k not in preserved_keys
        }

        return CodingTask(
            task_id=str(row["instance_id"]),
            problem_statement=str(row.get("problem_statement", "")),
            repo_url=repo_url,
            base_commit=row.get("base_commit"),
            test_patch=row.get("test_patch"),
            gold_patch=row.get("patch"),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            metadata=extra_metadata,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_list(value) -> list[str]:
    """Parse a JSON string into a list, tolerating already-decoded values.

    SWE-bench stores ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` as JSON-encoded
    strings in the Parquet files.  Some HuggingFace versions auto-decode
    them while others leave them as raw strings.

    Args:
        value: Either a ``str`` containing a JSON array, an actual
            ``list``, or ``None``.

    Returns:
        A plain ``list[str]``.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (json.JSONDecodeError, TypeError):
            # Fall back to splitting on commas for malformed data
            if value.strip():
                return [v.strip() for v in value.split(",") if v.strip()]
    return []
