"""MBPP dataset adapter for the Fugu orchestrator.

Wraps the ``google-research-datasets/mbpp`` dataset from HuggingFace
and converts each record into a :class:`~fugu.datasets.base.CodingTask`.
MBPP tasks are standalone function-generation problems — no repository
cloning is required.

.. warning::

   Collection and evaluation through :class:`~fugu.env.coding_env.CodingEnvironment`
   are **not supported yet**. The env requires a ``repo_url`` and git-style
   patches. Use SWE-bench (``fugu-collect -d swebench-lite``, etc.) until a
   standalone Python workspace harness exists.

Usage (inspection only)::

    ds = MBPPDataset()
    for task in ds:
        print(task.task_id, task.problem_statement)
"""

from __future__ import annotations

import logging
from typing import Iterator

from fugu.datasets.base import BaseDataset, CodingTask

logger = logging.getLogger(__name__)

# HuggingFace dataset identifier
_HF_DATASET_PATH = "google-research-datasets/mbpp"


class MBPPDataset(BaseDataset):
    """Adapter for the Mostly Basic Python Problems (MBPP) benchmark.

    Parameters:
        hf_split: The HuggingFace split to load.  Common choices:
            ``"test"`` (the standard 500-problem evaluation split),
            ``"train"`` (the 374-problem training split),
            ``"validation"`` (10 few-shot examples),
            ``"prompt"`` (10 prompt examples).
            Defaults to ``"test"``.
    """

    def __init__(self, hf_split: str = "test") -> None:
        self._hf_split = hf_split
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
            "Loading MBPP dataset (split=%s) …", self._hf_split
        )
        self._dataset = load_dataset(
            _HF_DATASET_PATH, split=self._hf_split
        )
        logger.info("Loaded %d problems.", len(self._dataset))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mbpp"

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
            task_id     → task_id  (int → str)
            text        → problem_statement
            code        → gold_patch
            test_list   → test_patch  (joined with newlines)
        """
        task_id = str(row.get("task_id", ""))
        text = str(row.get("text", ""))
        code = str(row.get("code", ""))

        # test_list is a list of assert-based test strings
        test_list: list[str] = row.get("test_list", [])
        if isinstance(test_list, str):
            # Defensive: some versions may store as a single string
            test_list = [test_list]

        # Combine test assertions into a single test file
        test_code = "\n".join(test_list)

        # Attempt to extract the entry point (function name) from the
        # gold solution by looking for `def <name>(`.
        entry_point = _extract_entry_point(code)

        # Build a descriptive problem statement
        problem_statement = (
            f"Write a Python function to solve the following problem:\n\n"
            f"{text}"
        )

        # challenge_test_list may be present in some splits
        challenge_tests: list[str] = row.get("challenge_test_list", [])
        if isinstance(challenge_tests, str):
            challenge_tests = [challenge_tests]

        # Collect remaining fields as metadata
        preserved_keys = {
            "task_id",
            "text",
            "code",
            "test_list",
            "challenge_test_list",
        }
        extra_metadata = {
            k: v for k, v in row.items() if k not in preserved_keys
        }
        if challenge_tests:
            extra_metadata["challenge_test_list"] = challenge_tests

        return CodingTask(
            task_id=task_id,
            problem_statement=problem_statement,
            repo_url=None,
            base_commit=None,
            test_patch=test_code,
            gold_patch=code,
            entry_point=entry_point,
            metadata=extra_metadata,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_entry_point(code: str) -> str | None:
    """Extract the first function name from a Python code string.

    Looks for ``def <name>(`` patterns and returns *name*.
    Returns ``None`` if no function definition is found.
    """
    import re

    match = re.search(r"def\s+(\w+)\s*\(", code)
    return match.group(1) if match else None
