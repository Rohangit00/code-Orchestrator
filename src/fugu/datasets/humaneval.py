"""HumanEval dataset adapter for the Fugu orchestrator.

Wraps the ``openai_humaneval`` dataset from HuggingFace and converts
each record into a :class:`~fugu.datasets.base.CodingTask`.
HumanEval tasks are standalone function-generation problems — no
repository cloning is required.

.. warning::

   Collection and evaluation through :class:`~fugu.env.coding_env.CodingEnvironment`
   are **not supported yet**. The env requires a ``repo_url`` and git-style
   patches. Use SWE-bench (``fugu-collect -d swebench-lite``, etc.) until a
   standalone Python workspace harness exists.

Usage (inspection only)::

    ds = HumanEvalDataset()
    for task in ds:
        print(task.task_id, task.entry_point)
"""

from __future__ import annotations

import logging
from typing import Iterator

from fugu.datasets.base import BaseDataset, CodingTask

logger = logging.getLogger(__name__)


class HumanEvalDataset(BaseDataset):
    """Adapter for the OpenAI HumanEval benchmark.

    Parameters:
        hf_split: The HuggingFace split to load.  Defaults to
            ``"test"`` which is the only split in the dataset.
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
            "Loading HumanEval dataset (split=%s) …", self._hf_split
        )
        self._dataset = load_dataset(
            "openai_humaneval", split=self._hf_split
        )
        logger.info("Loaded %d problems.", len(self._dataset))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "humaneval"

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
            task_id              → task_id
            prompt               → problem_statement  (also starter_code)
            canonical_solution   → gold_patch
            test                 → test_patch
            entry_point          → entry_point
        """
        task_id = str(row.get("task_id", ""))
        prompt = str(row.get("prompt", ""))
        canonical_solution = str(row.get("canonical_solution", ""))
        test_code = str(row.get("test", ""))
        entry_point = str(row.get("entry_point", ""))

        # Build a self-contained problem statement that includes the
        # function signature / docstring from the prompt.
        problem_statement = (
            f"Implement the following Python function:\n\n{prompt}"
        )

        # Collect any remaining fields as metadata
        preserved_keys = {
            "task_id",
            "prompt",
            "canonical_solution",
            "test",
            "entry_point",
        }
        extra_metadata = {
            k: v for k, v in row.items() if k not in preserved_keys
        }

        return CodingTask(
            task_id=task_id,
            problem_statement=problem_statement,
            repo_url=None,
            base_commit=None,
            test_patch=test_code,
            gold_patch=canonical_solution,
            starter_code=prompt,
            entry_point=entry_point,
            metadata=extra_metadata,
        )
