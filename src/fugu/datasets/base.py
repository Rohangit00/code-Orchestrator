"""Base dataset abstractions for the Fugu orchestrator.

:class:`CodingTask` is the universal task descriptor; every dataset
adapter converts its native records into ``CodingTask`` instances.

:class:`BaseDataset` provides iteration, slicing, and grouping helpers
that concrete adapters inherit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class CodingTask:
    """A single coding task / issue to be solved.

    Fields cover the union of metadata needed by SWE-bench,
    HumanEval, MBPP, and custom benchmarks.

    Attributes:
        task_id: Unique identifier for this task.
        problem_statement: Natural-language description of the issue
            or function to implement.
        repo_url: Git clone URL (``None`` for standalone tasks).
        base_commit: Commit hash to check out before patching.
        test_patch: A diff that introduces the regression tests
            (SWE-bench style).
        test_command: Shell command to run the test suite.
        gold_patch: Reference solution patch / code.
        fail_to_pass: Test names that should *start* failing and
            later pass after the fix is applied.
        pass_to_pass: Test names that must continue to pass.
        starter_code: Partial code / function signature provided to
            the model (HumanEval-style).
        entry_point: Name of the function under test (HumanEval/MBPP).
        metadata: Catch-all for dataset-specific extras.
    """

    task_id: str
    problem_statement: str
    repo_url: str | None = None
    base_commit: str | None = None
    test_patch: str | None = None
    test_command: str | None = None
    gold_patch: str | None = None
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    starter_code: str | None = None
    entry_point: str | None = None
    metadata: dict = field(default_factory=dict)


class BaseDataset(ABC):
    """Abstract base class for benchmark dataset adapters.

    Concrete sub-classes must implement :pyattr:`name`,
    :pyattr:`size`, and :meth:`__iter__`.  Higher-level helpers
    (``group_by_repo``, ``take``, ``filter``) are provided for free.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name (e.g. ``"swebench_lite"``)."""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Total number of tasks in the dataset."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[CodingTask]:
        """Yield :class:`CodingTask` instances, one per record."""
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return :pyattr:`size`."""
        return self.size

    def group_by_repo(self) -> dict[str, list[CodingTask]]:
        """Group tasks by their ``repo_url``.

        Tasks without a ``repo_url`` are grouped under the key
        ``"__no_repo__"``.

        Returns:
            A dict mapping repo URL strings to lists of tasks.
        """
        groups: dict[str, list[CodingTask]] = defaultdict(list)
        for task in self:
            key = task.repo_url or "__no_repo__"
            groups[key].append(task)
        return dict(groups)

    def take(self, n: int) -> list[CodingTask]:
        """Return the first *n* tasks as a list.

        Args:
            n: Maximum number of tasks to return.  If the dataset has
               fewer than *n* tasks, all tasks are returned.

        Returns:
            A list of at most *n* :class:`CodingTask` instances.
        """
        tasks: list[CodingTask] = []
        for task in self:
            tasks.append(task)
            if len(tasks) >= n:
                break
        return tasks

    def filter(self, predicate) -> list[CodingTask]:
        """Return tasks matching *predicate*.

        Args:
            predicate: A callable ``(CodingTask) -> bool``.

        Returns:
            A list of matching tasks.
        """
        return [task for task in self if predicate(task)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, size={self.size})"
