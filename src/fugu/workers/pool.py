"""Worker pool — maps planner actions to worker instances.

The :class:`WorkerPool` is the single point of dispatch: given a
:class:`~fugu.core.actions.PlannerAction` the pool returns the correct
:class:`~fugu.workers.base.BaseWorker`.

Typical usage::

    pool = WorkerPool.from_config(config.worker)
    worker = pool.get_worker(PlannerAction.CALL_QWEN)
    response = await worker.generate(...)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fugu.core.actions import WORKER_ACTIONS, PlannerAction
from fugu.workers.base import BaseWorker

if TYPE_CHECKING:
    from fugu.config import WorkerConfig

logger = logging.getLogger(__name__)


class WorkerPool:
    """Registry that maps :class:`PlannerAction` worker calls to concrete workers.

    Only actions in :data:`~fugu.core.actions.WORKER_ACTIONS` may be
    registered.  Attempting to register a non-worker action raises
    :exc:`ValueError`.
    """

    def __init__(self) -> None:
        self._workers: dict[PlannerAction, BaseWorker] = {}

    # -- Registration --------------------------------------------------------

    def register(self, action: PlannerAction, worker: BaseWorker) -> None:
        """Bind *action* to *worker*.

        Args:
            action: Must be one of ``CALL_QWEN``, ``CALL_GEMMA``, or
                ``CALL_ORNITH``.
            worker: The :class:`BaseWorker` instance to dispatch to.

        Raises:
            ValueError: If *action* is not a worker action.
        """
        if action not in WORKER_ACTIONS:
            raise ValueError(
                f"{action.name} is not a worker action.  "
                f"Valid worker actions: "
                f"{', '.join(a.name for a in WORKER_ACTIONS)}"
            )
        self._workers[action] = worker
        logger.info(
            "Registered worker %s for action %s",
            worker.name,
            action.name,
        )

    # -- Lookup --------------------------------------------------------------

    def get_worker(self, action: PlannerAction) -> BaseWorker:
        """Return the worker registered for *action*.

        Raises:
            KeyError: If no worker has been registered for *action*.
        """
        try:
            return self._workers[action]
        except KeyError:
            registered = ", ".join(a.name for a in self._workers) or "(none)"
            raise KeyError(
                f"No worker registered for {action.name}.  "
                f"Registered actions: {registered}"
            ) from None

    def has_worker(self, action: PlannerAction) -> bool:
        """Return ``True`` if a worker is registered for *action*."""
        return action in self._workers

    # -- Factory methods -----------------------------------------------------

    @classmethod
    def from_config(cls, config: WorkerConfig) -> WorkerPool:
        """Create a production pool from a :class:`WorkerConfig`.

        Instantiates one :class:`~fugu.workers.vllm.VLLMWorker` per
        configured endpoint (Qwen, Gemma, Ornith).
        """
        # Local import to avoid circular dependency at module level
        from fugu.workers.vllm import VLLMWorker

        pool = cls()
        pool.register(
            PlannerAction.CALL_QWEN,
            VLLMWorker(
                name="qwen",
                base_url=config.qwen_url,
                timeout=config.timeout,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            ),
        )
        pool.register(
            PlannerAction.CALL_GEMMA,
            VLLMWorker(
                name="gemma",
                base_url=config.gemma_url,
                timeout=config.timeout,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            ),
        )
        pool.register(
            PlannerAction.CALL_ORNITH,
            VLLMWorker(
                name="ornith",
                base_url=config.ornith_url,
                timeout=config.timeout,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            ),
        )
        logger.info(
            "Created worker pool from config with %d workers",
            len(pool._workers),
        )
        return pool

    @classmethod
    def mock(cls) -> WorkerPool:
        """Create a pool populated with :class:`MockWorker` instances.

        Useful for unit and integration tests that do not require real
        vLLM servers.
        """
        from fugu.workers.mock import MockWorker

        pool = cls()
        pool.register(PlannerAction.CALL_QWEN, MockWorker("mock_qwen"))
        pool.register(PlannerAction.CALL_GEMMA, MockWorker("mock_gemma"))
        pool.register(PlannerAction.CALL_ORNITH, MockWorker("mock_ornith"))
        logger.info("Created mock worker pool")
        return pool

    # -- Introspection -------------------------------------------------------

    @property
    def available_workers(self) -> list[str]:
        """Return human-readable names of all registered workers."""
        return [w.name for w in self._workers.values()]

    @property
    def registered_actions(self) -> list[PlannerAction]:
        """Return all actions that have a worker registered."""
        return list(self._workers.keys())

    def __len__(self) -> int:
        return len(self._workers)

    def __contains__(self, action: PlannerAction) -> bool:
        return action in self._workers

    # -- Lifecycle -----------------------------------------------------------

    async def close_all(self) -> None:
        """Close all workers that expose a ``close()`` coroutine.

        Safe to call multiple times.
        """
        for action, worker in self._workers.items():
            if hasattr(worker, "close") and callable(worker.close):
                try:
                    await worker.close()
                    logger.debug(
                        "Closed worker %s (%s)", worker.name, action.name
                    )
                except Exception as exc:
                    logger.warning(
                        "Error closing worker %s: %s", worker.name, exc
                    )

    # -- Representation ------------------------------------------------------

    def __repr__(self) -> str:
        workers_repr = ", ".join(
            f"{a.name}: {w.name}" for a, w in self._workers.items()
        )
        return f"WorkerPool({{{workers_repr}}})"
