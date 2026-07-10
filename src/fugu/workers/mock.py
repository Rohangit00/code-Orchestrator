"""Mock workers for deterministic testing.

:class:`MockWorker` returns a configurable static patch with simulated
latency and optional failure injection.

:class:`RecordedMockWorker` replays a list of pre-recorded
:class:`WorkerResponse` objects in order, which is useful for integration
tests that need precise control over multi-step orchestration.
"""

from __future__ import annotations

import asyncio
import logging

from fugu.workers.base import BaseWorker, WorkerResponse

logger = logging.getLogger(__name__)

# Default patch used when none is supplied to MockWorker
_DEFAULT_PATCH = (
    "--- a/example.py\n"
    "+++ b/example.py\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new\n"
)


class MockWorker(BaseWorker):
    """Worker that returns configurable responses for testing.

    Parameters:
        name: Worker identifier (default ``"mock"``).
        patch: The git diff string to return on each call.
        success: Whether the response should be marked successful.
        latency_ms: Simulated latency in milliseconds (sleeps via
            :func:`asyncio.sleep`).
        tokens_used: Simulated total token count.
        fail_after: If positive, calls beyond this count return an error
            response.  ``-1`` (default) disables failure injection.
    """

    def __init__(
        self,
        name: str = "mock",
        patch: str = _DEFAULT_PATCH,
        success: bool = True,
        latency_ms: float = 100.0,
        tokens_used: int = 500,
        fail_after: int = -1,
    ) -> None:
        super().__init__(name)
        self._patch = patch
        self._success = success
        self._latency_ms = latency_ms
        self._tokens_used = tokens_used
        self._fail_after = fail_after
        self._call_count: int = 0

    async def generate(
        self,
        prompt: str,
        repository_context: str,
        history: list[dict],
        test_results: dict | None = None,
    ) -> WorkerResponse:
        """Return a mock response, optionally simulating latency and failure.

        The internal call counter is incremented **before** the
        ``fail_after`` check so that ``fail_after=N`` means the first *N*
        calls succeed and every subsequent call fails.
        """
        self._call_count += 1

        # Simulate network latency
        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000.0)

        # Failure injection after N successful calls
        if 0 < self._fail_after < self._call_count:
            logger.debug(
                "MockWorker %s failing (call #%d > fail_after=%d)",
                self.name,
                self._call_count,
                self._fail_after,
            )
            return WorkerResponse(
                patch="",
                raw_output="",
                success=False,
                error="Mock failure (injected after call limit)",
                latency_ms=self._latency_ms,
            )

        logger.debug(
            "MockWorker %s returning patch (call #%d)",
            self.name,
            self._call_count,
        )
        return WorkerResponse(
            patch=self._patch,
            raw_output=self._patch,
            explanation="mock explanation",
            tokens_used=self._tokens_used,
            prompt_tokens=self._tokens_used // 2,
            completion_tokens=self._tokens_used - self._tokens_used // 2,
            latency_ms=self._latency_ms,
            success=self._success,
        )

    @property
    def call_count(self) -> int:
        """Number of times :meth:`generate` has been called."""
        return self._call_count

    def reset(self) -> None:
        """Reset the internal call counter."""
        self._call_count = 0

    def __repr__(self) -> str:
        return (
            f"MockWorker(name={self.name!r}, success={self._success}, "
            f"fail_after={self._fail_after}, calls={self._call_count})"
        )


class RecordedMockWorker(BaseWorker):
    """Replays pre-recorded :class:`WorkerResponse` objects in sequence.

    Once all recorded responses have been consumed, subsequent calls return
    an error response.

    Parameters:
        name: Worker identifier.
        responses: Ordered list of responses to replay.
    """

    def __init__(self, name: str, responses: list[WorkerResponse]) -> None:
        super().__init__(name)
        self._responses = list(responses)  # defensive copy
        self._index: int = 0

    async def generate(
        self,
        prompt: str,
        repository_context: str,
        history: list[dict],
        test_results: dict | None = None,
    ) -> WorkerResponse:
        """Return the next recorded response.

        Returns a failure :class:`WorkerResponse` if the sequence is
        exhausted.
        """
        if self._index >= len(self._responses):
            logger.warning(
                "RecordedMockWorker %s exhausted (%d responses)",
                self.name,
                len(self._responses),
            )
            return WorkerResponse(
                patch="",
                raw_output="",
                success=False,
                error=(
                    f"No more recorded responses "
                    f"(replayed {len(self._responses)})"
                ),
            )

        resp = self._responses[self._index]
        logger.debug(
            "RecordedMockWorker %s replaying response %d/%d",
            self.name,
            self._index + 1,
            len(self._responses),
        )
        self._index += 1
        return resp

    @property
    def remaining(self) -> int:
        """Number of responses still available for replay."""
        return max(0, len(self._responses) - self._index)

    def reset(self) -> None:
        """Rewind to the beginning of the recorded sequence."""
        self._index = 0

    def __repr__(self) -> str:
        return (
            f"RecordedMockWorker(name={self.name!r}, "
            f"total={len(self._responses)}, remaining={self.remaining})"
        )
