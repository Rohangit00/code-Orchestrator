"""vLLM worker — calls an OpenAI-compatible chat completions endpoint.

The coding LLMs are served on vLLM (H200 GPUs), same host, different ports.
Each :class:`VLLMWorker` instance targets one vLLM server and lazily creates
a shared :class:`httpx.AsyncClient` for connection pooling.

Typical usage::

    worker = VLLMWorker("qwen", "http://localhost:8001/v1")
    response = await worker.generate(prompt, repo_ctx, history)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from fugu.workers.base import BaseWorker, WorkerResponse

logger = logging.getLogger(__name__)


class VLLMWorker(BaseWorker):
    """Worker backed by a vLLM server with the OpenAI-compatible API.

    Parameters:
        name: Human-readable name (e.g. ``"qwen"``).
        base_url: vLLM server URL including the ``/v1`` prefix,
            e.g. ``"http://localhost:8001/v1"``.
        model: Model name on the server.  When ``None`` the worker
            auto-detects via ``GET /v1/models`` on first call.
        timeout: HTTP request timeout in seconds.
        max_tokens: Maximum number of tokens the model may generate.
        temperature: Sampling temperature (lower → more deterministic).
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str | None = None,
        timeout: float = 120.0,
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(name)
        self.base_url: str = base_url.rstrip("/")
        self.model: str | None = model
        self.timeout: float = timeout
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature
        self._client: httpx.AsyncClient | None = None

    # -- HTTP client lifecycle -----------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """Return the shared async HTTP client, creating it lazily."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Gracefully close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.debug("Closed HTTP client for worker %s", self.name)

    # -- Model auto-detection ------------------------------------------------

    async def _detect_model(self) -> str:
        """Query ``GET /v1/models`` and return the first model id.

        Raises:
            httpx.HTTPStatusError: If the server responds with an error.
            IndexError: If no models are listed.
        """
        client = await self._get_client()
        url = f"{self.base_url}/models"
        logger.debug("Auto-detecting model from %s", url)
        resp = await client.get(url)
        resp.raise_for_status()
        models: list[dict[str, Any]] = resp.json()["data"]
        if not models:
            raise RuntimeError(
                f"No models found on vLLM server at {self.base_url}"
            )
        model_id: str = models[0]["id"]
        logger.info(
            "Worker %s auto-detected model: %s", self.name, model_id
        )
        return model_id

    # -- Core generation -----------------------------------------------------

    async def generate(
        self,
        prompt: str,
        repository_context: str,
        history: list[dict],
        test_results: dict | None = None,
    ) -> WorkerResponse:
        """Call the vLLM chat completions endpoint and return a patch.

        On any network, HTTP, or parsing error the method returns a
        :class:`WorkerResponse` with ``success=False`` and a descriptive
        ``error`` string rather than raising.
        """
        # Auto-detect model name on first invocation
        if not self.model:
            try:
                self.model = await self._detect_model()
            except Exception as exc:
                logger.error(
                    "Worker %s failed to detect model: %s",
                    self.name,
                    exc,
                )
                return WorkerResponse(
                    patch="",
                    raw_output="",
                    success=False,
                    error=f"Model detection failed: {exc}",
                )

        system_msg = self._build_system_prompt()
        user_msg = self._build_user_prompt(
            prompt, repository_context, history, test_results
        )

        client = await self._get_client()
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        start = time.perf_counter()
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except httpx.TimeoutException as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(
                "Worker %s timed out after %.0f ms: %s",
                self.name,
                elapsed,
                exc,
            )
            return WorkerResponse(
                patch="",
                raw_output="",
                success=False,
                error=f"Request timed out after {self.timeout}s: {exc}",
                latency_ms=elapsed,
            )
        except httpx.HTTPStatusError as exc:
            elapsed = (time.perf_counter() - start) * 1000
            body = exc.response.text[:500] if exc.response else ""
            logger.error(
                "Worker %s HTTP %s: %s",
                self.name,
                exc.response.status_code if exc.response else "???",
                body,
            )
            return WorkerResponse(
                patch="",
                raw_output=body,
                success=False,
                error=f"HTTP {exc.response.status_code}: {body}",
                latency_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "Worker %s unexpected error: %s", self.name, exc
            )
            return WorkerResponse(
                patch="",
                raw_output="",
                success=False,
                error=str(exc),
                latency_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000

        # -- Parse response --------------------------------------------------
        try:
            raw: str = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            logger.error(
                "Worker %s malformed response: %s", self.name, exc
            )
            return WorkerResponse(
                patch="",
                raw_output=str(data),
                success=False,
                error=f"Malformed response: {exc}",
                latency_ms=elapsed,
            )

        usage: dict[str, int] = data.get("usage", {})
        patch = self._extract_patch(raw)

        logger.info(
            "Worker %s completed in %.0f ms (%d tokens)",
            self.name,
            elapsed,
            usage.get("total_tokens", 0),
        )

        return WorkerResponse(
            patch=patch,
            raw_output=raw,
            explanation="",
            tokens_used=usage.get("total_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=elapsed,
        )

    # -- Representation -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VLLMWorker(name={self.name!r}, base_url={self.base_url!r}, "
            f"model={self.model!r})"
        )
