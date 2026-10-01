"""HTTP clients for the Cloudflare Workers AI Clef endpoint.

Provides :class:`ClefClient` (sync) and :class:`AsyncClefClient` (async) with:

* automatic retries with exponential backoff and jitter for transient errors
  (HTTP 429/5xx, timeouts, network errors), honouring ``Retry-After``
* configurable timeouts
* structured errors (see :mod:`clef_compactor.exceptions`)
* logging through the standard ``clef_compactor`` logger

Both clients accept an ``httpx.MockTransport`` via ``transport=`` for testing,
and both are context managers.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import httpx

from .config import Settings
from .exceptions import (
    ClefAPIError,
    ClefAuthError,
    ClefNetworkError,
    ClefRateLimitError,
    ClefResponseError,
    ClefServerError,
    ClefTimeoutError,
)
from .models import ClefReply, TokenUsage

__all__ = ["ClefClient", "AsyncClefClient", "RETRYABLE_STATUS_CODES"]

logger = logging.getLogger("clef_compactor")

#: HTTP status codes that are worth retrying with backoff.
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504, 529})

_BASE_BACKOFF_SECONDS = 0.5
_BACKOFF_CAP_SECONDS = 8.0


def _request_id(response: httpx.Response) -> str | None:
    """Best-effort extraction of Cloudflare's request id."""
    return response.headers.get("cf-ray") or response.headers.get("x-request-id")


def _api_error(response: httpx.Response) -> ClefAPIError:
    """Map an error HTTP response to the most specific exception type."""
    request_id = _request_id(response)
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text

    if response.status_code in (401, 403):
        return ClefAuthError(
            f"Authentication failed with HTTP {response.status_code}; check "
            f"CLEF_API_TOKEN and its Workers AI permissions.",
            status_code=response.status_code,
            request_id=request_id,
            body=body,
        )
    if response.status_code == 429:
        retry_after: float | None = None
        raw_retry = response.headers.get("retry-after")
        if raw_retry is not None:
            try:
                retry_after = float(raw_retry)
            except ValueError:
                retry_after = None
        return ClefRateLimitError(
            "Rate limited by Cloudflare (HTTP 429) after all retries.",
            status_code=response.status_code,
            request_id=request_id,
            body=body,
            retry_after=retry_after,
        )
    if response.status_code >= 500:
        return ClefServerError(
            f"Cloudflare server error HTTP {response.status_code} after all retries.",
            status_code=response.status_code,
            request_id=request_id,
            body=body,
        )
    return ClefAPIError(
        f"Cloudflare API error HTTP {response.status_code}.",
        status_code=response.status_code,
        request_id=request_id,
        body=body,
    )


def _sleep_seconds(attempt: int, retry_after: float | None) -> float:
    """Exponential backoff with equal jitter, honouring ``Retry-After``."""
    if retry_after is not None:
        return max(retry_after, 0.0)
    raw = min(_BASE_BACKOFF_SECONDS * (2**attempt), _BACKOFF_CAP_SECONDS)
    return raw / 2 + random.uniform(0, raw / 2)


def _decode_json(response: httpx.Response, request_id: str | None) -> Any:
    """Decode the response body, raising :class:`ClefResponseError` on bad JSON."""
    try:
        return response.json()
    except ValueError as exc:
        raise ClefResponseError(
            f"Response body is not valid JSON: {exc}",
            request_id=request_id,
            body=response.text,
        ) from exc


def _parse_reply(payload: Any, *, model: str, latency_ms: float, request_id: str | None) -> ClefReply:
    """Validate the Cloudflare envelope and build a :class:`ClefReply`."""
    if not isinstance(payload, dict):
        raise ClefResponseError(
            "Expected a JSON object from the API.",
            request_id=request_id,
            body=payload,
        )
    if payload.get("success") is False:
        errors = payload.get("errors") or []
        detail = "; ".join(
            err.get("message", str(err)) if isinstance(err, dict) else str(err) for err in errors
        ) or "unknown error"
        raise ClefAPIError(
            f"Cloudflare API rejected the request: {detail}.",
            request_id=request_id,
            body=payload,
        )
    result = payload.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise ClefResponseError(
            "Response is missing the expected 'result.answers' object.",
            request_id=request_id,
            body=payload,
        )
    return ClefReply(
        model=str(result.get("model", model)),
        answers=result["answers"],
        usage=TokenUsage.from_api(result.get("usage")),
        request_id=request_id,
        latency_ms=latency_ms,
    )


def _build_payload(
    settings: Settings,
    state: str | dict[str, Any],
    questions: dict[str, dict[str, Any]],
    images: list[str] | None,
    model: str | None,
) -> dict[str, Any]:
    """Assemble the Clef request body."""
    if not questions:
        raise ValueError("questions must contain at least one entry")
    payload: dict[str, Any] = {
        "model": model or settings.model,
        "state": state,
        "questions": questions,
    }
    if images:
        payload["images"] = images
    return payload


class ClefClient:
    """Synchronous client for the Clef decision-model endpoint.

    Args:
        settings: Validated :class:`~clef_compactor.config.Settings`.
        transport: Optional ``httpx.BaseTransport`` (e.g. ``MockTransport``)
            injected in tests; ignored once ``http_client`` is given.
        http_client: Optional pre-built ``httpx.Client`` to use instead of
            constructing one from settings.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self._client = http_client or httpx.Client(
            timeout=settings.timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {settings.api_token}"},
        )

    def ask(
        self,
        state: str | dict[str, Any],
        questions: dict[str, dict[str, Any]],
        *,
        images: list[str] | None = None,
        model: str | None = None,
    ) -> ClefReply:
        """Run one decision request and return the normalized reply.

        Args:
            state: The content to evaluate (text or structured data).
            questions: Clef question schema (1..64 typed questions).
            images: Optional inline base64 PNG/JPEG/WebP images (max 4).
            model: Override the model selector for this call only.

        Raises:
            ClefAuthError: On HTTP 401/403.
            ClefRateLimitError: On HTTP 429 after retries.
            ClefServerError: On HTTP 5xx after retries.
            ClefAPIError: On other API errors or error envelopes.
            ClefResponseError: On malformed success payloads.
            ClefTimeoutError: On timeouts after retries.
            ClefNetworkError: On connection failures after retries.
        """
        payload = _build_payload(self.settings, state, questions, images, model)
        attempts = self.settings.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                response = self._client.post(self.settings.endpoint, json=payload)
            except httpx.TimeoutException:
                last_error = ClefTimeoutError(
                    f"Request timed out after {self.settings.timeout}s "
                    f"(attempt {attempt + 1}/{attempts})."
                )
                logger.warning("%s", last_error)
            except httpx.TransportError as exc:
                last_error = ClefNetworkError(
                    f"Connection to Cloudflare failed: {exc} "
                    f"(attempt {attempt + 1}/{attempts})."
                )
                logger.warning("%s", last_error)
            else:
                latency_ms = (time.perf_counter() - started) * 1000
                request_id = _request_id(response)
                if response.is_success:
                    reply = _parse_reply(
                        _decode_json(response, request_id),
                        model=payload["model"],
                        latency_ms=latency_ms,
                        request_id=request_id,
                    )
                    logger.debug(
                        "clef %s answered %d question(s) in %.1f ms",
                        payload["model"],
                        len(questions),
                        latency_ms,
                    )
                    return reply
                error = _api_error(response)
                if response.status_code not in RETRYABLE_STATUS_CODES or attempt == attempts - 1:
                    logger.warning("Giving up after HTTP %s: %s", response.status_code, error)
                    raise error
                last_error = error
                logger.warning(
                    "Retryable HTTP %s (attempt %d/%d)",
                    response.status_code,
                    attempt + 1,
                    attempts,
                )
            if attempt < attempts - 1:
                time.sleep(_sleep_seconds(attempt, getattr(last_error, "retry_after", None)))
        raise last_error  # pragma: no cover - the loop always raises or returns

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> ClefClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class AsyncClefClient:
    """Async counterpart of :class:`ClefClient` built on ``httpx.AsyncClient``.

    Args:
        settings: Validated :class:`~clef_compactor.config.Settings`.
        transport: Optional ``httpx.AsyncBaseTransport`` for tests.
        http_client: Optional pre-built ``httpx.AsyncClient``.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self._client = http_client or httpx.AsyncClient(
            timeout=settings.timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {settings.api_token}"},
        )

    async def ask(
        self,
        state: str | dict[str, Any],
        questions: dict[str, dict[str, Any]],
        *,
        images: list[str] | None = None,
        model: str | None = None,
    ) -> ClefReply:
        """Async version of :meth:`ClefClient.ask`."""
        payload = _build_payload(self.settings, state, questions, images, model)
        attempts = self.settings.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                response = await self._client.post(self.settings.endpoint, json=payload)
            except httpx.TimeoutException:
                last_error = ClefTimeoutError(
                    f"Request timed out after {self.settings.timeout}s "
                    f"(attempt {attempt + 1}/{attempts})."
                )
                logger.warning("%s", last_error)
            except httpx.TransportError as exc:
                last_error = ClefNetworkError(
                    f"Connection to Cloudflare failed: {exc} "
                    f"(attempt {attempt + 1}/{attempts})."
                )
                logger.warning("%s", last_error)
            else:
                latency_ms = (time.perf_counter() - started) * 1000
                request_id = _request_id(response)
                if response.is_success:
                    reply = _parse_reply(
                        _decode_json(response, request_id),
                        model=payload["model"],
                        latency_ms=latency_ms,
                        request_id=request_id,
                    )
                    logger.debug(
                        "clef %s answered %d question(s) in %.1f ms",
                        payload["model"],
                        len(questions),
                        latency_ms,
                    )
                    return reply
                error = _api_error(response)
                if response.status_code not in RETRYABLE_STATUS_CODES or attempt == attempts - 1:
                    logger.warning("Giving up after HTTP %s: %s", response.status_code, error)
                    raise error
                last_error = error
                logger.warning(
                    "Retryable HTTP %s (attempt %d/%d)",
                    response.status_code,
                    attempt + 1,
                    attempts,
                )
            if attempt < attempts - 1:
                await asyncio.sleep(_sleep_seconds(attempt, getattr(last_error, "retry_after", None)))
        raise last_error  # pragma: no cover - the loop always raises or returns

    async def aclose(self) -> None:
        """Release the underlying async HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> AsyncClefClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
