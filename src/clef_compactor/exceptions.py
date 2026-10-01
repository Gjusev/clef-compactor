"""Structured exception hierarchy for clef-compactor.

Every error raised by this library derives from :class:`ClefError`, so callers
can catch a single type and still access structured details such as the HTTP
status code or the Cloudflare request id::

    from clef_compactor.exceptions import ClefError

    try:
        result = compactor.compact(query, chunks)
    except ClefError as exc:
        log.error("compaction failed: %s (request_id=%s)", exc, exc.request_id)
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ClefError",
    "ConfigurationError",
    "ClefAPIError",
    "ClefAuthError",
    "ClefRateLimitError",
    "ClefServerError",
    "ClefResponseError",
    "ClefTimeoutError",
    "ClefNetworkError",
]


class ClefError(Exception):
    """Base class for every error raised by clef-compactor.

    Attributes:
        message: Human-readable description of the failure.
        request_id: Cloudflare request id (``cf-ray`` header), when known.
        retryable: Whether retrying the same request could plausibly succeed.
    """

    retryable: bool = False

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.request_id = request_id

    def __str__(self) -> str:
        if self.request_id:
            return f"{self.message} (request_id={self.request_id})"
        return self.message


class ConfigurationError(ClefError):
    """Raised when configuration is missing or invalid.

    The message lists *all* detected problems at once: unset
    ``CLEF_ACCOUNT_ID`` / ``CLEF_API_TOKEN`` variables, an unsupported model
    name, a non-positive timeout, and so on.
    """


class ClefAPIError(ClefError):
    """Raised when the Cloudflare API returns an error response.

    Attributes:
        status_code: HTTP status code of the response, when the failure came
            from an HTTP status rather than a JSON error envelope.
        request_id: Cloudflare request id, when known.
        retryable: Whether the failure is transient (429/5xx/transport).
        body: Decoded response body, when available.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        retryable: bool | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.status_code = status_code
        # Subclasses declare their default retryability as a class attribute;
        # an explicit argument still wins.
        self.retryable = retryable if retryable is not None else type(self).retryable
        self.body = body


class ClefAuthError(ClefAPIError):
    """Raised on HTTP 401/403: the API token is missing, invalid or lacks scope.

    Retrying with the same credentials will not help.
    """


class ClefRateLimitError(ClefAPIError):
    """Raised on HTTP 429 after retries were exhausted.

    Attributes:
        retry_after: Seconds suggested by the ``Retry-After`` header, when sent.
    """

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ClefServerError(ClefAPIError):
    """Raised on HTTP 5xx after retries were exhausted."""

    retryable = True


class ClefResponseError(ClefAPIError):
    """Raised when the response body does not match the expected Clef schema.

    The HTTP exchange itself succeeded; the payload is unusable (invalid JSON,
    a missing ``result.answers`` object, malformed answers, ...).
    """


class ClefTimeoutError(ClefError):
    """Raised when a request exceeds the configured timeout after retries."""

    retryable = True


class ClefNetworkError(ClefError):
    """Raised when a network-level failure (DNS, TLS, connection reset) survives retries."""

    retryable = True
