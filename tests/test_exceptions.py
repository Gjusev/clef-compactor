"""Tests for the structured exception hierarchy."""

from __future__ import annotations

import pytest

from clef_compactor.exceptions import (
    ClefAPIError,
    ClefAuthError,
    ClefError,
    ClefNetworkError,
    ClefRateLimitError,
    ClefResponseError,
    ClefServerError,
    ClefTimeoutError,
    ConfigurationError,
)


@pytest.mark.parametrize(
    ("exc_type", "retryable"),
    [
        (ClefAuthError, False),
        (ClefRateLimitError, True),
        (ClefServerError, True),
        (ClefResponseError, False),
    ],
)
def test_api_error_hierarchy(exc_type: type[ClefAPIError], retryable: bool) -> None:
    error = exc_type("boom", status_code=500, request_id="ray-1")
    assert isinstance(error, ClefAPIError)
    assert isinstance(error, ClefError)
    assert isinstance(error, Exception)
    assert error.status_code == 500
    assert error.request_id == "ray-1"
    assert error.retryable is retryable


@pytest.mark.parametrize("exc_type", [ClefTimeoutError, ClefNetworkError])
def test_transport_error_hierarchy(exc_type: type[ClefError]) -> None:
    error = exc_type("boom", request_id="ray-2")
    assert isinstance(error, ClefError)
    assert not isinstance(error, ClefAPIError)
    assert error.request_id == "ray-2"
    assert error.retryable is True


def test_base_error_defaults() -> None:
    error = ClefError("simple failure")
    assert error.request_id is None
    assert error.retryable is False
    assert str(error) == "simple failure"


def test_str_includes_request_id() -> None:
    error = ClefError("simple failure", request_id="ray-42")
    assert str(error) == "simple failure (request_id=ray-42)"


def test_api_error_body_attribute() -> None:
    error = ClefAPIError("bad", status_code=400, body={"errors": []})
    assert error.body == {"errors": []}
    assert error.retryable is False


def test_rate_limit_retry_after() -> None:
    error = ClefRateLimitError("slow down", status_code=429, retry_after=12.0)
    assert error.retry_after == 12.0
    assert error.retryable is True


def test_configuration_error_is_clef_error() -> None:
    error = ConfigurationError("bad config")
    assert isinstance(error, ClefError)
    assert not isinstance(error, ClefAPIError)
