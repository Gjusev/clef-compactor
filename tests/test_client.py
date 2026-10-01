"""Tests for the sync and async Clef HTTP clients (mocked transport only)."""

from __future__ import annotations

import json

import httpx
import pytest

from clef_compactor.client import (
    RETRYABLE_STATUS_CODES,
    AsyncClefClient,
    ClefClient,
    _sleep_seconds,
)
from clef_compactor.config import Settings
from clef_compactor.exceptions import (
    ClefAPIError,
    ClefAuthError,
    ClefNetworkError,
    ClefRateLimitError,
    ClefResponseError,
    ClefServerError,
    ClefTimeoutError,
)
from clef_compactor.models import TokenUsage
from conftest import canned_transport, envelope, noul_answer, scripted_transport

QUESTIONS = {"chunk_1": {"type": "noul", "instructions": "needed?", "criteria": {}}}


def test_retryable_status_codes_are_transient() -> None:
    assert {408, 429, 500, 502, 503, 504, 529} <= RETRYABLE_STATUS_CODES


class TestBackoff:
    def test_retry_after_header_wins(self) -> None:
        assert _sleep_seconds(0, 7.0) == 7.0
        assert _sleep_seconds(3, 0.0) == 0.0

    def test_exponential_growth_with_jitter_is_capped(self) -> None:
        for attempt in range(6):
            value = _sleep_seconds(attempt, None)
            cap = min(0.5 * 2**attempt, 8.0)
            assert cap / 2 <= value <= cap


class TestClefClient:
    def test_success_reply_fields(self, settings: Settings) -> None:
        client = ClefClient(settings, transport=canned_transport(envelope({"chunk_1": noul_answer(0.75)})))
        reply = client.ask("state", QUESTIONS)
        assert reply.model == "clef"
        assert reply.answers == {"chunk_1": noul_answer(0.75)}
        assert reply.usage == TokenUsage(input_tokens=120, output_tokens=8)
        assert reply.request_id == "test-ray-001"
        assert reply.latency_ms >= 0.0
        client.close()

    def test_request_payload_and_auth_header(self, settings: Settings) -> None:
        transport, requests = scripted_transport(lambda req, seen: httpx.Response(200, json=envelope({})))
        client = ClefClient(settings, transport=transport)
        client.ask("the state", QUESTIONS, model="clef-flash", images=["data:image/png;base64,AAA"])
        assert len(requests) == 1
        request = requests[0]
        assert request.url == settings.endpoint == (
            "https://api.cloudflare.com/client/v4/accounts/test-account/ai/run/@cf/cloudflare/clef"
        )
        assert request.headers["Authorization"] == "Bearer test-token"
        body = json.loads(request.content)
        assert body == {
            "model": "clef-flash",
            "state": "the state",
            "questions": QUESTIONS,
            "images": ["data:image/png;base64,AAA"],
        }
        client.close()

    def test_error_envelope_maps_to_api_error(self, settings: Settings) -> None:
        error_body = {"success": False, "errors": [{"code": 10000, "message": "Invalid API token"}], "result": None}
        client = ClefClient(settings, transport=canned_transport(error_body))
        with pytest.raises(ClefAPIError) as excinfo:
            client.ask("state", QUESTIONS)
        assert "Invalid API token" in str(excinfo.value)
        assert excinfo.value.request_id == "test-ray-001"
        assert excinfo.value.retryable is False
        client.close()

    def test_401_maps_to_auth_error(self, settings: Settings) -> None:
        client = ClefClient(settings, transport=canned_transport({"error": "no"}, status_code=401))
        with pytest.raises(ClefAuthError):
            client.ask("state", QUESTIONS)
        client.close()

    def test_404_is_not_retryable(self, settings: Settings, no_sleep: None) -> None:
        transport, requests = scripted_transport(lambda req, seen: httpx.Response(404, json={}))
        client = ClefClient(settings, transport=transport)
        with pytest.raises(ClefAPIError) as excinfo:
            client.ask("state", QUESTIONS)
        assert excinfo.value.status_code == 404
        assert excinfo.value.retryable is False
        assert len(requests) == 1
        client.close()

    def test_429_honors_retry_after_then_succeeds(self, settings: Settings, no_sleep: None) -> None:
        def handler(request: httpx.Request, seen: list[httpx.Request]) -> httpx.Response:
            if len(seen) == 1:
                return httpx.Response(429, json={}, headers={"Retry-After": "7"})
            return httpx.Response(200, json=envelope({"chunk_1": noul_answer(0.5)}))

        transport, requests = scripted_transport(handler)
        client = ClefClient(settings, transport=transport)
        reply = client.ask("state", QUESTIONS)
        assert reply.answers["chunk_1"]["noul"] == 0.5
        assert len(requests) == 2
        client.close()

    def test_429_sleeps_for_retry_after_seconds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr("clef_compactor.client.time.sleep", sleeps.append)
        transport, _ = scripted_transport(
            lambda req, seen: httpx.Response(429, json={}, headers={"Retry-After": "3.5"})
        )
        retry_settings = Settings(account_id="a", api_token="t", max_retries=1)
        client = ClefClient(retry_settings, transport=transport)
        with pytest.raises(ClefRateLimitError) as excinfo:
            client.ask("state", QUESTIONS)
        assert excinfo.value.retry_after == 3.5
        assert sleeps == [3.5]
        client.close()

    def test_429_exhausts_retries(self, settings: Settings, no_sleep: None) -> None:
        transport, requests = scripted_transport(lambda req, seen: httpx.Response(429, json={}))
        client = ClefClient(settings, transport=transport)
        with pytest.raises(ClefRateLimitError) as excinfo:
            client.ask("state", QUESTIONS)
        assert excinfo.value.retryable is True
        assert len(requests) == settings.max_retries + 1
        client.close()

    def test_500_exhausts_retries(self, no_sleep: None) -> None:
        transport, requests = scripted_transport(lambda req, seen: httpx.Response(500, json={}))
        retry_settings = Settings(account_id="a", api_token="t", max_retries=2)
        client = ClefClient(retry_settings, transport=transport)
        with pytest.raises(ClefServerError):
            client.ask("state", QUESTIONS)
        assert len(requests) == 3
        client.close()

    def test_500_then_success_recovers(self, settings: Settings, no_sleep: None) -> None:
        def handler(request: httpx.Request, seen: list[httpx.Request]) -> httpx.Response:
            if len(seen) == 1:
                return httpx.Response(503, json={})
            return httpx.Response(200, json=envelope({}))

        transport, requests = scripted_transport(handler)
        client = ClefClient(settings, transport=transport)
        assert client.ask("state", QUESTIONS).answers == {}
        assert len(requests) == 2
        client.close()

    def test_timeout_maps_to_timeout_error(self, settings: Settings, no_sleep: None) -> None:
        def handler(request: httpx.Request, seen: list[httpx.Request]) -> httpx.Response:
            raise httpx.ConnectTimeout("too slow", request=request)

        transport, _ = scripted_transport(handler)
        client = ClefClient(settings, transport=transport)
        with pytest.raises(ClefTimeoutError):
            client.ask("state", QUESTIONS)
        client.close()

    def test_network_failure_maps_to_network_error(self, settings: Settings, no_sleep: None) -> None:
        def handler(request: httpx.Request, seen: list[httpx.Request]) -> httpx.Response:
            raise httpx.ConnectError("connection reset", request=request)

        transport, _ = scripted_transport(handler)
        client = ClefClient(settings, transport=transport)
        with pytest.raises(ClefNetworkError):
            client.ask("state", QUESTIONS)
        client.close()

    def test_invalid_json_is_response_error(self, settings: Settings) -> None:
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text="<html>oops</html>"))
        client = ClefClient(settings, transport=transport)
        with pytest.raises(ClefResponseError):
            client.ask("state", QUESTIONS)
        client.close()

    def test_missing_answers_is_response_error(self, settings: Settings) -> None:
        client = ClefClient(settings, transport=canned_transport({"result": {}, "success": True}))
        with pytest.raises(ClefResponseError):
            client.ask("state", QUESTIONS)
        client.close()

    def test_non_object_payload_is_response_error(self, settings: Settings) -> None:
        client = ClefClient(settings, transport=canned_transport([1, 2, 3]))
        with pytest.raises(ClefResponseError):
            client.ask("state", QUESTIONS)
        client.close()

    def test_empty_questions_rejected(self, settings: Settings) -> None:
        client = ClefClient(settings, transport=canned_transport(envelope({})))
        with pytest.raises(ValueError):
            client.ask("state", {})
        client.close()

    def test_context_manager_closes(self, settings: Settings) -> None:
        with ClefClient(settings, transport=canned_transport(envelope({}))) as client:
            client.ask("state", QUESTIONS)
        with pytest.raises(RuntimeError):
            client.ask("state", QUESTIONS)


class TestAsyncClefClient:
    async def test_success_reply_fields(self, settings: Settings) -> None:
        client = AsyncClefClient(settings, transport=canned_transport(envelope({"chunk_1": noul_answer(0.75)})))
        reply = await client.ask("state", QUESTIONS)
        assert reply.answers["chunk_1"]["noul"] == 0.75
        assert reply.usage == TokenUsage(input_tokens=120, output_tokens=8)
        await client.aclose()

    async def test_429_retries_then_succeeds(self, no_async_sleep: None) -> None:
        def handler(request: httpx.Request, seen: list[httpx.Request]) -> httpx.Response:
            if len(seen) == 1:
                return httpx.Response(429, json={}, headers={"Retry-After": "1"})
            return httpx.Response(200, json=envelope({}))

        transport, requests = scripted_transport(handler)
        retry_settings = Settings(account_id="a", api_token="t", max_retries=1)
        client = AsyncClefClient(retry_settings, transport=transport)
        reply = await client.ask("state", QUESTIONS)
        assert reply.answers == {}
        assert len(requests) == 2
        await client.aclose()

    async def test_timeout_maps_to_timeout_error(self, settings: Settings, no_async_sleep: None) -> None:
        def handler(request: httpx.Request, seen: list[httpx.Request]) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        transport, _ = scripted_transport(handler)
        client = AsyncClefClient(settings, transport=transport)
        with pytest.raises(ClefTimeoutError):
            await client.ask("state", QUESTIONS)
        await client.aclose()

    async def test_401_raises_without_retry(self, settings: Settings) -> None:
        transport, requests = scripted_transport(lambda req, seen: httpx.Response(403, json={}))
        client = AsyncClefClient(settings, transport=transport)
        with pytest.raises(ClefAuthError):
            await client.ask("state", QUESTIONS)
        assert len(requests) == 1
        await client.aclose()

    async def test_async_context_manager(self, settings: Settings) -> None:
        async with AsyncClefClient(settings, transport=canned_transport(envelope({}))) as client:
            await client.ask("state", QUESTIONS)
        with pytest.raises(RuntimeError):
            await client.ask("state", QUESTIONS)
