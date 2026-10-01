"""Shared fixtures: a canned Clef API served by ``httpx.MockTransport``.

No test in the default run touches the real Cloudflare API; the only real-API
tests live in ``test_integration.py`` behind the ``integration`` marker.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from clef_compactor.config import Settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"

ACCOUNT_ID = "test-account"
API_TOKEN = "test-token"
ENDPOINT = (
    "https://api.cloudflare.com/client/v4/accounts/test-account/ai/run/@cf/cloudflare/clef"
)

#: Every configuration variable the library reads, for env isolation.
ENV_VARS = (
    "CLEF_ACCOUNT_ID",
    "CLEF_API_TOKEN",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_API_TOKEN",
    "CLEF_MODEL",
    "CLEF_BASE_URL",
    "CLEF_TIMEOUT",
    "CLEF_MAX_RETRIES",
    "CLEF_LOG_LEVEL",
)


def load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture from ``tests/fixtures``."""
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def noul_answer(probability: float) -> dict[str, Any]:
    """A noul answer object exactly as the Clef API returns it."""
    return {"type": "noul", "noul": probability}


def envelope(answers: dict[str, Any], *, usage: dict[str, int] | None = None) -> dict[str, Any]:
    """Build a Cloudflare success envelope like the real Workers AI response."""
    return {
        "result": {
            "model": "clef",
            "answers": answers,
            "usage": usage or {"input_tokens": 120, "output_tokens": 8},
        },
        "success": True,
        "errors": [],
        "messages": [],
    }


def canned_transport(
    body: dict[str, Any],
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.MockTransport:
    """Transport that always answers with one canned response."""
    return httpx.MockTransport(
        lambda request: httpx.Response(
            status_code,
            json=body,
            headers={"cf-ray": "test-ray-001", **(headers or {})},
        )
    )


def scripted_transport(
    handler: Callable[[httpx.Request, list[httpx.Request]], httpx.Response],
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """Transport backed by a handler; records every request for assertions."""
    requests: list[httpx.Request] = []

    def wrapper(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request, requests)

    return httpx.MockTransport(wrapper), requests


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from credentials or overrides set on the host machine."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real backoff sleeps in sync retry tests."""
    monkeypatch.setattr("clef_compactor.client.time.sleep", lambda seconds: None)


@pytest.fixture
def no_async_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real backoff sleeps in async retry tests."""

    async def _instant(seconds: float) -> None:
        return None

    monkeypatch.setattr("clef_compactor.client.asyncio.sleep", _instant)


@pytest.fixture
def settings() -> Settings:
    """Valid settings pointing at a fake account."""
    return Settings(
        account_id=ACCOUNT_ID,
        api_token=API_TOKEN,
        timeout=5.0,
        max_retries=1,
    )
