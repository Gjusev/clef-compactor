"""Offline smoke tests: the public surface works end-to-end without any API."""

from __future__ import annotations

import clef_compactor
from clef_compactor import AsyncClefClient, AsyncClefCompactor, ClefClient, ClefCompactor
from conftest import envelope, noul_answer


def test_package_exposes_version() -> None:
    from importlib.metadata import version

    assert clef_compactor.__version__ == version("clef-compactor")


def test_public_api_surface() -> None:
    for name in (
        "ClefCompactor",
        "AsyncClefCompactor",
        "ClefClient",
        "AsyncClefClient",
        "CompactResult",
        "ScoredChunk",
        "TokenUsage",
        "Settings",
        "ClefError",
        "ConfigurationError",
        "count_tokens",
    ):
        assert hasattr(clef_compactor, name), f"missing export: {name}"


def test_offline_compact_end_to_end() -> None:
    """Full compaction against a canned envelope; zero network access."""
    import httpx

    answers = {
        "chunk_1": noul_answer(0.9),
        "chunk_2": noul_answer(0.1),
        "chunk_3": noul_answer(0.8),
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=envelope(answers)))
    from clef_compactor.config import Settings

    client = ClefClient(
        Settings(account_id="a", api_token="t", max_retries=0), transport=transport
    )
    compactor = ClefCompactor(account_id="a", api_token="t", client=client)
    result = compactor.compact(
        "What is the refund policy?",
        [
            "Refunds are accepted within 30 days.",
            "The Eiffel Tower is in Paris.",
            "Refunds go to the original payment method.",
        ],
        token_budget=100,
    )
    assert [chunk.index for chunk in result.kept] == [0, 2]
    assert result.total_output_tokens <= 100
    assert result.saved_fraction > 0
    assert result.usage is not None
    assert result.cost_estimate >= 0.0
    compactor.close()


def test_async_exports_exist() -> None:
    assert AsyncClefCompactor is not None
    assert AsyncClefClient is not None
