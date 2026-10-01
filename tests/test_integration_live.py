"""Live tests against the real Cloudflare API.

Deselected by default. Run them explicitly with real credentials::

    export CLEF_ACCOUNT_ID=... CLEF_API_TOKEN=...
    pytest -m integration
"""

from __future__ import annotations

import os

import pytest

from clef_compactor import AsyncClefCompactor, ClefCompactor

pytestmark = pytest.mark.integration

requires_credentials = pytest.mark.skipif(
    not (os.environ.get("CLEF_ACCOUNT_ID") and os.environ.get("CLEF_API_TOKEN")),
    reason="CLEF_ACCOUNT_ID / CLEF_API_TOKEN not set",
)

CHUNKS = [
    "Our refund policy allows returns within 30 days of purchase.",
    "The Eiffel Tower is located in Paris, France.",
    "Refunds are processed to the original payment method.",
]


@requires_credentials
def test_live_compact_sync() -> None:
    with ClefCompactor() as compactor:
        result = compactor.compact("What is the refund policy?", CHUNKS, token_budget=200)
    assert result.total_output_tokens <= 200
    assert result.kept or result.dropped
    assert result.usage is not None


@requires_credentials
async def test_live_compact_async() -> None:
    async with AsyncClefCompactor(model="clef-flash") as compactor:
        result = await compactor.compact("What is the refund policy?", CHUNKS, token_budget=200)
    assert result.total_output_tokens <= 200
    assert result.model == "clef-flash"
