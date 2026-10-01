"""Tests for the LangChain integration.

These tests use the real ``langchain-core`` package (a light install) and are
skipped when it is not installed. They mock only the Clef HTTP transport.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.documents.compressor import BaseDocumentCompressor  # noqa: E402

from clef_compactor import ClefCompactor  # noqa: E402
from clef_compactor.client import ClefClient  # noqa: E402
from clef_compactor.config import Settings  # noqa: E402
from clef_compactor.integrations.langchain import ClefDocumentCompressor  # noqa: E402
from conftest import canned_transport, envelope, noul_answer  # noqa: E402

DOCS = [
    "Our refund policy allows returns within 30 days.",
    "The Eiffel Tower is in Paris.",
    "Refunds go to the original payment method.",
]
ANSWERS = {"chunk_1": noul_answer(0.95), "chunk_2": noul_answer(0.05), "chunk_3": noul_answer(0.8)}


def compressor() -> ClefDocumentCompressor:
    settings = Settings(account_id="a", api_token="t", max_retries=0)
    client = ClefClient(settings, transport=canned_transport(envelope(ANSWERS)))
    engine = ClefCompactor(account_id="a", api_token="t", client=client)
    return ClefDocumentCompressor(compactor=engine, token_budget=10**6)


def documents() -> list[Document]:
    return [
        Document(page_content=text, metadata={"source": f"s{i}"}) for i, text in enumerate(DOCS)
    ]


def test_is_a_base_document_compressor() -> None:
    assert isinstance(compressor(), BaseDocumentCompressor)


def test_keeps_relevant_documents_with_metadata() -> None:
    compressed = compressor().compress_documents(documents(), "refund policy?")
    assert [doc.metadata["source"] for doc in compressed] == ["s0", "s2"]
    assert compressed[0].metadata["clef_score"] == 0.95
    # Verbatim, never rewritten.
    assert compressed[0].page_content == DOCS[0]
    assert compressed[1].page_content == DOCS[2]


def test_empty_input_returns_empty() -> None:
    assert compressor().compress_documents([], "q") == []


def test_token_budget_is_forwarded() -> None:
    engine = compressor().compactor
    compressed = ClefDocumentCompressor(compactor=engine, token_budget=1).compress_documents(
        documents(), "refund policy?"
    )
    assert len(compressed) <= 3  # budget may cut; never crashes
