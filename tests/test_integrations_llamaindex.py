"""Tests for the LlamaIndex integration using lightweight framework stubs.

``llama-index-core`` pulls in a large dependency tree (nltk, pandas, and
friends) that is too heavy for this repo's development environment, so these
tests inject minimal stand-ins for the three framework names the adapter
imports (``llama_index.core.schema``, ``...postprocessor.types``) and then
import the adapter fresh. The stubs mirror the real classes' minimal surface:
``NodeWithScore(node, score)``, ``node.get_content()``, ``QueryBundle.query_str``
and an instantiable ``BaseNodePostprocessor``. If llama-index-core is ever
installed in the test environment, real tests should be preferred; the adapter
code itself is unchanged either way.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest

from clef_compactor import ClefCompactor
from clef_compactor.client import ClefClient
from clef_compactor.config import Settings
from conftest import canned_transport, envelope, noul_answer

LLAMA_INSTALLED = importlib.util.find_spec("llama_index") is not None
pytestmark = pytest.mark.skipif(
    LLAMA_INSTALLED, reason="llama-index-core is installed; stubs unnecessary"
)

ANSWERS = {"chunk_1": noul_answer(0.95), "chunk_2": noul_answer(0.05), "chunk_3": noul_answer(0.8)}


class _StubNode:
    def __init__(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        self._content = content
        self.metadata = metadata or {}

    def get_content(self) -> str:
        return self._content

    model_copy = None  # force the adapter's hasattr fallback path


class _StubNodeWithScore:
    def __init__(self, node: Any, score: float | None = None) -> None:
        self.node = node
        self.score = score


class _StubQueryBundle:
    def __init__(self, query_str: str) -> None:
        self.query_str = query_str


class _StubBaseNodePostprocessor:
    """Instantiable stand-in for the pydantic-based real base class."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def postprocess_nodes(self, nodes: Any, query_bundle: Any = None) -> Any:
        return self._postprocess_nodes(nodes, query_bundle)

    def _postprocess_nodes(self, nodes: Any, query_bundle: Any = None) -> Any:
        raise NotImplementedError


@pytest.fixture
def adapter_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the adapter fresh with stub framework modules on ``sys.modules``."""
    schema = types.ModuleType("llama_index.core.schema")
    schema.NodeWithScore = _StubNodeWithScore
    schema.QueryBundle = _StubQueryBundle
    postprocessor = types.ModuleType("llama_index.core.postprocessor.types")
    postprocessor.BaseNodePostprocessor = _StubBaseNodePostprocessor
    monkeypatch.setitem(sys.modules, "llama_index.core", types.ModuleType("llama_index.core"))
    monkeypatch.setitem(sys.modules, "llama_index.core.schema", schema)
    monkeypatch.setitem(
        sys.modules, "llama_index.core.postprocessor", types.ModuleType("llama_index.core.postprocessor")
    )
    monkeypatch.setitem(sys.modules, "llama_index.core.postprocessor.types", postprocessor)
    monkeypatch.delitem(sys.modules, "clef_compactor.integrations.llamaindex", raising=False)
    yield importlib.import_module("clef_compactor.integrations.llamaindex")
    monkeypatch.delitem(sys.modules, "clef_compactor.integrations.llamaindex", raising=False)


def postprocessor() -> Any:
    settings = Settings(account_id="a", api_token="t", max_retries=0)
    client = ClefClient(settings, transport=canned_transport(envelope(ANSWERS)))
    engine = ClefCompactor(account_id="a", api_token="t", client=client)
    return engine


def nodes() -> list[Any]:
    return [
        _StubNodeWithScore(_StubNode("Refund policy: 30 days."), 0.2),
        _StubNodeWithScore(_StubNode("The Eiffel Tower is in Paris."), 0.9),
        _StubNodeWithScore(_StubNode("Refunds go to the original payment method."), 0.1),
    ]


def test_keeps_relevant_nodes_ranked(adapter_module: Any) -> None:
    processor = adapter_module.ClefNodePostprocessor(compactor=postprocessor(), token_budget=10**6)
    kept = processor.postprocess_nodes(nodes(), _StubQueryBundle("refund policy?"))
    assert [item.node.get_content() for item in kept] == [
        "Refund policy: 30 days.",
        "Refunds go to the original payment method.",
    ]
    assert all(0.0 <= item.score <= 1.0 for item in kept)
    assert kept[0].score == 0.95


def test_empty_nodes_return_empty(adapter_module: Any) -> None:
    processor = adapter_module.ClefNodePostprocessor(compactor=postprocessor())
    assert processor.postprocess_nodes([], _StubQueryBundle("q")) == []


def test_missing_query_bundle_raises(adapter_module: Any) -> None:
    processor = adapter_module.ClefNodePostprocessor(compactor=postprocessor())
    with pytest.raises(ValueError):
        processor.postprocess_nodes(nodes(), None)


def test_empty_query_string_raises(adapter_module: Any) -> None:
    processor = adapter_module.ClefNodePostprocessor(compactor=postprocessor())
    with pytest.raises(ValueError):
        processor.postprocess_nodes(nodes(), _StubQueryBundle(""))
