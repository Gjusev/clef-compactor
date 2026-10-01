"""LlamaIndex integration: a node postprocessor powered by Clef.

Requires the optional dependency ``llama-index-core``::

    pip install clef-compactor[llamaindex]

Usage::

    from llama_index.core.query_engine import RetrieverQueryEngine
    from clef_compactor.integrations.llamaindex import ClefNodePostprocessor

    query_engine = RetrieverQueryEngine(
        retriever=base_retriever,
        node_postprocessors=[ClefNodePostprocessor(token_budget=800)],
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core import ClefCompactor

try:  # pragma: no cover - exercised only when llama-index-core is installed
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "ClefNodePostprocessor requires llama-index-core. "
        "Install it with: pip install 'clef-compactor[llamaindex]'"
    ) from _exc

__all__ = ["ClefNodePostprocessor"]


class ClefNodePostprocessor(BaseNodePostprocessor):
    """Post-process retrieved LlamaIndex nodes: keep the relevant, cut the rest.

    Kept nodes are returned in ranked order (best first) and their
    ``NodeWithScore.score`` is replaced with the Clef P(relevant), so
    downstream code sees a query-aware reranking for free. Node content is
    never rewritten.
    """

    def __init__(
        self,
        compactor: ClefCompactor | None = None,
        token_budget: int = 1000,
        chunk_preview_chars: int = 512,
        model: str = "clef",
        relevance_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._compactor = compactor or ClefCompactor(
            model=model,
            chunk_preview_chars=chunk_preview_chars,
            relevance_threshold=relevance_threshold,
        )
        self._token_budget = token_budget

    def _postprocess_nodes(
        self,
        nodes: Sequence[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Score nodes against the query and return the kept ones, ranked.

        Args:
            nodes: Retrieved nodes to compact.
            query_bundle: The query; its ``query_str`` drives relevance.

        Raises:
            ValueError: If ``query_bundle`` is missing or has no query string.
        """
        query = query_bundle.query_str if query_bundle is not None else ""
        if not query:
            raise ValueError(
                "ClefNodePostprocessor needs a query: pass a QueryBundle with a "
                "non-empty query_str."
            )
        if not nodes:
            return []
        result = self._compactor.compact(
            query,
            [node.node.get_content() for node in nodes],
            token_budget=self._token_budget,
        )
        original_by_index = {
            scored.index: original for scored, original in zip(result.scores, nodes, strict=False)
        }
        return [
            NodeWithScore(node=original_by_index[scored.index].node, score=round(scored.score, 4))
            for scored in result.kept
        ]
