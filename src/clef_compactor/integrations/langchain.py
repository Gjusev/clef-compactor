"""LangChain integration: a query-aware ``BaseDocumentCompressor``.

Requires the optional dependency ``langchain-core``::

    pip install clef-compactor[langchain]

Usage inside a retriever pipeline::

    from langchain_classic.retrievers import ContextualCompressionRetriever
    from clef_compactor.integrations.langchain import ClefDocumentCompressor

    compressor = ClefDocumentCompressor(token_budget=800)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core import ClefCompactor

try:  # pragma: no cover - exercised only when langchain-core is installed
    from langchain_core.documents import Document
    from langchain_core.documents.compressor import BaseDocumentCompressor
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "ClefDocumentCompressor requires langchain-core. "
        "Install it with: pip install 'clef-compactor[langchain]'"
    ) from _exc

__all__ = ["ClefDocumentCompressor"]


class ClefDocumentCompressor(BaseDocumentCompressor):
    """Compress LangChain documents against a query using Clef.

    Kept documents come back in ranked order (best first) with their
    P(relevant) added to ``metadata["clef_score"]``; dropped documents are
    removed. Document text is never rewritten.

    Attributes:
        compactor: Optional shared :class:`ClefCompactor` to reuse (its
            settings win). When omitted, one is built lazily from the
            attributes below and the ``CLEF_*`` environment variables.
        token_budget: Maximum tokens of the compacted context.
        model: ``"clef"`` or ``"clef-flash"``.
        chunk_preview_chars: Characters of each document sent to Clef.
        relevance_threshold: P(relevant) below which a document is dropped.
        account_id: Optional explicit Cloudflare account id.
        api_token: Optional explicit API token.
    """

    model_config = {"arbitrary_types_allowed": True}

    compactor: ClefCompactor | None = None
    token_budget: int = 1000
    model: str = "clef"
    chunk_preview_chars: int = 512
    relevance_threshold: float = 0.5
    account_id: str | None = None
    api_token: str | None = None
    _engine: ClefCompactor | None = None

    def _resolve_compactor(self) -> ClefCompactor:
        """Return the injected compactor or build (once) one from settings."""
        if self.compactor is not None:
            return self.compactor
        if self._engine is None:
            self._engine = ClefCompactor(
                account_id=self.account_id,
                api_token=self.api_token,
                model=self.model,
                chunk_preview_chars=self.chunk_preview_chars,
                relevance_threshold=self.relevance_threshold,
            )
        return self._engine

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Any = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Score *documents* against *query* and return the kept ones, ranked.

        Args:
            documents: Retrieved documents to compact.
            query: The user query driving relevance.
            callbacks: Ignored; present for LangChain API compatibility.
            kwargs: Ignored; present for forward compatibility.
        """
        if not documents:
            return []
        result = self._resolve_compactor().compact(
            query,
            [doc.page_content for doc in documents],
            token_budget=self.token_budget,
        )
        by_index = {scored.index: original for scored, original in zip(result.scores, documents, strict=False)}
        kept: list[Document] = []
        for scored in result.kept:
            original = by_index[scored.index]
            metadata = dict(original.metadata)
            metadata["clef_score"] = round(scored.score, 4)
            kept.append(Document(page_content=original.page_content, metadata=metadata))
        return kept
