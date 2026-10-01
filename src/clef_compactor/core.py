"""Query-aware compaction of RAG retrieval batches using Cloudflare Clef.

The compactor asks Clef one ``noul`` question per chunk -- "is this chunk
needed to answer the query?" -- and receives P(relevant) in ``[0, 1]``. It then
greedily fills a token budget with the best-scoring chunks. Chunks are kept
**verbatim**, never rewritten, so citations remain auditable.

Algorithm (deterministic given the same API answers):

1. Score all chunks, one ``noul`` question per chunk, in requests of at most 64
   questions (the Clef API limit).
2. Drop chunks scoring below ``relevance_threshold`` (reason: ``irrelevant``).
3. Sort survivors by ``(score desc, tokens asc, original index asc)`` and keep
   chunks while the running token total fits ``token_budget``; the rest are
   dropped with reason ``budget_exhausted``.

Cost reporting follows Cloudflare's published pricing: input tokens cost
$0.24 per million. No output-token price is published, so none is estimated.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import tiktoken

from .client import AsyncClefClient, ClefClient
from .config import Settings
from .models import CompactResult, DropReason, ScoredChunk, TokenUsage

__all__ = [
    "ClefCompactor",
    "AsyncClefCompactor",
    "MAX_QUESTIONS_PER_REQUEST",
    "RELEVANCE_THRESHOLD",
    "INPUT_PRICE_PER_MILLION_TOKENS_USD",
    "count_tokens",
    "build_state",
    "build_questions",
    "parse_relevance_answer",
    "score_chunks",
    "fill_budget",
    "estimate_input_cost",
]

logger = logging.getLogger("clef_compactor")

#: The Clef API accepts at most 64 questions per request.
MAX_QUESTIONS_PER_REQUEST = 64

#: Default P(relevant) below which a chunk is dropped as irrelevant.
RELEVANCE_THRESHOLD = 0.5

#: Cloudflare's published Clef input-token price, USD per million tokens.
INPUT_PRICE_PER_MILLION_TOKENS_USD = 0.24

_NOULD_INSTRUCTIONS = (
    "You are filtering retrieved context for a retrieval-augmented pipeline. "
    "Answer whether the document contains information needed to answer the "
    "question. true = the document is needed to answer the question; "
    "false = the document is not needed."
)

_ENCODINGS: dict[str, Any] = {}


def _encoding(name: str) -> Any:
    """Return a cached tiktoken encoding, or fall back to whitespace counts."""
    if name not in _ENCODINGS:
        try:
            _ENCODINGS[name] = tiktoken.get_encoding(name)
        except Exception as exc:  # pragma: no cover - only offline environments
            logger.warning(
                "Could not load tiktoken encoding %r (%s); falling back to "
                "whitespace token estimates.",
                name,
                exc,
            )
            _ENCODINGS[name] = None
    return _ENCODINGS[name]


def count_tokens(text: str, encoding_model: str = "cl100k_base") -> int:
    """Count tokens in *text* with a graceful offline fallback."""
    enc = _encoding(encoding_model)
    if enc is None:
        return len(text.split())
    return len(enc.encode(text))


def estimate_input_cost(input_tokens: int | None) -> float:
    """Estimate USD cost of *input_tokens* at the published Clef input price.

    Output tokens are deliberately excluded: Cloudflare does not publish an
    output-token price for Clef.
    """
    if not input_tokens:
        return 0.0
    return input_tokens * INPUT_PRICE_PER_MILLION_TOKENS_USD / 1_000_000


def build_state(query: str, chunks: Sequence[str], preview_chars: int, offset: int = 0) -> str:
    """Render the Clef ``state`` block: the query plus numbered chunk previews.

    ``offset`` keeps numbering stable across batched requests so answers stay
    aligned with the original chunk order.
    """
    lines = [f"Question: {query}", "", "Documents to evaluate:"]
    for position, chunk in enumerate(chunks, start=offset + 1):
        preview = " ".join(chunk.split())
        lines.append(f"\n[Doc {position}]: {preview[:preview_chars]}")
    return "\n".join(lines)


def build_questions(count: int, offset: int = 0) -> dict[str, dict[str, Any]]:
    """Build one ``noul`` question per chunk, keyed ``chunk_1..count+offset``."""
    return {
        f"chunk_{position + offset}": {
            "type": "noul",
            "instructions": _NOULD_INSTRUCTIONS,
            "criteria": {
                "true": "The document is needed to answer the question.",
                "false": "The document is not needed to answer the question.",
            },
        }
        for position in range(1, count + 1)
    }


def parse_relevance_answer(value: Any) -> float:
    """Normalize one Clef answer to a relevance probability in ``[0.0, 1.0]``.

    Accepts the documented ``{"type": "noul", "noul": 0..1}`` answer object, a
    bare number, or a numeric string.
    """
    raw: Any = value
    if isinstance(value, dict):
        raw = value.get("noul", value.get("score"))
    if isinstance(raw, bool) or raw is None:
        raise ValueError(f"Unparseable relevance answer: {value!r}")
    if isinstance(raw, str):
        try:
            raw = float(raw)
        except ValueError as exc:
            raise ValueError(f"Unparseable relevance answer: {value!r}") from exc
    if not isinstance(raw, (int, float)):
        raise ValueError(f"Unparseable relevance answer: {value!r}")
    return max(0.0, min(float(raw), 1.0))


def score_chunks(
    chunks: Sequence[str],
    answers: dict[str, Any],
    *,
    encoding_model: str = "cl100k_base",
) -> list[ScoredChunk]:
    """Pair every chunk with its answer; missing or unparseable answers score 0."""
    scored: list[ScoredChunk] = []
    for position, text in enumerate(chunks):
        raw = answers.get(f"chunk_{position + 1}")
        if raw is None:
            logger.warning("Missing answer for chunk %d; scoring it 0.", position + 1)
            score = 0.0
        else:
            try:
                score = parse_relevance_answer(raw)
            except ValueError:
                logger.warning("Unparseable answer %r for chunk %d; scoring it 0.", raw, position + 1)
                score = 0.0
        scored.append(
            ScoredChunk(
                index=position,
                text=text,
                score=score,
                tokens=count_tokens(text, encoding_model),
            )
        )
    return scored


def fill_budget(
    scored: Sequence[ScoredChunk],
    token_budget: int,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> tuple[tuple[ScoredChunk, ...], tuple[ScoredChunk, ...]]:
    """Split scored chunks into kept/dropped tuples within *token_budget*.

    Ranking is deterministic: score descending, then fewer tokens, then the
    original input order.
    """
    ranked = sorted(scored, key=lambda chunk: (-chunk.score, chunk.tokens, chunk.index))
    kept: list[ScoredChunk] = []
    dropped: list[ScoredChunk] = []
    running = 0
    for chunk in ranked:
        if chunk.score < relevance_threshold:
            dropped.append(_with_reason(chunk, DropReason.IRRELEVANT))
        elif running + chunk.tokens <= token_budget:
            kept.append(chunk)
            running += chunk.tokens
        else:
            dropped.append(_with_reason(chunk, DropReason.BUDGET_EXHAUSTED))
    return tuple(kept), tuple(dropped)


def _with_reason(chunk: ScoredChunk, reason: DropReason) -> ScoredChunk:
    """Return a copy of *chunk* annotated with *reason*."""
    return ScoredChunk(
        index=chunk.index,
        text=chunk.text,
        score=chunk.score,
        tokens=chunk.tokens,
        drop_reason=reason,
    )


def _empty_result(query: str, token_budget: int, model: str) -> CompactResult:
    """Result for an empty chunk batch: no API call, nothing kept."""
    return CompactResult(
        kept=(),
        dropped=(),
        scores=(),
        usage=None,
        cost_estimate=0.0,
        token_budget=token_budget,
        query=query,
        model=model,
    )


def _batch_ranges(count: int) -> list[tuple[int, int]]:
    """Split *count* chunks into ``(start, length)`` batches of at most 64."""
    return [
        (start, min(MAX_QUESTIONS_PER_REQUEST, count - start))
        for start in range(0, count, MAX_QUESTIONS_PER_REQUEST)
    ]


class ClefCompactor:
    """Score and compact RAG retrieval batches with Cloudflare Clef.

    Args:
        account_id: Cloudflare account id; defaults to ``CLEF_ACCOUNT_ID``
            (fallback ``CLOUDFLARE_ACCOUNT_ID``).
        api_token: API token; defaults to ``CLEF_API_TOKEN`` (fallback
            ``CLOUDFLARE_API_TOKEN``).
        model: ``"clef"`` (precise) or ``"clef-flash"`` (fast).
        base_url: API root; override for AI Gateway or tests.
        timeout: Per-request timeout in seconds.
        max_retries: Retries for transient failures per request.
        encoding_model: tiktoken encoding used for token accounting.
        chunk_preview_chars: Characters of each chunk placed in the scoring
            state. Clef's 64k window allows raising this to score full chunks.
        relevance_threshold: P(relevant) below which a chunk is dropped as
            irrelevant regardless of budget.
        client: Pre-built :class:`ClefClient` for dependency injection.

    Raises:
        ConfigurationError: If credentials are missing or settings are invalid;
            the message lists every problem at once.
    """

    def __init__(
        self,
        *,
        account_id: str | None = None,
        api_token: str | None = None,
        model: str = "clef",
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        log_level: str | None = None,
        encoding_model: str = "cl100k_base",
        chunk_preview_chars: int = 512,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        client: ClefClient | None = None,
    ) -> None:
        self.settings = Settings.from_env(
            account_id=account_id,
            api_token=api_token,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            log_level=log_level,
        )
        self.settings.configure_logging()
        self.encoding_model = encoding_model
        self.chunk_preview_chars = chunk_preview_chars
        self.relevance_threshold = relevance_threshold
        self.client = client or ClefClient(self.settings)

    def compact(
        self,
        query: str,
        chunks: Sequence[str],
        token_budget: int = 1000,
        *,
        preview_chars: int | None = None,
    ) -> CompactResult:
        """Score *chunks* against *query* and keep the best within budget.

        Args:
            query: The user query the chunks should answer.
            chunks: Retrieved chunk texts; kept verbatim, never rewritten.
            token_budget: Maximum tokens of the compacted context.
            preview_chars: Per-call override of :attr:`chunk_preview_chars`.

        Returns:
            A :class:`CompactResult` with kept/dropped chunks, per-chunk
            scores, API usage and the input-token cost estimate.

        Raises:
            ClefError: On any API, network or response failure.
        """
        if not chunks:
            return _empty_result(query, token_budget, self.settings.model)
        reply_answers, usage, latency_ms = self._score_all(query, chunks, preview_chars)
        return self._result_from_answers(
            query, chunks, token_budget, reply_answers, usage, latency_ms
        )

    def score(
        self,
        query: str,
        chunks: Sequence[str],
        *,
        preview_chars: int | None = None,
    ) -> list[ScoredChunk]:
        """Score chunks without applying a budget; original order preserved."""
        if not chunks:
            return []
        answers, _, _ = self._score_all(query, chunks, preview_chars)
        return score_chunks(chunks, answers, encoding_model=self.encoding_model)

    def _score_all(
        self,
        query: str,
        chunks: Sequence[str],
        preview_chars: int | None,
    ) -> tuple[dict[str, Any], TokenUsage | None, float]:
        """Score every chunk, batching questions to respect the 64-question limit.

        Returns merged answers, merged usage and the total latency.
        """
        chars = preview_chars if preview_chars is not None else self.chunk_preview_chars
        merged_answers: dict[str, Any] = {}
        usage: TokenUsage | None = None
        total_latency = 0.0
        for offset, length in _batch_ranges(len(chunks)):
            batch = chunks[offset : offset + length]
            reply = self.client.ask(
                build_state(query, batch, chars, offset=offset),
                build_questions(length, offset=offset),
            )
            merged_answers.update(reply.answers)
            usage = usage.merged_with(reply.usage) if usage else reply.usage
            total_latency += reply.latency_ms
        return merged_answers, usage, total_latency

    def _result_from_answers(
        self,
        query: str,
        chunks: Sequence[str],
        token_budget: int,
        answers: dict[str, Any],
        usage: TokenUsage | None,
        latency_ms: float,
    ) -> CompactResult:
        """Build the final :class:`CompactResult` from merged answers."""
        scored = score_chunks(chunks, answers, encoding_model=self.encoding_model)
        kept, dropped = fill_budget(scored, token_budget, self.relevance_threshold)
        cost = estimate_input_cost(usage.input_tokens if usage else None)
        logger.debug(
            "compaction kept %d/%d chunks (%.1f%% saved), estimated input cost $%.6f",
            len(kept),
            len(scored),
            100 * (1 - (sum(c.tokens for c in kept) / max(sum(c.tokens for c in scored), 1))),
            cost,
        )
        return CompactResult(
            kept=kept,
            dropped=dropped,
            scores=tuple(scored),
            usage=usage,
            cost_estimate=cost,
            token_budget=token_budget,
            query=query,
            model=self.settings.model,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self.client.close()

    def __enter__(self) -> ClefCompactor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class AsyncClefCompactor:
    """Async counterpart of :class:`ClefCompactor` for ``asyncio`` pipelines.

    Shares the algorithm and configuration of :class:`ClefCompactor`; only the
    transport and method signatures differ.
    """

    def __init__(
        self,
        *,
        account_id: str | None = None,
        api_token: str | None = None,
        model: str = "clef",
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        log_level: str | None = None,
        encoding_model: str = "cl100k_base",
        chunk_preview_chars: int = 512,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        client: AsyncClefClient | None = None,
    ) -> None:
        self.settings = Settings.from_env(
            account_id=account_id,
            api_token=api_token,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            log_level=log_level,
        )
        self.settings.configure_logging()
        self.encoding_model = encoding_model
        self.chunk_preview_chars = chunk_preview_chars
        self.relevance_threshold = relevance_threshold
        self.client = client or AsyncClefClient(self.settings)

    async def compact(
        self,
        query: str,
        chunks: Sequence[str],
        token_budget: int = 1000,
        *,
        preview_chars: int | None = None,
    ) -> CompactResult:
        """Async version of :meth:`ClefCompactor.compact`."""
        if not chunks:
            return _empty_result(query, token_budget, self.settings.model)
        answers, usage, latency_ms = await self._score_all(query, chunks, preview_chars)
        return self._result_from_answers(
            query, chunks, token_budget, answers, usage, latency_ms
        )

    async def score(
        self,
        query: str,
        chunks: Sequence[str],
        *,
        preview_chars: int | None = None,
    ) -> list[ScoredChunk]:
        """Async version of :meth:`ClefCompactor.score`."""
        if not chunks:
            return []
        answers, _, _ = await self._score_all(query, chunks, preview_chars)
        return score_chunks(chunks, answers, encoding_model=self.encoding_model)

    async def _score_all(
        self,
        query: str,
        chunks: Sequence[str],
        preview_chars: int | None,
    ) -> tuple[dict[str, Any], TokenUsage | None, float]:
        """Async version of :meth:`ClefCompactor._score_all`."""
        chars = preview_chars if preview_chars is not None else self.chunk_preview_chars
        merged_answers: dict[str, Any] = {}
        usage: TokenUsage | None = None
        total_latency = 0.0
        for offset, length in _batch_ranges(len(chunks)):
            batch = chunks[offset : offset + length]
            reply = await self.client.ask(
                build_state(query, batch, chars, offset=offset),
                build_questions(length, offset=offset),
            )
            merged_answers.update(reply.answers)
            usage = usage.merged_with(reply.usage) if usage else reply.usage
            total_latency += reply.latency_ms
        return merged_answers, usage, total_latency

    def _result_from_answers(
        self,
        query: str,
        chunks: Sequence[str],
        token_budget: int,
        answers: dict[str, Any],
        usage: TokenUsage | None,
        latency_ms: float,
    ) -> CompactResult:
        """Identical result construction to the sync compactor."""
        return ClefCompactor._result_from_answers(  # noqa: SLF001 - shared, stateless logic
            self, query, chunks, token_budget, answers, usage, latency_ms
        )

    async def aclose(self) -> None:
        """Release the underlying async HTTP connection pool."""
        await self.client.aclose()

    async def __aenter__(self) -> AsyncClefCompactor:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
