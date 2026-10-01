"""Dataclasses describing compaction inputs, outputs and API replies.

All public results are frozen dataclasses, so callers can safely share them
across threads and log them without mutation risks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "DropReason",
    "TokenUsage",
    "ClefReply",
    "ScoredChunk",
    "CompactResult",
]


class DropReason(str, Enum):
    """Why a chunk did not make it into the compacted context."""

    IRRELEVANT = "irrelevant"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting reported by the Clef API.

    ``None`` fields mean the API did not report that counter for this call.
    Output tokens are always reported separately: Cloudflare publishes an
    input-token price only, never an output-token price.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any] | None) -> TokenUsage | None:
        """Build usage from the API ``usage`` object, tolerating key variants."""
        if not isinstance(raw, dict):
            return None
        usage = cls(
            input_tokens=_first_int(raw, "input_tokens", "prompt_tokens"),
            output_tokens=_first_int(raw, "output_tokens", "completion_tokens"),
            total_tokens=_first_int(raw, "total_tokens"),
        )
        if usage.input_tokens is None and usage.output_tokens is None and usage.total_tokens is None:
            return None
        return usage

    def merged_with(self, other: TokenUsage | None) -> TokenUsage:
        """Sum two usage records field by field, treating ``None`` as absent."""
        if other is None:
            return self
        return TokenUsage(
            input_tokens=_sum_or_none(self.input_tokens, other.input_tokens),
            output_tokens=_sum_or_none(self.output_tokens, other.output_tokens),
            total_tokens=_sum_or_none(self.total_tokens, other.total_tokens),
        )


def _sum_or_none(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def _first_int(raw: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


@dataclass(frozen=True)
class ClefReply:
    """Normalized reply from one Clef API call."""

    model: str
    answers: dict[str, Any]
    usage: TokenUsage | None = None
    request_id: str | None = None
    latency_ms: float = 0.0


@dataclass(frozen=True)
class ScoredChunk:
    """A single chunk with its Clef relevance verdict.

    Attributes:
        index: Zero-based position of the chunk in the original input.
        text: The full, unmodified chunk text (never rewritten).
        score: P(relevant) from the Clef noul answer, ``0.0`` to ``1.0``.
        tokens: Token count of the chunk text.
        drop_reason: Why the chunk was cut, or ``None`` if it was kept.
    """

    index: int
    text: str
    score: float
    tokens: int
    drop_reason: DropReason | None = None

    @property
    def kept(self) -> bool:
        """Whether the chunk survived compaction."""
        return self.drop_reason is None


@dataclass(frozen=True)
class CompactResult:
    """Outcome of compacting one retrieval batch.

    Attributes:
        kept: Chunks that survived, ranked best-first (score desc, then
            fewer tokens, then original order).
        dropped: Chunks that were cut, ranked the same way.
        scores: Every scored chunk in the original input order.
        usage: Aggregated Clef token usage for the scoring call(s).
        cost_estimate: Estimated USD cost of the scoring call input tokens at
            Cloudflare's published rate. Output-token pricing is not published
            by Cloudflare, so it is never estimated here.
        token_budget: Budget the compaction targeted.
        query: The query the chunks were scored against.
        model: Model selector used for scoring.
        latency_ms: Total wall-clock time of the scoring API call(s).
    """

    kept: tuple[ScoredChunk, ...]
    dropped: tuple[ScoredChunk, ...]
    scores: tuple[ScoredChunk, ...]
    usage: TokenUsage | None
    cost_estimate: float
    token_budget: int
    query: str
    model: str = "clef"
    latency_ms: float = 0.0

    @property
    def total_input_tokens(self) -> int:
        """Tokens across all input chunks."""
        return sum(chunk.tokens for chunk in self.scores)

    @property
    def total_output_tokens(self) -> int:
        """Tokens of the kept chunks, i.e. the compacted context."""
        return sum(chunk.tokens for chunk in self.kept)

    @property
    def saved_tokens(self) -> int:
        """Tokens removed from the context."""
        return max(self.total_input_tokens - self.total_output_tokens, 0)

    @property
    def saved_fraction(self) -> float:
        """Fraction of input tokens removed, in ``[0.0, 1.0]``."""
        if self.total_input_tokens <= 0:
            return 0.0
        return self.saved_tokens / self.total_input_tokens

    def kept_texts(self) -> list[str]:
        """The kept chunk texts in ranked order, ready to join into a prompt."""
        return [chunk.text for chunk in self.kept]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary of this result."""
        return {
            "query": self.query,
            "model": self.model,
            "kept": [_chunk_dict(chunk) for chunk in self.kept],
            "dropped": [_chunk_dict(chunk) for chunk in self.dropped],
            "scores": [_chunk_dict(chunk) for chunk in self.scores],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "saved_tokens": self.saved_tokens,
            "saved_fraction": round(self.saved_fraction, 4),
            "token_budget": self.token_budget,
            "usage": self.usage
            and {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "cost_estimate": self.cost_estimate,
            "latency_ms": round(self.latency_ms, 2),
        }


def _chunk_dict(chunk: ScoredChunk) -> dict[str, Any]:
    return {
        "index": chunk.index,
        "score": round(chunk.score, 4),
        "tokens": chunk.tokens,
        "drop_reason": chunk.drop_reason.value if chunk.drop_reason else None,
        "text": chunk.text,
    }
