"""Tests for the result models."""

from __future__ import annotations

import json

from clef_compactor.models import (
    ClefReply,
    CompactResult,
    DropReason,
    ScoredChunk,
    TokenUsage,
)


def chunk(index: int, score: float, tokens: int, reason: DropReason | None = None) -> ScoredChunk:
    return ScoredChunk(index=index, text=f"chunk-{index}", score=score, tokens=tokens, drop_reason=reason)


def test_token_usage_from_api_canonical_keys() -> None:
    usage = TokenUsage.from_api({"input_tokens": 412, "output_tokens": 96})
    assert usage == TokenUsage(input_tokens=412, output_tokens=96, total_tokens=None)


def test_token_usage_from_api_legacy_keys() -> None:
    usage = TokenUsage.from_api({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
    assert usage == TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12)


def test_token_usage_from_api_garbage_returns_none() -> None:
    assert TokenUsage.from_api(None) is None
    assert TokenUsage.from_api({"unrelated": 1}) is None
    assert TokenUsage.from_api([1, 2]) is None


def test_token_usage_merged_with() -> None:
    first = TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12)
    second = TokenUsage(input_tokens=5, total_tokens=None)
    merged = first.merged_with(second)
    assert merged == TokenUsage(input_tokens=15, output_tokens=2, total_tokens=12)
    assert first.merged_with(None) == first


def test_scored_chunk_kept_property() -> None:
    kept = chunk(0, 0.9, 10)
    cut = chunk(1, 0.1, 10, DropReason.IRRELEVANT)
    assert kept.kept is True
    assert kept.drop_reason is None
    assert cut.kept is False


def test_compact_result_token_totals() -> None:
    scores = (chunk(0, 0.9, 30), chunk(1, 0.1, 20), chunk(2, 0.8, 50))
    result = CompactResult(
        kept=scores[:1] + scores[2:],
        dropped=(scores[1],),
        scores=scores,
        usage=TokenUsage(input_tokens=412, output_tokens=96),
        cost_estimate=412 * 0.24 / 1_000_000,
        token_budget=100,
        query="q",
        model="clef",
    )
    assert result.total_input_tokens == 100
    assert result.total_output_tokens == 80
    assert result.saved_tokens == 20
    assert result.saved_fraction == 0.2


def test_compact_result_saved_fraction_zero_division() -> None:
    result = CompactResult(
        kept=(), dropped=(), scores=(), usage=None, cost_estimate=0.0,
        token_budget=10, query="q",
    )
    assert result.saved_fraction == 0.0
    assert result.saved_tokens == 0


def test_compact_result_kept_texts() -> None:
    kept = (ScoredChunk(index=3, text="alpha", score=0.9, tokens=5),)
    result = CompactResult(
        kept=kept, dropped=(), scores=kept, usage=None, cost_estimate=0.0,
        token_budget=10, query="q",
    )
    assert result.kept_texts() == ["alpha"]


def test_compact_result_to_dict_is_json_serializable() -> None:
    kept = chunk(0, 0.91234, 30)
    dropped = chunk(1, 0.1, 20, DropReason.BUDGET_EXHAUSTED)
    result = CompactResult(
        kept=(kept,),
        dropped=(dropped,),
        scores=(kept, dropped),
        usage=TokenUsage(input_tokens=412, output_tokens=96),
        cost_estimate=9.888e-05,
        token_budget=100,
        query="refund?",
        model="clef-flash",
        latency_ms=12.3456,
    )
    payload = result.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["saved_fraction"] == 0.4
    assert payload["usage"] == {"input_tokens": 412, "output_tokens": 96, "total_tokens": None}
    assert payload["scores"][0]["score"] == 0.9123
    assert payload["dropped"][0]["drop_reason"] == "budget_exhausted"


def test_clef_reply_defaults() -> None:
    reply = ClefReply(model="clef", answers={"chunk_1": {"noul": 0.5}})
    assert reply.usage is None
    assert reply.request_id is None
    assert reply.latency_ms == 0.0
