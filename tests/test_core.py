"""Tests for the compaction algorithm, batching, and sync/async compactors."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from clef_compactor.client import AsyncClefClient, ClefClient
from clef_compactor.config import Settings
from clef_compactor.core import (
    MAX_QUESTIONS_PER_REQUEST,
    RELEVANCE_THRESHOLD,
    AsyncClefCompactor,
    ClefCompactor,
    build_questions,
    build_state,
    count_tokens,
    estimate_input_cost,
    fill_budget,
    parse_relevance_answer,
    score_chunks,
)
from clef_compactor.exceptions import ClefError
from clef_compactor.models import ClefReply, DropReason, TokenUsage
from conftest import envelope, load_fixture, noul_answer, scripted_transport

CHUNKS = [
    "Our refund policy allows returns within 30 days of purchase.",
    "Refunds are processed to the original payment method in 5 business days.",
    "The Eiffel Tower is located in Paris, France.",
    "To request a refund, email support@example.com with your order number.",
    "Company holidays are listed on the internal calendar.",
]
ANSWERS = {
    "chunk_1": noul_answer(0.93),
    "chunk_2": noul_answer(0.87),
    "chunk_3": noul_answer(0.05),
    "chunk_4": noul_answer(0.78),
    "chunk_5": noul_answer(0.42),
}


def make_compactor(transport: httpx.BaseTransport, **kwargs: Any) -> ClefCompactor:
    settings = Settings(
        account_id="acct",
        api_token="tok",
        timeout=5.0,
        max_retries=0,
        model=kwargs.pop("model", "clef"),
    )
    client = ClefClient(settings, transport=transport)
    return ClefCompactor(account_id="acct", api_token="tok", client=client, **kwargs)


def make_async_compactor(transport: httpx.AsyncBaseTransport) -> AsyncClefCompactor:
    settings = Settings(account_id="acct", api_token="tok", timeout=5.0, max_retries=0)
    client = AsyncClefClient(settings, transport=transport)
    return AsyncClefCompactor(account_id="acct", api_token="tok", client=client)


class FakeClient:
    """Dependency-injection double: answers from a canned mapping, records calls."""

    def __init__(self, answers: dict[str, Any], usage: TokenUsage | None = None) -> None:
        self.answers = answers
        self.usage = usage
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def ask(self, state: str, questions: dict[str, Any], **kwargs: Any) -> ClefReply:
        self.calls.append((state, questions))
        return ClefReply(model="clef", answers=self.answers, usage=self.usage, latency_ms=1.5)

    def close(self) -> None:
        """Satisfy the ClefClient protocol."""
        self.closed = True


def scored(index: int, score: float, tokens: int) -> Any:
    from clef_compactor.models import ScoredChunk

    return ScoredChunk(index=index, text=f"c{index}", score=score, tokens=tokens)


class TestBuildState:
    def test_renders_query_and_numbered_previews(self) -> None:
        state = build_state("refund?", ["alpha beta", "gamma"], 512)
        assert state.startswith("Question: refund?")
        assert "[Doc 1]: alpha beta" in state
        assert "[Doc 2]: gamma" in state

    def test_offset_keeps_numbering_stable_across_batches(self) -> None:
        state = build_state("q", ["only chunk"], 512, offset=64)
        assert "[Doc 65]: only chunk" in state

    def test_preview_truncation_and_whitespace_collapse(self) -> None:
        state = build_state("q", ["a" * 100 + "   " + "b"], 10)
        line = [line for line in state.splitlines() if line.startswith("[Doc 1]")][0]
        assert line == "[Doc 1]: " + "a" * 10


class TestBuildQuestions:
    def test_one_noul_question_per_chunk(self) -> None:
        questions = build_questions(3)
        assert list(questions) == ["chunk_1", "chunk_2", "chunk_3"]
        for question in questions.values():
            assert question["type"] == "noul"
            assert "question" in question["instructions"]
            assert set(question["criteria"]) == {"true", "false"}

    def test_offset_numbering(self) -> None:
        questions = build_questions(2, offset=64)
        assert list(questions) == ["chunk_65", "chunk_66"]

    def test_respects_api_question_limit(self) -> None:
        assert MAX_QUESTIONS_PER_REQUEST == 64
        assert len(build_questions(64)) == 64


class TestParseRelevanceAnswer:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ({"type": "noul", "noul": 0.75}, 0.75),
            ({"noul": 0}, 0.0),
            ({"score": 0.33}, 0.33),
            (0.5, 0.5),
            (1, 1.0),
            ("0.25", 0.25),
        ],
    )
    def test_accepted_shapes(self, raw: Any, expected: float) -> None:
        assert parse_relevance_answer(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [1.5, -0.2, 2])
    def test_clamped_to_unit_interval(self, raw: Any) -> None:
        assert 0.0 <= parse_relevance_answer(raw) <= 1.0
        assert parse_relevance_answer(1.5) == 1.0
        assert parse_relevance_answer(-0.2) == 0.0

    @pytest.mark.parametrize("raw", [None, True, False, "essential", {"noul": None}, ["x"]])
    def test_rejects_garbage(self, raw: Any) -> None:
        with pytest.raises(ValueError):
            parse_relevance_answer(raw)


class TestScoreChunks:
    def test_pairs_answers_with_chunks(self) -> None:
        scored_chunks = score_chunks(CHUNKS, ANSWERS)
        assert [chunk.score for chunk in scored_chunks] == [0.93, 0.87, 0.05, 0.78, 0.42]
        assert scored_chunks[0].text == CHUNKS[0]
        assert all(chunk.tokens > 0 for chunk in scored_chunks)

    def test_missing_or_garbage_answers_score_zero(self) -> None:
        scored_chunks = score_chunks(["a", "b"], {"chunk_2": "nonsense"})
        assert scored_chunks[0].score == 0.0
        assert scored_chunks[1].score == 0.0


class TestFillBudget:
    def test_deterministic_ranking_and_reasons(self) -> None:
        chunks = [scored(0, 0.9, 50), scored(1, 0.1, 10), scored(2, 0.9, 20), scored(3, 0.7, 200)]
        kept, dropped = fill_budget(chunks, token_budget=70)
        assert [chunk.index for chunk in kept] == [2, 0]
        reasons = {chunk.index: chunk.drop_reason for chunk in dropped}
        assert reasons[1] == DropReason.IRRELEVANT
        assert reasons[3] == DropReason.BUDGET_EXHAUSTED

    def test_ties_break_on_tokens_then_original_order(self) -> None:
        chunks = [scored(0, 0.9, 30), scored(1, 0.9, 10), scored(2, 0.9, 20)]
        kept, _ = fill_budget(chunks, token_budget=100)
        assert [chunk.index for chunk in kept] == [1, 2, 0]

    def test_zero_budget_keeps_nothing_positive(self) -> None:
        chunks = [scored(0, 0.9, 10)]
        kept, dropped = fill_budget(chunks, token_budget=0)
        assert kept == ()
        assert dropped[0].drop_reason == DropReason.BUDGET_EXHAUSTED

    def test_default_threshold_is_half(self) -> None:
        assert RELEVANCE_THRESHOLD == 0.5
        _, dropped = fill_budget([scored(0, 0.5, 5)], token_budget=100)
        assert dropped == ()


class TestCostEstimate:
    def test_published_input_price_only(self) -> None:
        assert estimate_input_cost(1_000_000) == pytest.approx(0.24)
        assert estimate_input_cost(412) == pytest.approx(412 * 0.24 / 1_000_000)

    def test_missing_usage_costs_nothing(self) -> None:
        assert estimate_input_cost(None) == 0.0
        assert estimate_input_cost(0) == 0.0


class TestClefCompactor:
    def test_end_to_end_with_fixture_envelope(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json=load_fixture("noul_score_response.json"))
        )
        compactor = make_compactor(transport)
        result = compactor.compact("What is the refund policy?", CHUNKS, token_budget=100_000)
        # chunk_3 (0.05) and chunk_5 (0.42) fall below the 0.5 threshold; the
        # rest fit the huge budget.
        assert [chunk.index for chunk in result.kept] == [0, 1, 3]
        assert [chunk.index for chunk in result.dropped] == [4, 2]
        assert all(chunk.drop_reason == DropReason.IRRELEVANT for chunk in result.dropped)
        assert result.scores[2].score == 0.05
        # Kept chunks stay verbatim.
        assert result.kept_texts()[0] == CHUNKS[0]
        assert result.usage == TokenUsage(input_tokens=412, output_tokens=96)
        assert result.cost_estimate == pytest.approx(412 * 0.24 / 1_000_000)
        assert result.model == "clef"
        compactor.close()

    def test_budget_exhaustion(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json=envelope(ANSWERS))
        )
        compactor = make_compactor(transport)
        result = compactor.compact("refund policy", CHUNKS, token_budget=15)
        assert [chunk.index for chunk in result.kept] == [0]
        assert result.total_output_tokens <= 15
        compactor.close()

    def test_custom_threshold(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=envelope(ANSWERS)))
        compactor = make_compactor(transport, relevance_threshold=0.9)
        result = compactor.compact("q", CHUNKS, token_budget=100_000)
        assert [chunk.index for chunk in result.kept] == [0]
        compactor.close()

    def test_empty_chunks_skip_the_api(self) -> None:
        transport, requests = scripted_transport(lambda req, seen: httpx.Response(200, json=envelope({})))
        compactor = make_compactor(transport)
        result = compactor.compact("q", [], token_budget=10)
        assert result.kept == ()
        assert result.scores == ()
        assert result.usage is None
        assert result.cost_estimate == 0.0
        assert requests == []
        compactor.close()

    def test_score_preserves_original_order(self) -> None:
        compactor = make_compactor(httpx.MockTransport(lambda request: httpx.Response(200, json=envelope(ANSWERS))))
        scored_chunks = compactor.score("q", CHUNKS)
        assert [chunk.index for chunk in scored_chunks] == [0, 1, 2, 3, 4]
        compactor.close()

    def test_batches_requests_over_64_questions(self) -> None:
        answers = {f"chunk_{i}": noul_answer(0.9) for i in range(1, 131)}
        transport, requests = scripted_transport(
            lambda req, seen: httpx.Response(200, json=envelope(answers))
        )
        compactor = make_compactor(transport)
        chunks = [f"chunk number {i}" for i in range(130)]
        result = compactor.compact("q", chunks, token_budget=100_000)
        assert len(requests) == 3
        body_sizes = [len(json_questions(request)) for request in requests]
        assert body_sizes == [64, 64, 2]
        # Global numbering keeps answers aligned across batches.
        first_body = json.loads(requests[0].content)
        assert list(first_body["questions"])[0] == "chunk_1"
        last_body = json.loads(requests[-1].content)
        assert list(last_body["questions"]) == ["chunk_129", "chunk_130"]
        assert len(result.kept) == 130
        compactor.close()

    def test_dependency_injected_fake_client(self) -> None:
        client = FakeClient(ANSWERS, usage=TokenUsage(input_tokens=412, output_tokens=96))
        compactor = ClefCompactor(account_id="a", api_token="t", client=client)  # type: ignore[arg-type]
        result = compactor.compact("q", CHUNKS, token_budget=100_000)
        assert len(client.calls) == 1
        assert len(result.kept) == 3
        compactor.close()

    def test_context_manager_closes_client(self) -> None:
        compactor = make_compactor(httpx.MockTransport(lambda request: httpx.Response(200, json=envelope(ANSWERS))))
        with compactor:
            compactor.score("q", CHUNKS)
        with pytest.raises(RuntimeError):
            compactor.score("q", CHUNKS)

    def test_api_errors_propagate(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(401, json={}))
        compactor = make_compactor(transport)
        with pytest.raises(ClefError):
            compactor.compact("q", CHUNKS, token_budget=100)
        compactor.close()


class TestAsyncClefCompactor:
    async def test_end_to_end(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=envelope(ANSWERS)))
        compactor = make_async_compactor(transport)
        result = await compactor.compact("refund policy", CHUNKS, token_budget=100_000)
        assert [chunk.index for chunk in result.kept] == [0, 1, 3]
        await compactor.aclose()

    async def test_score_preserves_original_order(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=envelope(ANSWERS)))
        compactor = make_async_compactor(transport)
        scored_chunks = await compactor.score("q", CHUNKS)
        assert [chunk.index for chunk in scored_chunks] == [0, 1, 2, 3, 4]
        await compactor.aclose()

    async def test_empty_chunks_skip_the_api(self) -> None:
        transport, requests = scripted_transport(lambda req, seen: httpx.Response(200, json=envelope({})))
        compactor = make_async_compactor(transport)
        result = await compactor.compact("q", [], token_budget=10)
        assert result.kept == ()
        assert requests == []
        await compactor.aclose()

    async def test_batches_requests_over_64_questions(self) -> None:
        answers = {f"chunk_{i}": noul_answer(0.8) for i in range(1, 70)}
        transport, requests = scripted_transport(
            lambda req, seen: httpx.Response(200, json=envelope(answers))
        )
        compactor = make_async_compactor(transport)
        result = await compactor.compact("q", [f"chunk {i}" for i in range(69)], token_budget=100_000)
        assert len(requests) == 2
        assert [len(json_questions(request)) for request in requests] == [64, 5]
        assert len(result.kept) == 69
        await compactor.aclose()

    async def test_async_context_manager(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=envelope(ANSWERS)))
        async with make_async_compactor(transport) as compactor:
            result = await compactor.compact("q", CHUNKS, token_budget=1000)
            assert result.kept
        with pytest.raises(RuntimeError):
            await compactor.score("q", CHUNKS)


def json_questions(request: httpx.Request) -> dict[str, Any]:
    import json

    return json.loads(request.content)["questions"]


def test_token_count_fallback_is_positive() -> None:
    assert count_tokens("one two three") > 0
