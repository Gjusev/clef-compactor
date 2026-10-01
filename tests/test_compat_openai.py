"""Tests for the OpenAI-compatible wrapper (compactor is canned, no API calls)."""

from __future__ import annotations

import json

import pytest

from clef_compactor import ClefCompactor
from clef_compactor.client import ClefClient
from clef_compactor.compat.openai import compact_context, handle_chat_completions
from clef_compactor.config import Settings
from conftest import canned_transport, envelope, noul_answer

CHUNKS = [
    "Our refund policy allows returns within 30 days.",
    "The Eiffel Tower is in Paris.",
    "Refunds go to the original payment method.",
]
ANSWERS = {
    "chunk_1": noul_answer(0.95),
    "chunk_2": noul_answer(0.05),
    "chunk_3": noul_answer(0.85),
}


def engine_with_answers(answers: dict[str, object]) -> ClefCompactor:
    settings = Settings(account_id="a", api_token="t", max_retries=0)
    client = ClefClient(settings, transport=canned_transport(envelope(answers)))
    return ClefCompactor(account_id="a", api_token="t", client=client)


class TestHandleChatCompletions:
    def test_response_is_openai_shaped(self) -> None:
        engine = engine_with_answers(ANSWERS)
        payload = {
            "model": "clef-compactor",
            "messages": [{"role": "user", "content": "What is the refund policy?"}],
            "clef": {"chunks": CHUNKS, "query": "What is the refund policy?"},
        }
        response = handle_chat_completions(payload, compactor=engine)
        json.dumps(response)  # must be JSON-serializable
        assert response["object"] == "chat.completion"
        assert response["model"] == "clef-compactor"
        assert response["id"].startswith("chatcmpl-clef-")
        assert isinstance(response["created"], int)
        choice = response["choices"][0]
        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["role"] == "assistant"
        assert "[Doc 1 | P=0.95]" in choice["message"]["content"]
        assert "Eiffel Tower" not in choice["message"]["content"]
        assert response["usage"]["prompt_tokens"] > 0
        assert response["clef"]["kept"] == 2
        assert response["clef"]["dropped"] == 1

    def test_messages_fallback_splits_context_and_query(self) -> None:
        engine = engine_with_answers(ANSWERS)
        payload = {
            "model": "clef-compactor",
            "messages": [
                {"role": "system", "content": CHUNKS[0]},
                {"role": "assistant", "content": CHUNKS[1]},
                {"role": "system", "content": CHUNKS[2]},
                {"role": "user", "content": "refund policy?"},
                {"role": "user", "content": "and the tower?"},
            ],
        }
        response = handle_chat_completions(payload, compactor=engine)
        # 4 messages -> query is the last user message; 3 context chunks.
        assert response["clef"]["kept"] + response["clef"]["dropped"] == 3

    def test_content_part_arrays_are_flattened(self) -> None:
        engine = engine_with_answers(ANSWERS)
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": CHUNKS[0]}, {"type": "text", "text": ""}],
                },
                {"role": "user", "content": [{"type": "text", "text": "refund policy?"}]},
            ],
            "clef": {"chunks": CHUNKS},
        }
        response = handle_chat_completions(payload, compactor=engine)
        assert response["choices"][0]["message"]["content"]

    def test_deterministic_id_for_same_query(self) -> None:
        engine = engine_with_answers(ANSWERS)
        payload = {"messages": [{"role": "user", "content": "q"}], "clef": {"chunks": CHUNKS}}
        first = handle_chat_completions(payload, compactor=engine)["id"]
        second = handle_chat_completions(payload, compactor=engine)["id"]
        assert first == second

    def test_all_irrelevant_yields_placeholder(self) -> None:
        engine = engine_with_answers({key: noul_answer(0.01) for key in ANSWERS})
        payload = {"messages": [{"role": "user", "content": "q"}], "clef": {"chunks": CHUNKS}}
        content = handle_chat_completions(payload, compactor=engine)["choices"][0]["message"]["content"]
        assert content == "(no chunk passed the relevance bar)"

    def test_token_budget_extension_is_honored(self) -> None:
        engine = engine_with_answers(ANSWERS)
        payload = {
            "messages": [{"role": "user", "content": "q"}],
            "clef": {"chunks": CHUNKS, "token_budget": 1},
        }
        response = handle_chat_completions(payload, compactor=engine)
        assert response["usage"]["completion_tokens"] <= 1

    @pytest.mark.parametrize(
        "payload, match",
        [
            ({"messages": []}, "non-empty"),
            ({"messages": [{"role": "user", "content": "q"}], "clef": "nope"}, "clef must be"),
            ({"messages": [{"role": "system", "content": "ctx"}], "clef": {"chunks": ["d"]}}, "query"),
            ({"messages": [{"role": "user", "content": "q"}]}, "chunks"),
        ],
    )
    def test_invalid_payloads_raise_value_error(self, payload: dict, match: str) -> None:
        engine = engine_with_answers(ANSWERS)
        with pytest.raises(ValueError, match=match):
            handle_chat_completions(payload, compactor=engine)


class TestCompactContext:
    def test_joins_kept_chunks_verbatim(self) -> None:
        engine = engine_with_answers(ANSWERS)
        context = compact_context("refund policy?", CHUNKS, token_budget=10**6, compactor=engine)
        assert CHUNKS[0] in context
        assert CHUNKS[1] not in context
