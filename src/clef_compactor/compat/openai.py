"""OpenAI-compatible surface for clef-compactor.

Maps the OpenAI ``chat.completions`` schema onto context compaction so any
OpenAI SDK client can talk to a compactor. The entry point is
:func:`handle_chat_completions`, a pure function from an OpenAI-style request
body to an OpenAI-style ``chat.completion`` response. Mount it in FastAPI,
Flask or a Lambda handler in a few lines; see the README.

Request contract (the ``clef`` block is optional)::

    {
        "model": "clef-compactor",
        "messages": [
            {"role": "system", "content": "Retrieved context to compact"},
            {"role": "user", "content": "The query"}
        ],
        "clef": {"token_budget": 1000, "chunks": ["...", "..."]}
    }

* With ``clef.chunks``: those chunks are scored against the last user message
  (or ``clef.query``) and the kept ones are returned in
  ``choices[0].message.content``.
* Without ``clef.chunks``: the non-user messages act as the context to
  compact and the last user message is the query.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

from ..core import ClefCompactor

__all__ = ["handle_chat_completions", "compact_context"]

_COMPACTOR_MODEL_NAME = "clef-compactor"


def _default_compactor() -> ClefCompactor:
    return ClefCompactor()


def compact_context(
    query: str,
    chunks: Sequence[str],
    token_budget: int = 1000,
    *,
    compactor: ClefCompactor | None = None,
) -> str:
    """Return the chunks that survive compaction, joined verbatim."""
    engine = compactor or _default_compactor()
    result = engine.compact(query, list(chunks), token_budget=token_budget)
    return "\n\n".join(result.kept_texts())


def _content_of(message: Mapping[str, Any]) -> str:
    """Flatten one message's content, accepting strings or content-part arrays."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, Mapping)]
        return "\n".join(part for part in parts if part)
    return ""


def _split_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    """Split messages into ``(query, context_chunks)``."""
    query = ""
    context: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        text = _content_of(message)
        if role == "user":
            query = text
        else:
            context.append(text)
    return query, [text for text in context if text]


def _request_id(seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"chatcmpl-clef-{digest}"


def handle_chat_completions(
    payload: Mapping[str, Any],
    *,
    compactor: ClefCompactor | None = None,
) -> dict[str, Any]:
    """Handle an OpenAI ``POST /v1/chat/completions`` body and respond in kind.

    Args:
        payload: Parsed JSON request body (OpenAI chat-completions schema with
            the optional ``clef`` extension block).
        compactor: Shared :class:`ClefCompactor`; one is created per call when
            omitted. Prefer passing a long-lived one in real services.

    Returns:
        An OpenAI ``chat.completion`` response dict. The first choice carries
        the compacted context; ``usage`` reports chunk tokens before and after
        compaction; the ``clef`` block summarizes the cut.

    Raises:
        ValueError: If messages are missing, malformed, or no query/chunks
            can be derived from the payload.
    """
    messages = payload.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise ValueError("payload.messages must be a non-empty list")

    extension = payload.get("clef") or {}
    if not isinstance(extension, Mapping):
        raise ValueError("payload.clef must be an object when present")
    token_budget = int(extension.get("token_budget", 1000))
    engine = compactor or _default_compactor()

    if extension.get("chunks"):
        chunks = [str(chunk) for chunk in extension["chunks"]]
        query = str(extension.get("query") or _split_messages(messages)[0])
    else:
        query, chunks = _split_messages(messages)
    if not query:
        raise ValueError("No query found: provide a user message or clef.query")
    if not chunks:
        raise ValueError("No chunks found: provide clef.chunks or non-user messages")

    result = engine.compact(query, chunks, token_budget=token_budget)

    sections = [
        f"[Doc {chunk.index + 1} | P={chunk.score:.2f}]\n{chunk.text}"
        for chunk in result.kept
    ]
    content = "\n\n".join(sections) if sections else "(no chunk passed the relevance bar)"
    return {
        "id": _request_id(query),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", _COMPACTOR_MODEL_NAME),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.total_input_tokens,
            "completion_tokens": result.total_output_tokens,
            "total_tokens": result.total_input_tokens + result.total_output_tokens,
        },
        "clef": {
            "kept": len(result.kept),
            "dropped": len(result.dropped),
            "saved_fraction": round(result.saved_fraction, 4),
        },
    }
