"""clef-compactor: query-aware RAG context compaction using Cloudflare Clef.

Score every retrieved chunk against the query with Clef ``noul`` questions,
then keep the best chunks within a token budget. Chunks are kept verbatim --
never rewritten -- so citations stay auditable.

Quick start::

    from clef_compactor import ClefCompactor

    compactor = ClefCompactor()  # reads CLEF_ACCOUNT_ID / CLEF_API_TOKEN
    result = compactor.compact(query, chunks, token_budget=1000)
    context = "\\n\\n".join(result.kept_texts())
"""

from .client import AsyncClefClient, ClefClient
from .config import Settings
from .core import AsyncClefCompactor, ClefCompactor, count_tokens
from .exceptions import (
    ClefAPIError,
    ClefAuthError,
    ClefError,
    ClefNetworkError,
    ClefRateLimitError,
    ClefResponseError,
    ClefServerError,
    ClefTimeoutError,
    ConfigurationError,
)
from .models import ClefReply, CompactResult, DropReason, ScoredChunk, TokenUsage

__version__ = "0.2.1"

__all__ = [
    "AsyncClefCompactor",
    "AsyncClefClient",
    "ClefCompactor",
    "ClefClient",
    "CompactResult",
    "ScoredChunk",
    "DropReason",
    "TokenUsage",
    "ClefReply",
    "Settings",
    "ConfigurationError",
    "ClefError",
    "ClefAPIError",
    "ClefAuthError",
    "ClefRateLimitError",
    "ClefServerError",
    "ClefTimeoutError",
    "ClefNetworkError",
    "ClefResponseError",
    "count_tokens",
    "__version__",
]
