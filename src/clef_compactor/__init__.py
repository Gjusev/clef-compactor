"""clef-compactor: Query-aware RAG context compaction using Cloudflare's Clef."""

import os
from dataclasses import dataclass, field
from typing import Optional

import httpx
import tiktoken


@dataclass
class CompactResult:
    """Result of compacting a retrieval batch."""
    kept: list[dict]  # documents that survived, with their scores
    dropped: list[dict]  # documents cut, with score and reason
    total_input_tokens: int
    total_output_tokens: int
    token_budget: int
    raw_response: dict = field(default_factory=dict, repr=False)


@dataclass
class ClefCompactor:
    """Score and compact RAG retrieval batches using Cloudflare's Clef."""

    account_id: str = field(default_factory=lambda: os.environ.get("CLEF_ACCOUNT_ID", ""))
    api_token: str = field(default_factory=lambda: os.environ.get("CLEF_API_TOKEN", ""))
    model: str = "@cf/cloudflare/clef"
    base_url: str = "https://api.cloudflare.com/client/v4"
    timeout: float = 60.0
    encoding_model: str = "cl100k_base"  # for token counting

    def __post_init__(self):
        if not self.account_id:
            raise ValueError("Set CLEF_ACCOUNT_ID env var or pass account_id")
        if not self.api_token:
            raise ValueError("Set CLEF_API_TOKEN env var or pass api_token")

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}/accounts/{self.account_id}/ai/run/{self.model}"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}

    def _count_tokens(self, text: str) -> int:
        enc = tiktoken.get_encoding(self.encoding_model)
        return len(enc.encode(text))

    def compact(
        self,
        question: str,
        documents: list[str],
        token_budget: int = 1000,
    ) -> CompactResult:
        """Score each document against the question and keep the best within budget."""
        if not documents:
            return CompactResult(kept=[], dropped=[], total_input_tokens=0,
                                 total_output_tokens=0, token_budget=token_budget)

        state = f"Question: {question}\n\nDocuments to score:\n"
        questions = {}
        for i, doc in enumerate(documents):
            snippet = doc[:300]  # truncate for context efficiency
            state += f"\n[Doc {i+1}]: {snippet}...\n"
            questions[f"relevance_{i+1}"] = {
                "type": "score",
                "context": f"How relevant is document {i+1} to answering the question? "
                           "essential = directly answers the question. "
                           "relevant = provides supporting information. "
                           "background = tangentially related. "
                           "irrelevant = not useful.",
                "levels": ["irrelevant", "background", "relevant", "essential"],
            }

        payload = {"state": state, "questions": questions}
        resp = httpx.post(
            self._endpoint, json=payload, headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json().get("result", {})

        score_map = {"irrelevant": 0, "background": 1, "relevant": 2, "essential": 3}
        scored = []
        for i, doc in enumerate(documents):
            raw_score = data.get(f"relevance_{i+1}", "irrelevant")
            score = score_map.get(raw_score, 0)
            tokens = self._count_tokens(doc)
            scored.append({"doc": doc, "score": score, "raw": raw_score, "tokens": tokens})

        # Sort by score descending, keep within budget
        scored.sort(key=lambda x: x["score"], reverse=True)
        kept, dropped = [], []
        running_tokens = 0
        for item in scored:
            if item["score"] <= 0:
                dropped.append({**item, "reason": "irrelevant"})
            elif running_tokens + item["tokens"] <= token_budget:
                kept.append(item)
                running_tokens += item["tokens"]
            else:
                dropped.append({**item, "reason": "budget_exhausted"})

        total_input = sum(item["tokens"] for item in scored)
        return CompactResult(
            kept=kept, dropped=dropped,
            total_input_tokens=total_input,
            total_output_tokens=running_tokens,
            token_budget=token_budget,
        )


__version__ = "0.1.0"
__all__ = ["ClefCompactor", "CompactResult"]
