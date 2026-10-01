# clef-compactor

Query-aware RAG context compaction using Cloudflare's [Clef](https://blog.cloudflare.com/clef-decision-models/).

Your retrieval pipeline returns 12 chunks. Your LLM reads all of them. You pay for all of them. clef-compactor scores each document against the query using Clef's 64k context window and cuts the noise.

**Delete, do not rewrite.** Documents are kept verbatim or removed with an auditable reason. Never summarized.

## Quick start

```bash
pip install clef-compactor
export CLEF_ACCOUNT_ID=your_id
export CLEF_API_TOKEN=your_token
```

```python
from clef_compactor import ClefCompactor

compactor = ClefCompactor()
result = compactor.compact(
    question="What is the refund policy?",
    documents=retrieved_chunks,
    token_budget=1000,
)
# result.kept: documents that survived
# result.dropped: [{doc, score, reason}] for the ones cut
# result.total_output_tokens < result.total_input_tokens
```

## CLI

```bash
clef-compact -q "refund policy" -d "doc1..." "doc2..." --budget 1000
```

## Why Clef for compaction?

Clef's 64k context window means you can score more documents in a single pass than with laya (32k). The quality is higher (BFCL 98.76 vs laya's 38.13), which means fewer false positives when scoring document relevance.

For sub-10ms local compaction, see [laya-compactor](https://github.com/Gjusev/laya-compactor).

## License

Apache 2.0
