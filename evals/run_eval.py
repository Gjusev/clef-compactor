"""Reproducible evaluation harness for clef-compaction quality and cost.

Modes
-----
``replay`` (default)
    Runs the real :class:`ClefCompactor.compact` pipeline against a
    deterministic simulated Clef scorer: each gold label is perturbed with a
    seeded Gaussian so relevance probabilities overlap the decision threshold.
    This exercises the whole pipeline (scoring, ranking, budget) and reports
    honest metrics against the dataset's gold labels. The scorer is a
    simulation, so treat these numbers as pipeline validation, not model
    quality.

``live``
    Calls the real Cloudflare Clef API (needs ``CLEF_ACCOUNT_ID`` and
    ``CLEF_API_TOKEN``). Produces the numbers that belong in the README.

Usage::

    python evals/run_eval.py                          # replay, writes results
    python evals/run_eval.py --mode live              # real API
    python evals/run_eval.py --min-accuracy 0.9       # CI gate: exit 1 below
    python evals/run_eval.py --compare-laya           # print blog comparison
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clef_compactor import ClefCompactor
from clef_compactor.config import Settings
from clef_compactor.core import estimate_input_cost
from clef_compactor.models import ClefReply

EVALS_DIR = Path(__file__).parent
DATA_GLOB = str(EVALS_DIR / "data" / "*.jsonl")
DEFAULT_OUT = EVALS_DIR / "results" / "results.json"

#: Cloudflare's published Clef input price, USD per million tokens.
INPUT_PRICE_PER_M = 0.24

#: Blog-of-record numbers (https://blog.cloudflare.com/clef-decision-models/).
LAYA_COMPARISON = {
    "bfcl_case_exact": {"clef": 98.47, "clef-flash": 98.76, "laya": 38.13},
    "toolret_ndcg_at_10": {"clef": 69.19, "clef-flash": 66.43, "laya": 12.69},
    "api_bank_accuracy": {"clef": 91.93, "clef-flash": 93.11, "laya": 11.41},
    "median_latency_ms": {"clef": 209.3, "clef-flash": 38.8, "laya": 5.8},
    "p95_latency_ms": {"clef": 238.6, "clef-flash": 122.4, "laya": 222.5},
}


@dataclass(frozen=True)
class EvalCase:
    """One dataset row: a query, labeled chunks and the token budget."""

    id: str
    query: str
    chunks: list[str]
    labels: list[bool]
    token_budget: int


@dataclass(frozen=True)
class CaseOutcome:
    """Per-case results of one compaction."""

    case_id: str
    correct: int
    total: int
    kept_relevant: int
    total_relevant: int
    kept_irrelevant: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    api_input_tokens: int | None


def load_cases(pattern: str) -> list[EvalCase]:
    """Load and validate every dataset row from *pattern*."""
    cases: list[EvalCase] = []
    paths = sorted(glob_module.glob(pattern))
    if not paths:
        raise SystemExit(f"error: no dataset files match {pattern}")
    for path_str in paths:
        path = Path(path_str)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            documents = row["documents"]
            cases.append(
                EvalCase(
                    id=row["id"],
                    query=row["question"],
                    chunks=[doc["text"] for doc in documents],
                    labels=[bool(doc["relevant"]) for doc in documents],
                    token_budget=int(row.get("token_budget", 1000)),
                )
            )
    if not cases:
        raise SystemExit(f"error: no eval cases found for {pattern}")
    return cases


class SimulatedClefClient:
    """Deterministic stand-in for ClefClient inside :class:`ClefCompactor`.

    Reads the query from the rendered state, looks up gold labels, and emits
    ``noul`` answers perturbed with seeded noise. Relevant chunks score high
    but occasionally near the threshold; irrelevant chunks score low with the
    same overlap, so the harness measures real decision quality.
    """

    def __init__(self, labels_by_query: dict[str, list[bool]], seed: int) -> None:
        self.labels_by_query = labels_by_query
        self.seed = seed
        self.settings = Settings(account_id="eval", api_token="eval", max_retries=0)

    def _probability(self, query: str, position: int, relevant: bool) -> float:
        rng = random.Random(f"{self.seed}:{query}:{position}")
        mean = 0.90 if relevant else 0.12
        return max(0.0, min(1.0, rng.gauss(mean, 0.18)))

    def ask(
        self,
        state: str,
        questions: dict[str, dict[str, Any]],
        **kwargs: Any,
    ) -> ClefReply:
        query = state.splitlines()[0].removeprefix("Question: ").strip()
        labels = self.labels_by_query.get(query)
        if labels is None:
            raise KeyError(f"simulated scorer has no gold labels for query {query!r}")
        answers: dict[str, Any] = {}
        for name in questions:
            position = int(name.removeprefix("chunk_"))
            probability = self._probability(query, position, labels[position - 1])
            answers[name] = {"type": "noul", "noul": round(probability, 4)}
        return ClefReply(
            model="simulated-clef",
            answers=answers,
            usage=None,
            request_id="simulated",
            latency_ms=0.0,
        )


def run_case(engine: ClefCompactor, case: EvalCase) -> CaseOutcome:
    """Compact one case and score the keep/drop decisions against gold."""
    started = time.perf_counter()
    result = engine.compact(case.query, case.chunks, token_budget=case.token_budget)
    latency_ms = (time.perf_counter() - started) * 1000 if result.latency_ms == 0 else result.latency_ms

    decision: dict[int, bool] = {chunk.index: True for chunk in result.kept}
    decision.update({chunk.index: False for chunk in result.dropped})
    correct = sum(1 for index, label in enumerate(case.labels) if decision[index] == label)
    kept_relevant = sum(1 for index, label in enumerate(case.labels) if decision.get(index) and label)
    total_relevant = sum(1 for label in case.labels if label)
    kept_irrelevant = sum(1 for index, label in enumerate(case.labels) if decision.get(index) and not label)
    return CaseOutcome(
        case_id=case.id,
        correct=correct,
        total=len(case.labels),
        kept_relevant=kept_relevant,
        total_relevant=total_relevant,
        kept_irrelevant=kept_irrelevant,
        input_tokens=result.total_input_tokens,
        output_tokens=result.total_output_tokens,
        latency_ms=latency_ms,
        api_input_tokens=result.usage.input_tokens if result.usage else None,
    )


def aggregate(outcomes: Sequence[CaseOutcome], cost_model: str) -> dict[str, Any]:
    """Reduce per-case outcomes into the reported metric block."""
    total_chunks = sum(outcome.total for outcome in outcomes)
    correct = sum(outcome.correct for outcome in outcomes)
    kept_relevant = sum(outcome.kept_relevant for outcome in outcomes)
    total_relevant = sum(outcome.total_relevant for outcome in outcomes)
    kept_irrelevant = sum(outcome.kept_irrelevant for outcome in outcomes)
    kept = kept_relevant + kept_irrelevant
    input_tokens = sum(outcome.input_tokens for outcome in outcomes)
    output_tokens = sum(outcome.output_tokens for outcome in outcomes)
    latencies = sorted(outcome.latency_ms for outcome in outcomes)
    api_input_tokens = sum(outcome.api_input_tokens or 0 for outcome in outcomes)
    precision = kept_relevant / kept if kept else 0.0
    recall = kept_relevant / total_relevant if total_relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def percentile_share(fraction: float) -> float:
        if not latencies:
            return 0.0
        index = min(int(round(fraction * (len(latencies) - 1))), len(latencies) - 1)
        return round(latencies[index], 2)

    calls = len(outcomes)
    billable = api_input_tokens or input_tokens
    cost_per_1k = estimate_input_cost(billable) * (1000 / calls) if calls else 0.0
    return {
        "cases": calls,
        "chunks": total_chunks,
        "chunk_accuracy": round(correct / total_chunks, 4) if total_chunks else 0.0,
        "kept_precision": round(precision, 4),
        "relevant_recall": round(recall, 4),
        "kept_f1": round(f1, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "avg_saved_fraction": round(1 - output_tokens / input_tokens, 4) if input_tokens else 0.0,
        "latency_ms": {"p50": percentile_share(0.50), "p95": percentile_share(0.95), "p99": percentile_share(0.99)},
        "api_input_tokens": api_input_tokens or None,
        "cost_usd_per_1k_calls": round(cost_per_1k, 6),
        "cost_model": cost_model,
    }


def print_laya_comparison() -> None:
    """Print the published Clef vs laya numbers from Cloudflare's blog."""
    print("Published numbers (Cloudflare blog, 'Introducing Clef', 2026):")
    header = f"{'benchmark':<24}{'clef':>9}{'clef-flash':>12}{'laya':>9}"
    print(header)
    print("-" * len(header))
    for name, numbers in LAYA_COMPARISON.items():
        print(
            f"{name:<24}{numbers['clef']:>9}{numbers['clef-flash']:>12}{numbers['laya']:>9}"
        )
    print("\nContext window: clef 65,536 tokens; laya reported at 32k.")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code (1 below --min-accuracy)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("replay", "live", "local"), default="replay")
    parser.add_argument("--model-path", default="Cloudflare/clef-flash",
                        help="HF repo id or snapshot dir for --mode local")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"),
                        help="Torch dtype for --mode local (float16 on T4)")
    parser.add_argument("--device-map", default="auto", help="Accelerate device map for --mode local")
    parser.add_argument("--data", default=DATA_GLOB, help="Glob of dataset .jsonl files")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Path of the results JSON")
    parser.add_argument("--seed", type=int, default=20261001, help="Replay noise seed")
    parser.add_argument("--threshold", type=float, default=0.5, help="Relevance threshold")
    parser.add_argument("--min-accuracy", type=float, default=None, help="CI gate on chunk_accuracy")
    parser.add_argument("--compare-laya", action="store_true", help="Print published comparison")
    args = parser.parse_args(argv)

    cases = load_cases(args.data)
    if args.mode == "live":
        engine = ClefCompactor()
        scorer = "cloudflare-clef (live)"
        transport = "https://api.cloudflare.com"
        cost_model = "input tokens x $0.24/M (Cloudflare published price)"
    elif args.mode == "local":
        from backends import LocalClefClient

        model_label = args.model_path.split("/")[-1]
        client = LocalClefClient(
            Settings(account_id="local", api_token="local", model=model_label, max_retries=0),
            args.model_path,
            dtype=args.dtype,
            device_map=args.device_map,
        )
        engine = ClefCompactor(
            account_id="local",
            api_token="local",
            model=model_label,
            relevance_threshold=args.threshold,
            client=client,  # type: ignore[arg-type]
        )
        scorer = f"open-weights {args.model_path} ({args.dtype}, local GPU)"
        transport = "local torch forward pass"
        cost_model = "self-hosted open weights: no per-token API cost"
    else:
        labels_by_query = {case.query: case.labels for case in cases}
        engine = ClefCompactor(
            account_id="eval",
            api_token="eval",
            relevance_threshold=args.threshold,
            client=SimulatedClefClient(labels_by_query, args.seed),  # type: ignore[arg-type]
        )
        scorer = f"simulated-clef seed={args.seed} (pipeline validation, not model quality)"
        transport = "in-process mock"
        cost_model = "input tokens x $0.24/M (Cloudflare published price)"

    outcomes = [run_case(engine, case) for case in cases]
    metrics = aggregate(outcomes, cost_model)
    payload = {
        "meta": {
            "mode": args.mode,
            "scorer": scorer,
            "transport": transport,
            "dataset": sorted(glob_module.glob(args.data)),
            "relevance_threshold": args.threshold,
            "cost_model": cost_model,
            "price_note": "Output-token pricing is unpublished; only input tokens are ever costed.",
        },
        "metrics": metrics,
        "per_case": [vars(outcome) for outcome in outcomes],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"mode={args.mode}  cases={metrics['cases']}  chunks={metrics['chunks']}")
    print(f"chunk_accuracy={metrics['chunk_accuracy']:.4f}  kept_f1={metrics['kept_f1']:.4f}  "
          f"relevant_recall={metrics['relevant_recall']:.4f}")
    print(f"tokens saved: {metrics['avg_saved_fraction']:.1%}  "
          f"latency p50/p95/p99 ms: {metrics['latency_ms']['p50']}/"
          f"{metrics['latency_ms']['p95']}/{metrics['latency_ms']['p99']}")
    print(f"cost per 1k calls: ${metrics['cost_usd_per_1k_calls']:.4f}")
    print(f"results written to {out_path}")
    if args.compare_laya:
        print()
        print_laya_comparison()
    if args.min_accuracy is not None and metrics["chunk_accuracy"] < args.min_accuracy:
        print(f"gate FAILED: chunk_accuracy {metrics['chunk_accuracy']} < {args.min_accuracy}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
