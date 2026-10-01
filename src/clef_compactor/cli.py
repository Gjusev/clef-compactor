"""Command-line interface for clef-compactor.

Examples::

    clef-compact -q "refund policy" -d "doc one" "doc two" --budget 1000
    clef-compact -q "refund policy" -d @chunk1.txt @chunk2.txt --json
    clef-compact -q "..." -d @chunk1.txt --model clef-flash -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .core import ClefCompactor
from .exceptions import ClefError

__all__ = ["build_parser", "main"]


def _read_document(argument: str) -> str:
    """Read a document argument; ``@path`` reads the file, otherwise verbatim."""
    if argument.startswith("@"):
        path = Path(argument[1:])
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"error: cannot read document file {path}: {exc}") from exc
    return argument


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed for tests and help generation)."""
    parser = argparse.ArgumentParser(
        prog="clef-compact",
        description="Compact RAG context with Cloudflare Clef: score chunks "
        "against a query and keep the best within a token budget.",
    )
    parser.add_argument("--query", "-q", required=True, help="The user query")
    parser.add_argument(
        "--chunks",
        "-d",
        nargs="+",
        required=True,
        help="Chunk texts; use @path to read from a file",
    )
    parser.add_argument(
        "--budget",
        "-b",
        type=int,
        default=1000,
        help="Token budget for the compacted context (default: 1000)",
    )
    parser.add_argument(
        "--model",
        choices=("clef", "clef-flash"),
        default=None,
        help="Model selector (default: CLEF_MODEL or clef)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-request timeout in seconds (default: CLEF_TIMEOUT or 60)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Retries per request (default: CLEF_MAX_RETRIES or 2)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log requests and retries to stderr",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _print_human(result_json: dict) -> None:
    """Print a human-readable summary to stdout."""
    print(f"model:         {result_json['model']}")
    print(f"kept:          {len(result_json['kept'])} chunks")
    print(f"dropped:       {len(result_json['dropped'])} chunks")
    print(f"input tokens:  {result_json['total_input_tokens']}")
    print(f"output tokens: {result_json['total_output_tokens']}")
    print(f"saved:         {result_json['saved_fraction'] * 100:.1f}%")
    print(f"cost estimate: ${result_json['cost_estimate']:.6f} (input tokens at $0.24/M)")
    if result_json["dropped"]:
        print("dropped chunks:")
        for chunk in result_json["dropped"]:
            preview = chunk["text"][:60].replace("\n", " ")
            print(f"  - [{chunk['drop_reason']}] {preview}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point returning the process exit code."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        chunks = [_read_document(chunk) for chunk in args.chunks]
        kwargs: dict[str, object] = {}
        if args.model is not None:
            kwargs["model"] = args.model
        if args.timeout is not None:
            kwargs["timeout"] = args.timeout
        if args.retries is not None:
            kwargs["max_retries"] = args.retries
        with ClefCompactor(**kwargs) as compactor:  # type: ignore[arg-type]
            result = compactor.compact(args.query, chunks, token_budget=args.budget)
    except ClefError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
