"""CLI for clef-compactor."""

import argparse
import json

from . import ClefCompactor


def main():
    p = argparse.ArgumentParser(prog="clef-compact", description="Compact RAG context using Clef")
    p.add_argument("--question", "-q", required=True)
    p.add_argument("--docs", "-d", nargs="+", required=True, help="Document texts")
    p.add_argument("--budget", "-b", type=int, default=1000)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    compactor = ClefCompactor()
    result = compactor.compact(args.question, args.docs, token_budget=args.budget)

    if args.json:
        print(json.dumps({
            "kept": len(result.kept),
            "dropped": len(result.dropped),
            "input_tokens": result.total_input_tokens,
            "output_tokens": result.total_output_tokens,
            "saved_pct": round(100 * (1 - result.total_output_tokens / max(result.total_input_tokens, 1)), 1),
        }, indent=2))
    else:
        print(f"kept:          {len(result.kept)} docs")
        print(f"dropped:       {len(result.dropped)} docs")
        print(f"input tokens:  {result.total_input_tokens}")
        print(f"output tokens: {result.total_output_tokens}")
        print(f"saved:         {100 * (1 - result.total_output_tokens / max(result.total_input_tokens, 1)):.1f}%")


if __name__ == "__main__":
    main()
