#!/usr/bin/env python3
"""Compare a fresh measurement JSON against a committed baseline.

Part of the clef-evals regression gate. Metric specs look like::

    metrics.chunk_accuracy:max:0.02
    metrics.cost_usd_per_1k_calls:min:0.005

``max`` goals cover accuracy-like metrics: the current value must stay at or
above ``baseline - tolerance``. ``min`` goals cover cost- and error-like
metrics: the current value must stay at or below ``baseline + tolerance``.
Exit code 1 lists every regression.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def dig(payload: dict[str, Any], dotted: str) -> Any:
    """Follow a dotted path into nested dicts."""
    node: Any = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"path {dotted!r} not found")
        node = node[part]
    return node


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", required=True, help="Fresh measurement JSON")
    parser.add_argument("--baseline", required=True, help="Committed baseline JSON")
    parser.add_argument(
        "--metric",
        action="append",
        required=True,
        help="dotted.path:max[:tolerance] or dotted.path:min[:tolerance]",
    )
    args = parser.parse_args(argv)

    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    regressions: list[str] = []
    for spec in args.metric:
        parts = spec.split(":")
        if len(parts) < 2 or parts[1] not in ("max", "min"):
            print(f"error: malformed metric spec {spec!r} (want path:goal[:tolerance])")
            return 1
        path, goal = parts[0], parts[1]
        tolerance = float(parts[2]) if len(parts) > 2 else 0.02
        try:
            now = float(dig(current, path))
            before = float(dig(baseline, path))
        except KeyError as exc:
            print(f"error: {exc}")
            return 1
        if goal == "max" and now < before - tolerance:
            regressions.append(f"{path}: {now:.4f} < baseline {before:.4f} - {tolerance}")
        if goal == "min" and now > before + tolerance:
            regressions.append(f"{path}: {now:.4f} > baseline {before:.4f} + {tolerance}")

    if regressions:
        print("quality regression against baseline:")
        for line in regressions:
            print(f"  - {line}")
        return 1
    print(f"all {len(args.metric)} metric(s) within tolerance of the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
