#!/usr/bin/env python3
"""Regenerate ``results/aggregated_metrics.json`` from ``results/runs.jsonl``.

This is the script that turns 169 archived runs into the 48-cell comparison grid every
number in the README is quoted from.  It needs no GPU, no dataset and no network::

    uv run python scripts/aggregate_results.py

``--check`` re-derives the grid and exits non-zero if it differs from the committed
file, which is how CI proves that the published numbers are reproducible from the
committed evidence rather than transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vitvert.results import aggregate_runs, load_runs
from vitvert.results.aggregate import baseline_row, relative_improvement, top_fine_tuned

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = REPO_ROOT / "results" / "runs.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "aggregated_metrics.json"


def build_payload(runs_path: Path) -> dict[str, object]:
    """Build the full aggregated-metrics document."""
    runs = load_runs(runs_path)
    rows = aggregate_runs(runs)
    baseline = baseline_row(rows)
    baseline_error = baseline.best_pixel_error if baseline else None

    return {
        "schema_version": 1,
        "source": "results/runs.jsonl",
        "n_runs_scanned": len(runs),
        "n_cells": len(rows),
        "aggregation_rule": (
            "L1 runs only. Runs shorter than max(5, 10% of the configured epoch budget) are "
            "discarded. best_loss is the minimum over the cell; every other metric comes from "
            "the run in the cell with the lowest best validation loss, whose hash is reported."
        ),
        "interval_meaning": (
            "95% intervals describe sampling variability of the error summary on a fixed "
            "validation split. They are not calibrated predictive uncertainty."
        ),
        "baseline": {
            "model": baseline.model if baseline else None,
            "config": baseline.config if baseline else None,
            "mean_pixel_error": baseline_error,
        },
        "top_fine_tuned": [
            {
                **row.as_dict(),
                "improvement_vs_baseline_pct": relative_improvement(row.best_pixel_error, baseline_error),
            }
            for row in top_fine_tuned(rows)
        ],
        "cells": [row.as_dict() for row in rows],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed file matches instead of rewriting it",
    )
    args = parser.parse_args(argv)

    payload = build_payload(args.runs)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not args.output.is_file():
            print(f"missing {args.output}; run without --check to generate it", file=sys.stderr)
            return 1
        if args.output.read_text(encoding="utf-8") != rendered:
            print(
                f"{args.output} is stale: re-run scripts/aggregate_results.py and commit the result",
                file=sys.stderr,
            )
            return 1
        print(f"{args.output} is up to date ({payload['n_cells']} cells)")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote {args.output} ({payload['n_cells']} cells from {payload['n_runs_scanned']} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
