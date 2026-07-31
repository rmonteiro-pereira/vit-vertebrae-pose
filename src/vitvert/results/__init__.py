"""Loading and aggregation of the archived benchmark runs.

This subpackage depends only on the standard library and numpy, so the whole
results/tables pipeline runs on a clean clone without ``torch`` and without the
dataset.
"""

from __future__ import annotations

from vitvert.results.aggregate import (
    AggregatedRow,
    aggregate_runs,
    config_label,
    relative_improvement,
    top_fine_tuned,
)
from vitvert.results.records import RunRecord, load_runs

__all__ = [
    "AggregatedRow",
    "RunRecord",
    "aggregate_runs",
    "config_label",
    "load_runs",
    "relative_improvement",
    "top_fine_tuned",
]
