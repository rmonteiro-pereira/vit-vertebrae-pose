"""Aggregation rules that turn 169 archived runs into the published comparison grid.

The grid has one row per ``(model, configuration)`` cell, where configuration is one
of ``{Fine-tuned, Frozen} x {Control, Aug, Expand}``.  Several runs can land in the
same cell (repeated seeds, resumed runs).  The published rule -- reproduced here
unchanged from the original research code -- is:

* keep only ``loss_type == "l1"`` runs, so architecture is compared at a fixed
  objective and the loss study is a separate analysis;
* discard runs that stopped before ``max(5, 10% of the configured epoch budget)``;
* report ``best_loss`` as the minimum over the whole cell;
* take **every other** metric from the single run in the cell with the lowest best
  validation loss, and carry that run's hash into the output so each published number
  stays traceable to one run.

That last rule is optimistic -- see ``docs/limitations.md``, "replicate selection".
It is documented rather than quietly fixed, because the published article uses it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from vitvert.results.records import RunRecord

__all__ = [
    "MIN_EPOCHS",
    "MIN_EPOCH_FRACTION",
    "AggregatedRow",
    "aggregate_runs",
    "config_label",
    "image_diagonal_px",
    "relative_improvement",
    "top_fine_tuned",
]

MIN_EPOCHS = 5
MIN_EPOCH_FRACTION = 0.1

#: Cell used as the reference point for every "% better than baseline" claim.
BASELINE_MODEL = "vitpose-b"
BASELINE_CONFIG = "Fine-tuned / Control"


@dataclass(frozen=True)
class AggregatedRow:
    """One cell of the published model x configuration grid."""

    model: str
    config: str
    loss_type: str
    hash: str
    best_loss: float | None
    best_pixel_error: float | None
    median_pixel_error: float | None
    best_pixel_error_ci_half_width: float | None
    median_pixel_error_ci_half_width: float | None

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, for JSON serialisation."""
        return asdict(self)


def config_label(run: RunRecord) -> str:
    """Human-readable configuration label, e.g. ``"Fine-tuned / Aug"``.

    ``Expand`` (dataset expansion) takes precedence over ``Aug`` (online
    augmentation) when a run enables both, matching the published labelling.
    """
    backbone = "Frozen" if run.frozen_backbone else "Fine-tuned"
    if run.expanded:
        policy = "Expand"
    elif run.augmented:
        policy = "Aug"
    else:
        policy = "Control"
    return f"{backbone} / {policy}"


def _eligible(run: RunRecord) -> bool:
    if not run.is_complete or run.loss_type != "l1":
        return False
    budget = run.config.get("num_epochs", 100)
    try:
        minimum = max(MIN_EPOCHS, float(budget) * MIN_EPOCH_FRACTION)
    except (TypeError, ValueError):
        minimum = MIN_EPOCHS
    return run.epochs_trained >= minimum and run.best_valid_loss is not None


def aggregate_runs(runs: Iterable[RunRecord]) -> list[AggregatedRow]:
    """Collapse runs into one row per ``(model, configuration)`` cell.

    Args:
        runs: archived runs, typically from :func:`vitvert.results.load_runs`.

    Returns:
        Rows sorted by model then configuration, so the output is stable across
        machines and can be diffed in review.
    """
    cells: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for run in runs:
        if _eligible(run):
            cells[(run.model_name, config_label(run))].append(run)

    rows: list[AggregatedRow] = []
    for (model, config), group in cells.items():
        losses = [r.best_valid_loss for r in group if r.best_valid_loss is not None]
        canonical = min(group, key=lambda r: r.best_valid_loss if r.best_valid_loss is not None else math.inf)
        rows.append(
            AggregatedRow(
                model=model,
                config=config,
                loss_type="l1",
                hash=canonical.model_hash,
                best_loss=min(losses) if losses else None,
                best_pixel_error=canonical.mean_pixel_error,
                median_pixel_error=canonical.median_pixel_error,
                best_pixel_error_ci_half_width=canonical.ci_half_width("mean"),
                median_pixel_error_ci_half_width=canonical.ci_half_width("median"),
            )
        )
    rows.sort(key=lambda row: (row.model, row.config))
    return rows


def baseline_row(rows: Sequence[AggregatedRow]) -> AggregatedRow | None:
    """The ``vitpose-b`` fine-tuned control cell, or ``None`` if it is absent."""
    for row in rows:
        if row.model == BASELINE_MODEL and row.config == BASELINE_CONFIG:
            return row
    return None


def relative_improvement(error: float | None, baseline_error: float | None) -> float | None:
    """Percentage reduction in pixel error relative to the baseline cell.

    Positive means better than baseline.  Returns ``None`` when either value is
    missing or non-finite, rather than inventing a number.
    """
    if error is None or baseline_error is None:
        return None
    if not (math.isfinite(error) and math.isfinite(baseline_error)) or baseline_error == 0:
        return None
    return (baseline_error - error) / baseline_error * 100.0


def top_fine_tuned(rows: Sequence[AggregatedRow], limit: int = 5) -> list[AggregatedRow]:
    """The ``limit`` fine-tuned cells with the lowest mean pixel error.

    Frozen-backbone cells are excluded: the primary endpoint ranks fine-tuning
    configurations, and the frozen arm is reported separately as an ablation.
    """
    candidates = [
        row
        for row in rows
        if row.config.startswith("Fine-tuned")
        and row.best_pixel_error is not None
        and math.isfinite(row.best_pixel_error)
    ]
    candidates.sort(key=lambda row: (row.best_pixel_error, row.model))
    return candidates[:limit]


def image_diagonal_px(model: str) -> float:
    """Input diagonal in pixels, used to express errors as a percentage of the frame.

    ``vit-base`` consumes square 224x224 inputs; the ViTPose family consumes the
    256x192 crops its checkpoints were trained on.  Mixing the two diagonals is why
    percentage errors are reported per model rather than pooled.
    """
    if "vit-base" in model:
        return math.hypot(224.0, 224.0)
    return math.hypot(256.0, 192.0)
