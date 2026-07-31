#!/usr/bin/env python3
"""Regenerate every committed figure from ``results/runs.jsonl``.

Replaces the three exploratory notebooks of the original repository
(``eda.ipynb``, ``visualize_runs.ipynb``, ``visualize_training_curves.ipynb``), whose
committed outputs embedded rendered images and whose 3.4 MB of base64 could not be
reviewed line by line.  Everything this script draws is a chart over aggregate
metrics; it never opens an image, so it cannot emit a radiograph.

Usage::

    uv run python scripts/make_figures.py            # write figures/
    uv run python scripts/make_figures.py --out /tmp # write elsewhere
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from vitvert.results import aggregate_runs, load_runs
from vitvert.results.aggregate import AggregatedRow, baseline_row, relative_improvement
from vitvert.results.records import RunRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = REPO_ROOT / "results" / "runs.jsonl"
DEFAULT_OUT = REPO_ROOT / "figures"

MODEL_ORDER = [
    "vit-base",
    "vitpose-s",
    "vitpose-b",
    "vitpose-l",
    "vitpose++-s",
    "vitpose++-b",
    "vitpose++-l",
    "hrformer-b",
    "transpose-b",
]
CONFIG_ORDER = [
    "Fine-tuned / Control",
    "Fine-tuned / Aug",
    "Fine-tuned / Expand",
    "Frozen / Control",
    "Frozen / Aug",
    "Frozen / Expand",
]

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _errors_by_model(rows: Sequence[AggregatedRow], config: str) -> dict[str, float]:
    """Mean pixel error per model for one configuration, skipping cells with no value."""
    return {
        r.model: r.best_pixel_error for r in rows if r.config == config and r.best_pixel_error is not None
    }


def _grid(rows: Sequence[AggregatedRow], attribute: str) -> np.ndarray:
    lookup = {(r.model, r.config): getattr(r, attribute) for r in rows}
    return np.array(
        [[lookup.get((m, c), np.nan) for c in CONFIG_ORDER] for m in MODEL_ORDER],
        dtype=float,
    )


def figure_error_grid(rows: Sequence[AggregatedRow], out: Path) -> Path:
    """Heat map of mean validation pixel error with 95% interval half-widths."""
    values = _grid(rows, "best_pixel_error")
    half = _grid(rows, "best_pixel_error_ci_half_width")

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    image = ax.imshow(values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(CONFIG_ORDER)), CONFIG_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    ax.set_title("Mean validation pixel error by model and configuration\n(± = 95% CI half-width)")
    ax.grid(False)

    finite = values[np.isfinite(values)]
    midpoint = finite.min() + 0.55 * (finite.max() - finite.min()) if finite.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            label = f"{value:.2f}" if not np.isfinite(half[i, j]) else f"{value:.2f}±{half[i, j]:.2f}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value > midpoint else "black",
            )
    fig.colorbar(image, ax=ax, label="pixel error")
    path = out / "error_grid.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_frozen_penalty(rows: Sequence[AggregatedRow], out: Path) -> Path:
    """Fine-tuned versus frozen mean pixel error, per model."""
    fine = _errors_by_model(rows, "Fine-tuned / Control")
    frozen = _errors_by_model(rows, "Frozen / Control")
    models = [m for m in MODEL_ORDER if m in fine and m in frozen]
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(x - 0.2, [fine[m] for m in models], width=0.4, label="Fine-tuned", color="#2a7f62")
    ax.bar(x + 0.2, [frozen[m] for m in models], width=0.4, label="Frozen backbone", color="#c1462f")
    for i, model in enumerate(models):
        penalty = (frozen[model] - fine[model]) / fine[model] * 100.0
        ax.text(i, max(fine[model], frozen[model]) * 1.03, f"+{penalty:.0f}%", ha="center", fontsize=7.5)
    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.set_ylabel("mean validation pixel error")
    ax.set_title("Freezing the pretrained backbone costs more than any other single choice")
    ax.legend(frameon=False)
    path = out / "frozen_penalty.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_augmentation_effect(rows: Sequence[AggregatedRow], out: Path) -> Path:
    """Per-model change in pixel error from enabling online augmentation."""
    control = _errors_by_model(rows, "Fine-tuned / Control")
    augmented = _errors_by_model(rows, "Fine-tuned / Aug")
    models = [m for m in MODEL_ORDER if m in control and m in augmented]
    deltas = [(augmented[m] - control[m]) / control[m] * 100.0 for m in models]

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(models, deltas, color=["#c1462f" if d > 0 else "#2a7f62" for d in deltas])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("change in pixel error (%)")
    ax.set_title("Augmentation is not uniformly beneficial: sign flips by architecture\n(negative = better)")
    ax.tick_params(axis="x", rotation=20)
    path = out / "augmentation_effect.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_top_models(rows: Sequence[AggregatedRow], out: Path) -> Path:
    """Top fine-tuned cells with their intervals, against the baseline."""
    from vitvert.results.aggregate import top_fine_tuned

    baseline = baseline_row(rows)
    top = top_fine_tuned(rows, limit=8)
    labels = [f"{r.model}\n{r.config.split(' / ')[1]}" for r in top]
    values = [r.best_pixel_error or 0.0 for r in top]
    errors = [r.best_pixel_error_ci_half_width or 0.0 for r in top]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(labels, values, yerr=errors, capsize=3, color="#2a7f62")
    if baseline and baseline.best_pixel_error:
        ax.axhline(
            baseline.best_pixel_error,
            color="#c1462f",
            linestyle="--",
            linewidth=1.2,
            label=f"baseline vitpose-b Control = {baseline.best_pixel_error:.2f} px",
        )
        ax.legend(frameon=False)
    ax.set_ylabel("mean validation pixel error")
    ax.set_title("Best fine-tuned configurations (error bars: 95% CI on the mean)")
    path = out / "top_models.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_training_curves(runs: Sequence[RunRecord], rows: Sequence[AggregatedRow], out: Path) -> Path:
    """Validation loss curves for the top cells plus one frozen counterexample."""
    from vitvert.results.aggregate import top_fine_tuned

    by_hash = {r.model_hash: r for r in runs}
    selected = [
        (r.model, r.config, by_hash[r.hash]) for r in top_fine_tuned(rows, limit=4) if r.hash in by_hash
    ]
    frozen = next(
        (
            (r.model, r.config, by_hash[r.hash])
            for r in rows
            if r.config == "Frozen / Control" and r.model == "vitpose-b" and r.hash in by_hash
        ),
        None,
    )
    if frozen:
        selected.append(frozen)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for model, config, run in selected:
        style = "--" if config.startswith("Frozen") else "-"
        ax.plot(
            range(1, len(run.valid_losses) + 1),
            run.valid_losses,
            style,
            linewidth=1.3,
            label=f"{model} · {config}",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss (L1, pixels)")
    ax.set_yscale("log")
    ax.set_title("Validation loss: best fine-tuned runs versus a frozen backbone")
    ax.legend(frameon=False, fontsize=7.5)
    path = out / "training_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs)
    rows = aggregate_runs(runs)

    written = [
        figure_error_grid(rows, args.out),
        figure_frozen_penalty(rows, args.out),
        figure_augmentation_effect(rows, args.out),
        figure_top_models(rows, args.out),
        figure_training_curves(runs, rows, args.out),
    ]
    for path in written:
        print(f"wrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")

    baseline = baseline_row(rows)
    if baseline:
        best = min(
            (r for r in rows if r.config.startswith("Fine-tuned") and r.best_pixel_error is not None),
            key=lambda r: r.best_pixel_error or math.inf,
        )
        improvement = relative_improvement(best.best_pixel_error, baseline.best_pixel_error)
        error = best.best_pixel_error or math.nan
        print(
            f"best fine-tuned cell: {best.model} {best.config} {error:.2f} px "
            f"({improvement:.1f}% vs baseline)"
            if improvement is not None
            else ""
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
