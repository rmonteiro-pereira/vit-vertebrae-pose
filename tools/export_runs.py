#!/usr/bin/env python3
"""Export the run archive of the private research repo into ``results/runs.jsonl``.

This script is the *only* bridge between the private research repository (which holds
the licensed dataset, the model weights and the raw run tree) and this public
repository.  It reads the per-run artefacts that the training code wrote and emits a
single newline-delimited JSON file containing **metrics only**.

What is exported per run
------------------------
``model_hash``   short content hash that identified the run in the private tree
``config``       training configuration (model, loss, epochs, augmentation flags, ...)
``history``      per-epoch train/valid loss and per-epoch train/valid pixel error
``pixel_error``  final validation pixel-error summary with 95% confidence intervals
``n_train`` / ``n_valid``  split sizes

What is deliberately **not** exported
-------------------------------------
* image paths, file names, directory names of the private machine
* per-image predictions or errors (they are keyed to individual patient studies)
* anything that could be inverted back to a radiograph

Usage (from the root of the *private* research repo)::

    uv run python /path/to/vit-vertebrae-pose/tools/export_runs.py \
        --models-dir models --out /path/to/vit-vertebrae-pose/results/runs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Keys copied verbatim from the private ``summary.json`` config block.  Anything not on
# this list is dropped, which is what keeps machine-local paths out of the export.
CONFIG_ALLOWLIST = (
    "model_name",
    "loss_type",
    "num_keypoints",
    "architecture",
    "num_epochs",
    "batch_size",
    "learning_rate",
    "k_folds",
    "scheduler_type",
    "freeze_backbone",
    "augmentation_params",
    "dataset_expansion",
    "few_shot",
    "seed",
)

HISTORY_KEYS = (
    "train_losses",
    "valid_losses",
    "train_pixel_errors",
    "valid_pixel_errors",
)

PIXEL_ERROR_KEYS = (
    "mean",
    "median",
    "mean_ci_half_width",
    "mean_ci_low",
    "mean_ci_high",
    "median_ci_half_width",
    "median_ci_low",
    "median_ci_high",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload


def _find_experiment_results(model_dir: Path) -> dict[str, Any] | None:
    for candidate in sorted(model_dir.glob("run_*/results/experiment_results.json")):
        data = _read_json(candidate)
        if data is not None:
            return data
    return None


def export_run(model_dir: Path) -> dict[str, Any] | None:
    """Build the public record for one run directory, or ``None`` if unusable."""
    summary = _read_json(model_dir / "summary.json")
    results = _find_experiment_results(model_dir)
    if summary is None or results is None:
        return None

    fold = (results.get("results") or {}).get("fold_1") or {}
    history = {key: fold.get(key) or [] for key in HISTORY_KEYS}
    if not history["valid_losses"]:
        return None

    model_hash = summary.get("model_hash")
    if not model_hash:
        return None

    config = {k: v for k, v in (summary.get("config") or {}).items() if k in CONFIG_ALLOWLIST}

    pixel_raw = (results.get("summary") or {}).get("pixel_error") or {}
    pixel = {k: pixel_raw[k] for k in PIXEL_ERROR_KEYS if k in pixel_raw}

    return {
        "model_hash": model_hash,
        "config": config,
        "history": history,
        "pixel_error": pixel or None,
        "n_train": fold.get("num_train_samples"),
        "n_valid": fold.get("num_valid_samples"),
        "is_complete": bool(summary.get("is_complete")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not args.models_dir.is_dir():
        parser.error(f"{args.models_dir} is not a directory")

    records = []
    for model_dir in sorted(p for p in args.models_dir.iterdir() if p.is_dir()):
        record = export_run(model_dir)
        if record is not None:
            records.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"wrote {len(records)} runs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
