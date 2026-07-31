"""Typed access to ``results/runs.jsonl``.

One line of that file is one training run.  The file is the *only* record of the
experiments that ships publicly: the images, the weights and the per-image error
vectors stay in the private research repository.  See ``docs/dataset.md``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["RunRecord", "default_runs_path", "load_runs"]


def default_runs_path() -> Path:
    """Path to the committed run archive, resolved relative to the repository root."""
    return Path(__file__).resolve().parents[3] / "results" / "runs.jsonl"


@dataclass(frozen=True)
class RunRecord:
    """A single archived training run.

    Attributes:
        model_hash: short content hash identifying the run in the private tree.
        config: training configuration as recorded when the run started.
        train_losses / valid_losses: per-epoch loss for fold 1.
        train_pixel_errors / valid_pixel_errors: per-epoch mean pixel error for fold 1.
        pixel_error: final validation pixel-error summary with 95% intervals, or
            ``None`` when the run predates the interval instrumentation.
        n_train / n_valid: split sizes.
        is_complete: whether the run reached its configured epoch budget.
    """

    model_hash: str
    config: dict[str, Any]
    train_losses: list[float] = field(default_factory=list)
    valid_losses: list[float] = field(default_factory=list)
    train_pixel_errors: list[float] = field(default_factory=list)
    valid_pixel_errors: list[float] = field(default_factory=list)
    pixel_error: dict[str, float] | None = None
    n_train: int | None = None
    n_valid: int | None = None
    is_complete: bool = False

    # -- derived properties -------------------------------------------------

    @property
    def model_name(self) -> str:
        """Architecture label as recorded at training time."""
        return str(self.config.get("model_name", "unknown"))

    @property
    def loss_type(self) -> str:
        """Objective label as recorded at training time."""
        return str(self.config.get("loss_type", "l1"))

    @property
    def frozen_backbone(self) -> bool:
        """Whether the pretrained backbone was held fixed."""
        return bool(self.config.get("freeze_backbone", False))

    @property
    def augmented(self) -> bool:
        """Whether online augmentation was enabled."""
        params = self.config.get("augmentation_params")
        return bool(params.get("enable_augmentation", False)) if isinstance(params, dict) else False

    @property
    def expanded(self) -> bool:
        """Whether the offline-expanded frame set was used."""
        expansion = self.config.get("dataset_expansion")
        return bool(expansion.get("enabled", False)) if isinstance(expansion, dict) else False

    @property
    def epochs_trained(self) -> int:
        """Number of epochs actually completed."""
        return len(self.valid_losses)

    @property
    def best_valid_loss(self) -> float | None:
        """Lowest finite validation loss, or ``None`` if the run never produced one."""
        finite = [v for v in self.valid_losses if math.isfinite(v)]
        return min(finite) if finite else None

    @property
    def mean_pixel_error(self) -> float | None:
        """Final mean validation pixel error, falling back to the per-epoch minimum."""
        if self.pixel_error and self.pixel_error.get("mean") is not None:
            return float(self.pixel_error["mean"])
        return min(self.valid_pixel_errors) if self.valid_pixel_errors else None

    @property
    def median_pixel_error(self) -> float | None:
        """Final median validation pixel error, or ``None`` if it was not recorded."""
        if self.pixel_error and self.pixel_error.get("median") is not None:
            return float(self.pixel_error["median"])
        return None

    def ci_half_width(self, statistic: str) -> float | None:
        """Half-width of the archived 95% interval for ``"mean"`` or ``"median"``.

        Raises:
            ValueError: for any other statistic name.
        """
        if statistic not in {"mean", "median"}:
            raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}")
        if not self.pixel_error:
            return None
        value = self.pixel_error.get(f"{statistic}_ci_half_width")
        return float(value) if value is not None else None

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RunRecord:
        """Build a record from one decoded ``runs.jsonl`` line.

        Raises:
            KeyError: if the mandatory ``model_hash`` field is absent.
        """
        history = payload.get("history") or {}
        return cls(
            model_hash=str(payload["model_hash"]),
            config=dict(payload.get("config") or {}),
            train_losses=list(history.get("train_losses") or []),
            valid_losses=list(history.get("valid_losses") or []),
            train_pixel_errors=list(history.get("train_pixel_errors") or []),
            valid_pixel_errors=list(history.get("valid_pixel_errors") or []),
            pixel_error=dict(payload["pixel_error"]) if payload.get("pixel_error") else None,
            n_train=payload.get("n_train"),
            n_valid=payload.get("n_valid"),
            is_complete=bool(payload.get("is_complete", False)),
        )


def _iter_lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON line") from exc


def load_runs(path: str | Path | None = None) -> Sequence[RunRecord]:
    """Load every archived run.

    Args:
        path: location of ``runs.jsonl``; defaults to the committed archive.

    Returns:
        Records in file order.

    Raises:
        FileNotFoundError: if the archive is missing.
        ValueError: if a line is not valid JSON or lacks ``model_hash``.
    """
    runs_path = Path(path) if path is not None else default_runs_path()
    if not runs_path.is_file():
        raise FileNotFoundError(
            f"run archive not found at {runs_path}. It ships with the repository; "
            "regenerate it from the private research repo with tools/export_runs.py."
        )
    records = []
    for lineno, payload in enumerate(_iter_lines(runs_path), start=1):
        try:
            records.append(RunRecord.from_json(payload))
        except KeyError as exc:
            raise ValueError(f"{runs_path}:{lineno}: missing field {exc}") from exc
    return records
