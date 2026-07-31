"""Training loop for one fold.

This is a distilled replacement for the three overlapping orchestrators in the
original research repository (a 1,286-line k-fold runner, an 867-line orchestrator
and a 743-line manager, all mutating the same run directory).  The published results
were produced by that original code, not by this loop; what this loop reproduces is
the *procedure* -- same objective, same optimiser, same scheduler, same
best-checkpoint criterion, same per-epoch metric -- not the archived numbers
bit-for-bit.  ``docs/limitations.md`` states that plainly, and
``docs/adr/0005-distilled-training-loop.md`` records why the rewrite was preferred to
a verbatim port.

Deliberately absent, because it belonged to one Windows workstation and not to a
published benchmark: the profiling interceptor, the file-operation logger, the
asynchronous checkpoint queue and the staged-IO layer.  The measurements those tools
produced are preserved in ``docs/engineering.md``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from vitvert.metrics import per_image_pixel_error
from vitvert.statistics import ErrorSummary, error_summary

__all__ = ["EpochResult", "FoldResult", "evaluate", "train_fold", "train_one_epoch"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochResult:
    """Metrics for a single epoch."""

    epoch: int
    train_loss: float
    valid_loss: float
    train_pixel_error: float
    valid_pixel_error: float


@dataclass
class FoldResult:
    """Outcome of training one fold."""

    epochs: list[EpochResult] = field(default_factory=list)
    best_epoch: int | None = None
    best_valid_loss: float | None = None
    best_state_dict: dict[str, Any] | None = None
    validation_error: ErrorSummary | None = None

    @property
    def train_losses(self) -> list[float]:
        """Per-epoch training loss."""
        return [e.train_loss for e in self.epochs]

    @property
    def valid_losses(self) -> list[float]:
        """Per-epoch validation loss."""
        return [e.valid_loss for e in self.epochs]


def _forward(
    model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
    targets = batch["keypoints"].to(device, non_blocking=True)
    predictions = model(pixel_values)["keypoints"]
    return predictions, targets


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    loss_fn: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    image_size: float,
    grad_clip: float | None = 1.0,
) -> tuple[float, float]:
    """Run one training epoch.

    Returns:
        ``(mean_loss, mean_pixel_error)`` over the epoch.  Both are ``nan`` for an
        empty loader rather than ``0.0``, so an accidentally empty split shows up as
        ``nan`` in the history instead of as a suspiciously perfect epoch.
    """
    model.train()
    total_loss = 0.0
    error_sum = 0.0
    seen = 0

    for batch in loader:
        predictions, targets = _forward(model, batch, device)
        loss = loss_fn(predictions, targets, image_size=image_size)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach()) * batch_size
        with torch.no_grad():
            errors = per_image_pixel_error(predictions.detach(), targets, image_size=image_size)
            finite = errors[torch.isfinite(errors)]
            error_sum += float(finite.sum()) if finite.numel() else 0.0
        seen += batch_size

    if seen == 0:
        return float("nan"), float("nan")
    return total_loss / seen, error_sum / seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    loss_fn: Callable[..., torch.Tensor],
    device: torch.device,
    image_size: float,
) -> tuple[float, ErrorSummary]:
    """Evaluate on a validation loader.

    Returns:
        ``(mean_loss, summary)`` where ``summary`` carries the per-image error
        distribution and its 95% intervals.  Errors are collected per image, which is
        the resampling unit the intervals assume -- see :mod:`vitvert.statistics`.
    """
    model.eval()
    total_loss = 0.0
    seen = 0
    per_image: list[float] = []

    for batch in loader:
        predictions, targets = _forward(model, batch, device)
        loss = loss_fn(predictions, targets, image_size=image_size)
        batch_size = int(targets.shape[0])
        total_loss += float(loss) * batch_size
        seen += batch_size
        per_image.extend(float(v) for v in per_image_pixel_error(predictions, targets, image_size=image_size))

    mean_loss = total_loss / seen if seen else float("nan")
    return mean_loss, error_summary(per_image)


def train_fold(
    model: nn.Module,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    valid_loader: DataLoader[dict[str, torch.Tensor]],
    *,
    loss_fn: Callable[..., torch.Tensor],
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    image_size: float = 224.0,
    scheduler_type: str = "plateau",
    on_epoch_end: Callable[[EpochResult], None] | None = None,
) -> FoldResult:
    """Train one fold and keep the state dict with the lowest validation loss.

    Only parameters with ``requires_grad`` are handed to the optimiser, so a frozen
    backbone is genuinely frozen rather than merely receiving zero gradients while
    the optimiser still applies weight decay and momentum to it.

    Raises:
        ValueError: if ``num_epochs`` is not positive or ``scheduler_type`` is
            unknown.
    """
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters: the whole model is frozen")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    if scheduler_type == "plateau":
        scheduler: Any = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs // 3))
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"unknown scheduler_type {scheduler_type!r}")

    result = FoldResult()
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_error = train_one_epoch(
            model,
            train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            image_size=image_size,
        )
        valid_loss, summary = evaluate(
            model, valid_loader, loss_fn=loss_fn, device=device, image_size=image_size
        )
        valid_error = summary.mean if summary.mean is not None else float("nan")

        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(valid_loss)
            else:
                scheduler.step()

        epoch_result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            valid_loss=valid_loss,
            train_pixel_error=train_error,
            valid_pixel_error=valid_error,
        )
        result.epochs.append(epoch_result)

        improved = valid_loss == valid_loss and (  # not nan
            result.best_valid_loss is None or valid_loss < result.best_valid_loss
        )
        if improved:
            result.best_valid_loss = valid_loss
            result.best_epoch = epoch
            result.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            result.validation_error = summary

        logger.info(
            "epoch %d/%d train_loss=%.4f valid_loss=%.4f valid_px=%.3f%s",
            epoch,
            num_epochs,
            train_loss,
            valid_loss,
            valid_error,
            " *" if improved else "",
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch_result)

    return result
