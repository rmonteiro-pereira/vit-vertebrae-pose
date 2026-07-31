"""Evaluation metrics reported by the benchmark.

The headline metric is the mean, over validation images, of the mean Euclidean
distance between predicted and annotated landmarks, expressed in pixels of the
network's input frame.  Aggregating per image first (rather than pooling all
landmarks) is what makes the confidence intervals in :mod:`vitvert.statistics`
valid: the resampling unit is then one patient frame, not one landmark, and the two
landmarks on the same frame are not treated as independent observations.
"""

from __future__ import annotations

import torch

from vitvert.losses import VISIBILITY_THRESHOLD

__all__ = ["mean_pixel_error", "pck", "per_image_pixel_error"]


def per_image_pixel_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    image_size: float = 224.0,
) -> torch.Tensor:
    """Mean landmark distance for each image in the batch.

    Args:
        pred: ``(B, K, 2)`` normalised predictions.
        target: ``(B, K, 2)`` or ``(B, K, 3)`` normalised targets; the third channel,
            when present, is visibility.
        image_size: side length used to convert normalised units to pixels.

    Returns:
        ``(B,)`` tensor of mean pixel errors.  Images with no visible landmark yield
        ``nan`` -- deliberately, so they propagate visibly instead of being averaged
        in as a flattering zero. :func:`vitvert.statistics.error_summary` drops them.

    Raises:
        ValueError: on rank or shape mismatch.
    """
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError(
            f"expected rank-3 tensors (B, K, C), got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.shape[:2] != target.shape[:2]:
        raise ValueError(f"batch/keypoint mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")

    distances = torch.linalg.vector_norm((pred[..., :2] - target[..., :2]) * image_size, dim=-1)
    if target.shape[-1] >= 3:
        visible = target[..., 2] > VISIBILITY_THRESHOLD
    else:
        visible = torch.ones_like(distances, dtype=torch.bool)

    counts = visible.sum(dim=1)
    totals = torch.where(visible, distances, torch.zeros_like(distances)).sum(dim=1)
    return torch.where(
        counts > 0,
        totals / counts.clamp_min(1),
        torch.full_like(totals, float("nan")),
    )


def mean_pixel_error(pred: torch.Tensor, target: torch.Tensor, *, image_size: float = 224.0) -> float:
    """Batch mean of :func:`per_image_pixel_error`, ignoring images with no landmarks."""
    errors = per_image_pixel_error(pred, target, image_size=image_size)
    finite = errors[torch.isfinite(errors)]
    return float(finite.mean()) if finite.numel() else float("nan")


def pck(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold_px: float,
    image_size: float = 224.0,
) -> float:
    """Percentage of Correct Keypoints: share of visible landmarks within ``threshold_px``.

    Raises:
        ValueError: if ``threshold_px`` is not positive.
    """
    if threshold_px <= 0:
        raise ValueError(f"threshold_px must be positive, got {threshold_px!r}")

    distances = torch.linalg.vector_norm((pred[..., :2] - target[..., :2]) * image_size, dim=-1)
    if target.shape[-1] >= 3:
        visible = target[..., 2] > VISIBILITY_THRESHOLD
    else:
        visible = torch.ones_like(distances, dtype=torch.bool)

    total = int(visible.sum())
    if total == 0:
        return float("nan")
    correct = int(((distances <= threshold_px) & visible).sum())
    return 100.0 * correct / total
