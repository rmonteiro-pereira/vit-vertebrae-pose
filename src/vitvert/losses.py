"""Keypoint regression objectives compared in the benchmark.

All losses take normalised predictions and targets and evaluate the error **in
pixels**, so that a loss value is directly comparable to the reported pixel-error
metric. Shapes:

``pred``    ``(B, K, 2)`` -- normalised ``(x, y)`` in ``[0, 1]``
``target``  ``(B, K, 2)`` or ``(B, K, 3)``; the optional third channel is visibility

Visibility handling
-------------------
Every loss here honours the visibility channel: keypoints with ``visibility <= 0.1``
are excluded.  In the original research code only six of the eleven losses did, and
the rest silently trained on the ``(0, 0)`` padding written for occluded landmarks.
Making it uniform is a behaviour change and is recorded in
``docs/adr/0004-uniform-visibility-masking.md``; the published L1 results are not
affected, because the published runs use ``l1``, which already masked correctly.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Final

import torch
import torch.nn.functional as F

from vitvert.loss_names import available_losses, canonical_loss_name

__all__ = [
    "LOSS_REGISTRY",
    "adaptive_wing_loss",
    "available_losses",
    "cosine_loss",
    "euclidean_loss",
    "exponential_loss",
    "focal_loss",
    "get_loss_function",
    "huber_loss",
    "l1_loss",
    "mse_loss",
    "multi_threshold_loss",
    "wing_loss",
]

#: Landmarks with visibility at or below this value are treated as absent.
VISIBILITY_THRESHOLD: Final = 0.1

#: Side length in pixels that normalised coordinates are scaled by.
DEFAULT_IMAGE_SIZE: Final = 224.0

LossFn = Callable[..., torch.Tensor]


def _to_pixels(
    pred: torch.Tensor,
    target: torch.Tensor,
    image_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select visible landmarks and convert both tensors to pixel coordinates.

    Returns:
        ``(pred_px, target_px)``, each ``(N, 2)`` where ``N`` is the number of visible
        landmarks in the batch. ``N`` may be zero.

    Raises:
        ValueError: if the tensors are not rank 3, disagree in shape, or the last
            axis is neither 2 nor 3.
    """
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError(
            f"expected rank-3 tensors (B, K, C), got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.shape[:2] != target.shape[:2]:
        raise ValueError(f"batch/keypoint mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.shape[-1] < 2 or target.shape[-1] < 2:
        raise ValueError("last axis must hold at least (x, y)")

    pred_xy = pred[..., :2]
    target_xy = target[..., :2]
    if target.shape[-1] >= 3:
        visible = target[..., 2] > VISIBILITY_THRESHOLD
    else:
        visible = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)

    return pred_xy[visible] * image_size, target_xy[visible] * image_size


def _empty_loss(reference: torch.Tensor) -> torch.Tensor:
    """A differentiable zero, for batches in which nothing is visible."""
    return (reference * 0.0).sum()


def _distances(pred_px: torch.Tensor, target_px: torch.Tensor) -> torch.Tensor:
    """Euclidean distance per landmark, with an epsilon that keeps ``sqrt`` differentiable at 0."""
    return torch.sqrt(torch.sum((pred_px - target_px) ** 2, dim=-1) + 1e-8)


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")


def euclidean_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    image_size: float = DEFAULT_IMAGE_SIZE,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean Euclidean distance in pixels -- the metric itself used as an objective."""
    pred_px, target_px = _to_pixels(pred, target, image_size)
    if pred_px.numel() == 0:
        return _empty_loss(pred)
    return _reduce(_distances(pred_px, target_px), reduction)


def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    image_size: float = DEFAULT_IMAGE_SIZE,
    reduction: str = "mean",
) -> torch.Tensor:
    """Squared error on pixel coordinates."""
    pred_px, target_px = _to_pixels(pred, target, image_size)
    if pred_px.numel() == 0:
        return _empty_loss(pred)
    return F.mse_loss(pred_px, target_px, reduction=reduction)


def l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    image_size: float = DEFAULT_IMAGE_SIZE,
    reduction: str = "mean",
) -> torch.Tensor:
    """Absolute error on pixel coordinates. This is the objective every published run uses."""
    pred_px, target_px = _to_pixels(pred, target, image_size)
    if pred_px.numel() == 0:
        return _empty_loss(pred)
    return F.l1_loss(pred_px, target_px, reduction=reduction)


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    delta_px: float = 5.0,
    image_size: float = DEFAULT_IMAGE_SIZE,
    reduction: str = "mean",
) -> torch.Tensor:
    """Smooth L1: quadratic within ``delta_px`` pixels, linear beyond."""
    pred_px, target_px = _to_pixels(pred, target, image_size)
    if pred_px.numel() == 0:
        return _empty_loss(pred)
    return F.smooth_l1_loss(pred_px, target_px, reduction=reduction, beta=delta_px)


def multi_threshold_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    thresholds_px: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0),
    weights: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    image_size: float = DEFAULT_IMAGE_SIZE,
    reduction: str = "mean",
) -> torch.Tensor:
    """Piecewise-linear penalty whose slope increases with the error band.

    Raises:
        ValueError: if ``thresholds_px`` and ``weights`` differ in length, or
            ``thresholds_px`` is not strictly increasing.
    """
    if len(thresholds_px) != len(weights):
        raise ValueError(
            f"thresholds_px and weights must have equal length, got {len(thresholds_px)} and {len(weights)}"
        )
    if any(b <= a for a, b in itertools.pairwise(thresholds_px)):
        raise ValueError(f"thresholds_px must be strictly increasing, got {thresholds_px!r}")

    pred_px, target_px = _to_pixels(pred, target, image_size)
    if pred_px.numel() == 0:
        return _empty_loss(pred)

    distances = _distances(pred_px, target_px)
    loss = torch.zeros_like(distances)
    lower = 0.0
    for threshold, weight in zip(thresholds_px, weights, strict=True):
        band = (distances > lower) & (distances <= threshold)
        loss = torch.where(band, weight * (distances - lower), loss)
        lower = threshold
    tail = distances > lower
    loss = torch.where(tail, weights[-1] * (distances - lower), loss)
    return _reduce(loss, reduction)


def exponential_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    scale_px: float = 10.0,
    max_penalty: float = 100.0,
    image_size: float = DEFAULT_IMAGE_SIZE,
    reduction: str = "mean",
) -> torch.Tensor:
    """``exp(d / scale) - 1`` clamped at ``max_penalty``; punishes gross misses hard."""
    pred_px, target_px = _to_pixels(pred, target, image_size)
    if pred_px.numel() == 0:
        return _empty_loss(pred)
    distances = _distances(pred_px, target_px)
    loss = torch.clamp(torch.exp(distances / scale_px) - 1.0, max=max_penalty)
    return _reduce(loss, reduction)


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 1.0,
    gamma: float = 2.0,
    image_size: float = DEFAULT_IMAGE_SIZE,
    reduction: str = "mean",
) -> torch.Tensor:
    """Distance weighted by a focal factor, so easy landmarks contribute less."""
    pred_px, target_px = _to_pixels(pred, target, image_size)
    if pred_px.numel() == 0:
        return _empty_loss(pred)
    distances = _distances(pred_px, target_px)
    normalised = torch.sigmoid(distances / (image_size / 50.0))
    return _reduce(alpha * normalised.pow(gamma) * distances, reduction)


def wing_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    omega: float = 10.0,
    epsilon: float = 2.0,
    image_size: float = DEFAULT_IMAGE_SIZE,
    reduction: str = "mean",
) -> torch.Tensor:
    """Wing loss (Feng et al., 2018), applied per axis then summed."""
    pred_px, target_px = _to_pixels(pred, target, image_size)
    if pred_px.numel() == 0:
        return _empty_loss(pred)

    difference = torch.abs(pred_px - target_px)
    constant = omega - omega * torch.log1p(torch.tensor(omega / epsilon, device=difference.device))
    loss = torch.where(
        difference < omega,
        omega * torch.log1p(difference / epsilon),
        difference - constant,
    )
    return _reduce(loss.sum(dim=-1), reduction)


def adaptive_wing_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    omega: float = 14.0,
    theta: float = 0.5,
    epsilon: float = 1.0,
    alpha: float = 2.1,
    image_size: float = DEFAULT_IMAGE_SIZE,
    reduction: str = "mean",
) -> torch.Tensor:
    """Adaptive wing loss (Wang et al., 2019) on Euclidean distances."""
    pred_px, target_px = _to_pixels(pred, target, image_size)
    if pred_px.numel() == 0:
        return _empty_loss(pred)

    distances = _distances(pred_px, target_px)
    ratio = theta / epsilon
    slope = omega * (1.0 / (1.0 + ratio ** (alpha - 1.0))) * (alpha - 1.0) * ratio ** (alpha - 2.0) / epsilon
    intercept = theta * slope - omega * torch.log1p(
        torch.tensor(ratio ** (alpha - 1.0), device=distances.device)
    )
    loss = torch.where(
        distances < theta,
        omega * torch.log1p((distances / epsilon).clamp_min(0.0) ** (alpha - 1.0)),
        slope * distances - intercept,
    )
    return _reduce(loss, reduction)


def cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    image_size: float = DEFAULT_IMAGE_SIZE,
    reduction: str = "mean",
) -> torch.Tensor:
    """70% angular distance plus 30% diagonal-normalised Euclidean distance.

    The Euclidean term is what makes this trainable: pure cosine distance is blind to
    magnitude, so a prediction on the correct ray but at the wrong radius would incur
    no penalty at all.
    """
    pred_px, target_px = _to_pixels(pred, target, image_size)
    if pred_px.numel() == 0:
        return _empty_loss(pred)

    similarity = F.cosine_similarity(pred_px, target_px, dim=-1, eps=1e-8).clamp(-1.0, 1.0)
    diagonal = image_size * (2.0**0.5)
    euclidean = torch.norm(pred_px - target_px, dim=-1) / diagonal
    return _reduce(0.7 * (1.0 - similarity) + 0.3 * euclidean, reduction)


#: Canonical name -> implementation. Keys must match :data:`vitvert.loss_names.LOSS_NAMES`;
#: ``tests/test_losses.py`` enforces it so the torch-free name list cannot drift.
LOSS_REGISTRY: Final[dict[str, LossFn]] = {
    "adaptive_wing": adaptive_wing_loss,
    "cosine": cosine_loss,
    "euclidean": euclidean_loss,
    "exponential": exponential_loss,
    "focal": focal_loss,
    "huber": huber_loss,
    "l1": l1_loss,
    "mse": mse_loss,
    "multithreshold": multi_threshold_loss,
    "wing": wing_loss,
}


def get_loss_function(name: str) -> LossFn:
    """Resolve a loss name (or alias) to its implementation.

    Args:
        name: canonical name or alias; case-insensitive.

    Raises:
        KeyError: if the name is unknown.
    """
    return LOSS_REGISTRY[canonical_loss_name(name)]
