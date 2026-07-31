"""Prediction heads shared by every backbone in the benchmark.

Holding the head fixed is what makes the architecture comparison a comparison of
*backbones*: the same two-layer MLP, the same dropout, the same sigmoid output
parameterisation sits on top of every model, so a difference in pixel error is
attributable to the representation rather than to head capacity.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ClassificationHead", "KeypointHead"]


class KeypointHead(nn.Module):
    """Maps a pooled backbone feature to ``num_keypoints`` normalised coordinates.

    The sigmoid output confines predictions to the image frame.  It also bounds the
    gradient near saturation, which is the reason the frozen-backbone runs converge
    to a plausible-looking loss while sitting far from the landmarks: the head alone
    can only shift a global prior, and the sigmoid keeps that prior inside the frame.
    """

    def __init__(self, hidden_size: int, num_keypoints: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_keypoints * 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map pooled features to ``(B, K, 2)`` normalised coordinates."""
        return torch.sigmoid(self.mlp(features)).view(-1, self.num_keypoints, 2)


class ClassificationHead(nn.Module):
    """Auxiliary class logits, retained so archived checkpoints stay loadable.

    The benchmark uses a single class, so this head contributes nothing to the
    reported metrics.  It is kept because removing it would change the state-dict
    layout and orphan every archived checkpoint.
    """

    def __init__(self, hidden_size: int, num_classes: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map pooled features to class logits."""
        logits: torch.Tensor = self.mlp(features)
        return logits
