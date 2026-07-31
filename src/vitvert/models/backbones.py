"""Backbone wrappers for the seven model variants shipped by this package.

Two families are provided:

``vit-base``
    A plain ImageNet-pretrained ViT (``google/vit-base-patch16-224``) with no pose
    prior.  This is the control that the pose-specialised models are measured
    against.

``vitpose-{s,b,l}`` / ``vitpose++-{s,b,l}``
    Pose-specialised transformers loaded from the ``usyd-community/vitpose-plus-*``
    checkpoints.

.. warning::
   **The two ViTPose families load the same weights.**  Original ViTPose checkpoints
   are not published on the Hub, so both ``vitpose-x`` and ``vitpose++-x`` resolve to
   ``usyd-community/vitpose-plus-x``.  At equal scale the two therefore have
   byte-identical parameter counts (30,895,749 / 121,820,421 / 429,944,837) and are
   the same network under two labels.  Any ``vitpose`` vs ``vitpose++`` gap in the
   published results is run-to-run variance, not an architectural effect.  This is
   the single most important caveat in the benchmark; ``docs/limitations.md``
   restates it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import torch
import torch.nn.functional as F
from torch import nn

from vitvert.models.heads import ClassificationHead, KeypointHead

__all__ = ["VITPOSE_SPECS", "BackboneSpec", "PlainViT", "ViTPose"]


@dataclass(frozen=True)
class BackboneSpec:
    """Static description of a ViTPose variant."""

    checkpoint: str
    hidden_size: int
    input_size: tuple[int, int]


#: Every ViTPose entry maps to a ``vitpose-plus`` checkpoint -- see the module warning.
VITPOSE_SPECS: Final[dict[str, BackboneSpec]] = {
    "vitpose-s": BackboneSpec("usyd-community/vitpose-plus-small", 384, (256, 192)),
    "vitpose-b": BackboneSpec("usyd-community/vitpose-plus-base", 768, (256, 192)),
    "vitpose-l": BackboneSpec("usyd-community/vitpose-plus-large", 1024, (256, 192)),
    "vitpose++-s": BackboneSpec("usyd-community/vitpose-plus-small", 384, (256, 192)),
    "vitpose++-b": BackboneSpec("usyd-community/vitpose-plus-base", 768, (256, 192)),
    "vitpose++-l": BackboneSpec("usyd-community/vitpose-plus-large", 1024, (256, 192)),
}


class _KeypointModel(nn.Module):
    """Common freeze/forward contract for the wrapped backbones."""

    backbone: nn.Module

    def freeze_backbone(self) -> int:
        """Disable gradients on the backbone. Returns the number of frozen tensors."""
        frozen = 0
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
            frozen += 1
        return frozen

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PlainViT(_KeypointModel):
    """``vit-base``: ImageNet ViT plus the shared keypoint head.

    Uses the ``[CLS]`` token as the pooled representation, which is what the ViT
    pretraining objective optimised.
    """

    def __init__(
        self,
        *,
        num_keypoints: int = 2,
        num_classes: int = 1,
        checkpoint: str = "google/vit-base-patch16-224",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        from transformers import ViTConfig, ViTModel

        if pretrained:
            self.backbone = ViTModel.from_pretrained(checkpoint)
        else:
            self.backbone = ViTModel(ViTConfig.from_pretrained(checkpoint))

        hidden = int(self.backbone.config.hidden_size)
        self.keypoint_head = KeypointHead(hidden, num_keypoints)
        self.classification_head = ClassificationHead(hidden, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the backbone and both heads."""
        pooled = self.backbone(pixel_values).last_hidden_state[:, 0]
        return {
            "keypoints": self.keypoint_head(pooled),
            "classification": self.classification_head(pooled),
        }


class ViTPose(_KeypointModel):
    """Pose-specialised ViT backbone plus the shared keypoint head.

    The backbone is a mixture-of-experts over source datasets and requires a
    ``dataset_index``; index 0 (COCO) is used throughout.  Inputs are resized to the
    checkpoint's native resolution, because the position embeddings are not
    interpolated by the wrapper.
    """

    def __init__(
        self,
        variant: str,
        *,
        num_keypoints: int = 2,
        num_classes: int = 1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        try:
            spec = VITPOSE_SPECS[variant]
        except KeyError:
            raise KeyError(
                f"unknown ViTPose variant {variant!r}; available: {', '.join(sorted(VITPOSE_SPECS))}"
            ) from None

        from transformers import VitPoseBackbone, VitPoseConfig, VitPoseForPoseEstimation

        if pretrained:
            full: Any = VitPoseForPoseEstimation.from_pretrained(spec.checkpoint)
            self.backbone = full.backbone
            backbone_config = full.config.backbone_config
        else:
            config = VitPoseConfig.from_pretrained(spec.checkpoint)
            backbone_config = config.backbone_config
            self.backbone = VitPoseBackbone(backbone_config)  # type: ignore[arg-type]

        self.variant = variant
        hidden = int(getattr(backbone_config, "hidden_size", spec.hidden_size))
        self.input_size = tuple(getattr(backbone_config, "image_size", spec.input_size))
        self.keypoint_head = KeypointHead(hidden, num_keypoints)
        self.classification_head = ClassificationHead(hidden, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Resize to the checkpoint resolution, run the backbone, then both heads."""
        if tuple(pixel_values.shape[-2:]) != self.input_size:
            pixel_values = F.interpolate(
                pixel_values, size=self.input_size, mode="bilinear", align_corners=False
            )

        dataset_index = torch.zeros(pixel_values.shape[0], dtype=torch.long, device=pixel_values.device)
        features = self.backbone(pixel_values, dataset_index=dataset_index).feature_maps[-1]
        pooled = features.mean(dim=1)
        return {
            "keypoints": self.keypoint_head(pooled),
            "classification": self.classification_head(pooled),
        }
