"""Torch dataset over the licensed cervical-landmark frames.

The dataset is **not** distributed with this repository.  See ``docs/dataset.md`` for
the expected on-disk layout and for how to obtain the source material.

Windows note
------------
``torch.utils.data`` on Windows uses ``spawn``, so every worker re-imports the module
and re-constructs the dataset object.  Anything expensive done in ``__init__`` is
therefore paid once per worker per epoch unless ``persistent_workers=True``.  This
class keeps ``__init__`` to a directory listing and defers the image processor to
first use, so worker start-up stays cheap.  ``docs/engineering.md`` has the
measurements that motivated it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vitvert.data.annotations import label_path_for, read_annotation
from vitvert.data.augment import AugmentationPipeline

__all__ = ["IMAGE_SUFFIXES", "KeypointDataset", "Sample"]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class Sample:
    """One batch element."""

    pixel_values: torch.Tensor  # (C, H, W)
    keypoints: torch.Tensor  # (K, 3) normalised x, y, visibility
    label: torch.Tensor  # (1,) class id

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Batch-element mapping consumed by the training loop."""
        return {"pixel_values": self.pixel_values, "keypoints": self.keypoints, "labels": self.label}


class KeypointDataset(Dataset[dict[str, torch.Tensor]]):
    """Images plus normalised landmarks, ready for a ViT image processor.

    Args:
        root: a split directory containing ``images/`` and ``labels/``.
        processor: a HuggingFace image processor; loaded lazily by the caller and
            passed in, so the dataset itself stays picklable and cheap to construct.
        augment: pipeline applied to image and landmarks together, or ``None``.
        max_keypoints: landmark slots per sample; extra slots are padded invisible.

    Raises:
        FileNotFoundError: if ``root`` or its ``images`` subdirectory is missing, or
            if the split contains no images. An empty split silently yielding zero
            batches is the kind of failure that shows up three hours later as a flat
            loss curve.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        processor: Any,
        augment: AugmentationPipeline | None = None,
        max_keypoints: int = 2,
    ) -> None:
        self.root = Path(root)
        images_dir = self.root / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"{images_dir} not found. Expected <split>/images and <split>/labels; see docs/dataset.md."
            )

        self.image_paths: Sequence[Path] = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.image_paths:
            raise FileNotFoundError(f"no images with suffixes {IMAGE_SUFFIXES} under {images_dir}")

        self.processor = processor
        self.augment = augment
        self.max_keypoints = max_keypoints

    def __len__(self) -> int:
        """Number of images in the split."""
        return len(self.image_paths)

    def _load_keypoints(self, image_path: Path) -> tuple[np.ndarray, int]:
        annotation = read_annotation(label_path_for(image_path))
        if annotation is None:
            return np.zeros((self.max_keypoints, 3), dtype=np.float32), 0
        return annotation.padded(self.max_keypoints), annotation.class_id

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load one image and its landmarks, augment, and preprocess."""
        image_path = self.image_paths[index]
        keypoints, class_id = self._load_keypoints(image_path)

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")

        if self.augment is not None:
            # The image identifier drives the deterministic seed, so it must be
            # stable across machines: use the name, never the absolute path.
            image, keypoints = self.augment(image, keypoints, image_id=image_path.name)

        processed = self.processor(images=image, return_tensors="pt")
        sample = Sample(
            pixel_values=processed["pixel_values"].squeeze(0),
            keypoints=torch.from_numpy(keypoints),
            label=torch.tensor([class_id], dtype=torch.long),
        )
        return sample.as_dict()
