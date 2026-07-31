"""Deterministic, keypoint-aware augmentation.

Two properties are required of this pipeline and both are enforced by tests.

**Determinism.**  A transform's random parameters are derived from a SHA-256 hash of
the image identifier plus a per-transform salt, never from global RNG state.  The
same image therefore receives the same augmentation in every epoch, on every worker
process, on every machine.  That is what makes the preprocessing cache sound and what
makes a rerun reproduce a run.

**Label consistency.**  Every transform maps the image *and* the landmarks in one
call and returns both.  The original implementation instead applied the transform to
the image in one place and re-derived the very same random parameters in a second
place to move the landmarks -- two copies of the crop geometry that had to stay in
lockstep by inspection.  Any divergence would have silently mislabelled the training
set while every loss curve still looked healthy.  Collapsing it to a single call site
removes the failure mode by construction; see
``docs/adr/0003-single-call-site-augmentation.md``.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image, ImageEnhance

__all__ = [
    "AugmentationPipeline",
    "ColorJitter",
    "CropJitter",
    "HorizontalFlip",
    "KeypointTransform",
    "VerticalFlip",
    "build_pipeline",
    "derive_seed",
]

#: Landmark array layout: ``(K, 3)`` of ``(x_norm, y_norm, visibility)``.
Keypoints = np.ndarray


def derive_seed(image_id: str, salt: int) -> int:
    """Stable 32-bit seed for ``image_id`` under a per-transform ``salt``."""
    digest = hashlib.sha256(image_id.encode("utf-8")).hexdigest()
    return (int(digest[:16], 16) + salt) & 0xFFFFFFFF


class KeypointTransform(Protocol):
    """A transform that maps image and landmarks together."""

    salt: int

    def __call__(
        self, image: Image.Image, keypoints: Keypoints, *, image_id: str
    ) -> tuple[Image.Image, Keypoints]:
        """Apply the transform to image and landmarks, returning both."""
        ...


class _Base:
    salt: int = 0

    def _rng(self, image_id: str) -> random.Random:
        return random.Random(derive_seed(image_id, self.salt))


@dataclass
class CropJitter(_Base):
    """Random centre crop, then resize back to ``output_size``.

    Landmarks that fall outside the crop are marked invisible rather than clamped to
    the border: a clamped landmark is a wrong label, an invisible one is an honest
    absence that every loss in :mod:`vitvert.losses` skips.
    """

    output_size: tuple[int, int]
    scale_range: tuple[float, float] = (0.9, 1.0)
    shift_range: tuple[float, float] = (-0.02, 0.02)
    salt: int = 11

    def __call__(
        self, image: Image.Image, keypoints: Keypoints, *, image_id: str
    ) -> tuple[Image.Image, Keypoints]:
        """Crop, resize, and map the landmarks into the new frame."""
        rng = self._rng(image_id)
        width, height = image.size

        scale = min(max(rng.uniform(*self.scale_range), 0.5), 1.0)
        crop_w = max(int(width * scale), 1)
        crop_h = max(int(height * scale), 1)
        max_dx = max(width - crop_w, 0)
        max_dy = max(height - crop_h, 0)

        shift_x = rng.uniform(*self.shift_range)
        shift_y = rng.uniform(*self.shift_range)
        offset_x = int(width / 2 + shift_x * max_dx - crop_w / 2)
        offset_y = int(height / 2 + shift_y * max_dy - crop_h / 2)
        offset_x = max(0, min(offset_x, max_dx))
        offset_y = max(0, min(offset_y, max_dy))

        cropped = image.crop((offset_x, offset_y, offset_x + crop_w, offset_y + crop_h))
        cropped = cropped.resize(self.output_size, Image.Resampling.BILINEAR)

        moved = keypoints.copy()
        visible = moved[:, 2] > 0
        x_rel = (moved[visible, 0] * width - offset_x) / crop_w
        y_rel = (moved[visible, 1] * height - offset_y) / crop_h
        inside = (x_rel >= 0.0) & (x_rel <= 1.0) & (y_rel >= 0.0) & (y_rel <= 1.0)

        index = np.flatnonzero(visible)
        moved[index[inside], 0] = x_rel[inside]
        moved[index[inside], 1] = y_rel[inside]
        moved[index[~inside]] = 0.0
        return cropped, moved


@dataclass
class HorizontalFlip(_Base):
    """Mirror left-right with probability ``p``."""

    p: float = 0.5
    salt: int = 23

    def __call__(
        self, image: Image.Image, keypoints: Keypoints, *, image_id: str
    ) -> tuple[Image.Image, Keypoints]:
        """Mirror image and landmarks left-right, or return both unchanged."""
        if not _draw(self.p, self._rng(image_id)):
            return image, keypoints
        flipped = keypoints.copy()
        visible = flipped[:, 2] > 0
        flipped[visible, 0] = 1.0 - flipped[visible, 0]
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), flipped


@dataclass
class VerticalFlip(_Base):
    """Mirror top-bottom with probability ``p``."""

    p: float = 0.5
    salt: int = 31

    def __call__(
        self, image: Image.Image, keypoints: Keypoints, *, image_id: str
    ) -> tuple[Image.Image, Keypoints]:
        """Mirror image and landmarks top-bottom, or return both unchanged."""
        if not _draw(self.p, self._rng(image_id)):
            return image, keypoints
        flipped = keypoints.copy()
        visible = flipped[:, 2] > 0
        flipped[visible, 1] = 1.0 - flipped[visible, 1]
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), flipped


@dataclass
class ColorJitter(_Base):
    """Brightness and contrast jitter. Photometric only, so landmarks are untouched."""

    brightness: float = 0.2
    contrast: float = 0.2
    salt: int = 47

    def __call__(
        self, image: Image.Image, keypoints: Keypoints, *, image_id: str
    ) -> tuple[Image.Image, Keypoints]:
        """Jitter brightness and contrast; landmarks pass through unchanged."""
        rng = self._rng(image_id)
        adjusted = ImageEnhance.Brightness(image).enhance(
            1.0 + rng.uniform(-self.brightness, self.brightness)
        )
        adjusted = ImageEnhance.Contrast(adjusted).enhance(1.0 + rng.uniform(-self.contrast, self.contrast))
        return adjusted, keypoints


def _draw(probability: float, rng: random.Random) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return rng.random() < probability


@dataclass
class AugmentationPipeline:
    """Ordered composition of :class:`KeypointTransform` objects.

    Raises:
        ValueError: at construction, if two transforms share a salt. Identical salts
            make transforms draw identical parameter sequences for the same image,
            correlating a crop with a flip in a way that is invisible in the loss
            curve and impossible to debug after the fact.
    """

    transforms: Sequence[KeypointTransform]

    def __post_init__(self) -> None:
        """Reject duplicate salts at construction."""
        salts = [t.salt for t in self.transforms]
        if len(set(salts)) != len(salts):
            raise ValueError(f"transforms must use distinct salts, got {salts}")

    def __call__(
        self, image: Image.Image, keypoints: Keypoints, *, image_id: str
    ) -> tuple[Image.Image, Keypoints]:
        """Apply every transform in order."""
        for transform in self.transforms:
            image, keypoints = transform(image, keypoints, image_id=image_id)
        return image, keypoints

    def __len__(self) -> int:
        """Number of transforms in the pipeline."""
        return len(self.transforms)


def build_pipeline(
    *,
    output_size: tuple[int, int] = (224, 224),
    crop_jitter: bool = True,
    color_jitter: bool = True,
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
) -> AugmentationPipeline:
    """Assemble the augmentation policy labelled ``Aug`` in the published results.

    The published ``Aug`` runs enable all four transforms; ``Control`` runs use an
    empty pipeline.  ``Expand`` additionally draws on an offline-expanded frame set,
    which is a dataset property rather than a transform and so lives outside this
    module.
    """
    transforms: list[KeypointTransform] = []
    if crop_jitter:
        transforms.append(CropJitter(output_size=output_size))
    if horizontal_flip:
        transforms.append(HorizontalFlip())
    if vertical_flip:
        transforms.append(VerticalFlip())
    if color_jitter:
        transforms.append(ColorJitter())
    return AugmentationPipeline(transforms)
