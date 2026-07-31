"""Dataset access and augmentation.

``vitvert.data.annotations`` and ``vitvert.data.augment`` have no torch dependency and
are importable anywhere. ``vitvert.data.dataset`` needs torch and is imported lazily
by :func:`__getattr__` so that the analysis pipeline stays torch-free.
"""

from __future__ import annotations

from typing import Any

from vitvert.data.annotations import Annotation, AnnotationError, parse_annotation
from vitvert.data.augment import AugmentationPipeline, build_pipeline

__all__ = [
    "Annotation",
    "AnnotationError",
    "AugmentationPipeline",
    "KeypointDataset",
    "build_pipeline",
    "parse_annotation",
]


def __getattr__(name: str) -> Any:
    if name == "KeypointDataset":
        from vitvert.data.dataset import KeypointDataset

        return KeypointDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
