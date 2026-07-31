"""Parsing of the YOLO-style keypoint annotation format used by the dataset.

One ``.txt`` file per image, one line per instance::

    <class_id> <cx> <cy> <w> <h> <x1> <y1> <v1> <x2> <y2> <v2> ...

All coordinates are normalised to ``[0, 1]``.  ``v`` is visibility: ``0`` absent,
``1`` occluded-but-labelled, ``2`` clearly visible.  This project uses two landmarks
per image (inferior endplate of C2 and of C4).

The parser is pure text handling with no image access, which is why it is the one
part of the data path that is fully covered by tests in a repository that ships no
images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Annotation", "AnnotationError", "label_path_for", "parse_annotation", "read_annotation"]

_BBOX_FIELDS = 4
_MIN_FIELDS = 1 + _BBOX_FIELDS + 3  # class + bbox + one keypoint triplet


class AnnotationError(ValueError):
    """Raised when an annotation file cannot be interpreted."""


@dataclass(frozen=True)
class Annotation:
    """One parsed annotation line."""

    class_id: int
    bbox: tuple[float, float, float, float]
    keypoints: np.ndarray  # (K, 3) float32: x, y, visibility

    @property
    def num_keypoints(self) -> int:
        """Number of landmark triplets parsed from the line."""
        return int(self.keypoints.shape[0])

    def padded(self, max_keypoints: int) -> np.ndarray:
        """Landmarks padded or truncated to ``max_keypoints`` rows.

        Padding rows are ``(0, 0, 0)``: coordinate zero *and* visibility zero, so a
        padded slot is skipped by every loss instead of pulling predictions toward
        the top-left corner.

        Raises:
            ValueError: if ``max_keypoints`` is not positive.
        """
        if max_keypoints <= 0:
            raise ValueError(f"max_keypoints must be positive, got {max_keypoints!r}")
        padded = np.zeros((max_keypoints, 3), dtype=np.float32)
        take = min(max_keypoints, self.num_keypoints)
        padded[:take] = self.keypoints[:take]
        return padded


def parse_annotation(text: str) -> Annotation | None:
    """Parse the first non-empty line of an annotation file.

    Args:
        text: raw file contents.

    Returns:
        The parsed annotation, or ``None`` when the file is empty -- an image with no
        annotated landmark, which the dataset represents as fully invisible.

    Raises:
        AnnotationError: if a non-empty line is malformed: too few fields, a
            non-numeric field, a trailing partial keypoint triplet, or a coordinate
            outside ``[0, 1]``. The original parser returned ``None`` for short lines
            and silently discarded trailing partial triplets, so a truncated label
            file trained as an unannotated image without any signal that it happened.
    """
    line = next((raw.strip() for raw in text.splitlines() if raw.strip()), "")
    if not line:
        return None

    fields = line.split()
    if len(fields) < _MIN_FIELDS:
        raise AnnotationError(
            f"expected at least {_MIN_FIELDS} fields (class, bbox, one keypoint), got {len(fields)}: {line!r}"
        )

    try:
        class_id = int(float(fields[0]))
        bbox = tuple(float(v) for v in fields[1 : 1 + _BBOX_FIELDS])
        rest = [float(v) for v in fields[1 + _BBOX_FIELDS :]]
    except ValueError as exc:
        raise AnnotationError(f"non-numeric field in annotation: {line!r}") from exc

    if len(rest) % 3 != 0:
        raise AnnotationError(
            "keypoint fields must come in (x, y, visibility) triplets, "
            f"got {len(rest)} trailing values: {line!r}"
        )

    keypoints = np.asarray(rest, dtype=np.float32).reshape(-1, 3)
    # Only visible landmarks are range-checked: the padding convention for an absent
    # landmark is (0, 0, 0), and an occluded one may legitimately be recorded anywhere.
    visible_coordinates = keypoints[keypoints[:, 2] > 0][:, :2]
    out_of_range = ((visible_coordinates < 0.0) | (visible_coordinates > 1.0)).any()
    if out_of_range:
        raise AnnotationError(f"visible keypoint coordinates must be normalised to [0, 1]: {line!r}")

    return Annotation(class_id=class_id, bbox=bbox, keypoints=keypoints)  # type: ignore[arg-type]


def label_path_for(image_path: Path) -> Path:
    """Map ``.../images/foo.jpg`` to ``.../labels/foo.txt``.

    Raises:
        AnnotationError: if the path has no ``images`` component, which almost always
            means the dataset root points at the wrong directory.
    """
    parts = list(image_path.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError:
        raise AnnotationError(
            f"expected an 'images' directory in {image_path}; see docs/dataset.md for the required layout"
        ) from None
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_annotation(path: Path) -> Annotation | None:
    """Read and parse an annotation file.

    Raises:
        AnnotationError: if the file is missing or malformed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnnotationError(f"cannot read annotation {path}: {exc}") from exc
    return parse_annotation(text)
