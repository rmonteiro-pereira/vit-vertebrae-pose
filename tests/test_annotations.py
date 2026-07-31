"""Tests for annotation parsing -- the one data-path component that needs no images."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from vitvert.data.annotations import (
    Annotation,
    AnnotationError,
    label_path_for,
    parse_annotation,
    read_annotation,
)

VALID_LINE = "0 0.5 0.5 0.3 0.4 0.48 0.42 2 0.51 0.66 2"


class TestParsing:
    def test_parses_a_two_keypoint_line(self) -> None:
        annotation = parse_annotation(VALID_LINE)
        assert annotation is not None
        assert annotation.class_id == 0
        assert annotation.bbox == pytest.approx((0.5, 0.5, 0.3, 0.4))
        assert annotation.num_keypoints == 2
        np.testing.assert_allclose(annotation.keypoints[0], [0.48, 0.42, 2.0], rtol=1e-6)

    def test_empty_file_means_no_annotation(self) -> None:
        assert parse_annotation("") is None
        assert parse_annotation("\n  \n") is None

    def test_leading_blank_lines_are_skipped(self) -> None:
        assert parse_annotation(f"\n\n{VALID_LINE}\n") is not None

    def test_truncated_line_raises_instead_of_silently_dropping_the_label(self) -> None:
        """A truncated line must raise.

        The original parser returned ``None`` here, so a truncated label file trained
        as an unannotated image with no signal that anything was wrong.
        """
        with pytest.raises(AnnotationError, match="at least"):
            parse_annotation("0 0.5 0.5 0.3 0.4")

    def test_partial_trailing_triplet_raises(self) -> None:
        with pytest.raises(AnnotationError, match="triplets"):
            parse_annotation(VALID_LINE + " 0.7 0.8")

    def test_non_numeric_field_raises(self) -> None:
        with pytest.raises(AnnotationError, match="non-numeric"):
            parse_annotation("0 0.5 0.5 0.3 0.4 x 0.42 2")

    def test_out_of_range_visible_coordinate_raises(self) -> None:
        with pytest.raises(AnnotationError, match="normalised"):
            parse_annotation("0 0.5 0.5 0.3 0.4 1.4 0.42 2")

    def test_invisible_keypoints_may_sit_outside_the_frame(self) -> None:
        """Padding rows carry (0, 0, 0); an occluded landmark is not a range error."""
        annotation = parse_annotation("0 0.5 0.5 0.3 0.4 0.0 0.0 0 0.51 0.66 2")
        assert annotation is not None
        assert annotation.keypoints[0][2] == 0.0


class TestPadding:
    def test_pads_with_invisible_slots(self) -> None:
        annotation = parse_annotation(VALID_LINE)
        padded = annotation.padded(4)
        assert padded.shape == (4, 3)
        assert padded[2:].sum() == 0.0  # coordinates and visibility both zero

    def test_truncates_extra_keypoints(self) -> None:
        annotation = parse_annotation(VALID_LINE)
        assert annotation.padded(1).shape == (1, 3)

    def test_rejects_non_positive_size(self) -> None:
        annotation = parse_annotation(VALID_LINE)
        with pytest.raises(ValueError, match="max_keypoints"):
            annotation.padded(0)


class TestLabelPath:
    def test_maps_images_to_labels(self) -> None:
        result = label_path_for(Path("dataset") / "fold1" / "train" / "images" / "frame.jpg")
        assert result == Path("dataset") / "fold1" / "train" / "labels" / "frame.txt"

    def test_rewrites_only_the_last_images_component(self) -> None:
        result = label_path_for(Path("images") / "train" / "images" / "a.png")
        assert result == Path("images") / "train" / "labels" / "a.txt"

    def test_missing_images_component_names_the_fix(self) -> None:
        with pytest.raises(AnnotationError, match=re.escape("docs/dataset.md")):
            label_path_for(Path("dataset") / "train" / "pictures" / "frame.jpg")


class TestReading:
    def test_round_trips_through_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "a.txt"
        path.write_text(VALID_LINE, encoding="utf-8")
        annotation = read_annotation(path)
        assert isinstance(annotation, Annotation)
        assert annotation.num_keypoints == 2

    def test_missing_file_raises_annotation_error(self, tmp_path: Path) -> None:
        with pytest.raises(AnnotationError, match="cannot read"):
            read_annotation(tmp_path / "absent.txt")
