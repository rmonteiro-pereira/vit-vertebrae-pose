"""Tests for deterministic, keypoint-aware augmentation.

The property that matters is that image and landmarks move together.  A transform
that shifts pixels without shifting the label corrupts the training set while every
loss curve still looks healthy, which is exactly the failure the original two-call-site
design could produce.  These tests find the landmark in the pixels and check it went
where the label says it went.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vitvert.data.augment import (
    AugmentationPipeline,
    ColorJitter,
    CropJitter,
    HorizontalFlip,
    VerticalFlip,
    build_pipeline,
    derive_seed,
)

SIZE = 64


def marked_image(x: int, y: int) -> Image.Image:
    """A black image with one white 3x3 marker at ``(x, y)``."""
    pixels = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    pixels[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = 255
    return Image.fromarray(pixels)


def marker_centre(image: Image.Image) -> tuple[float, float]:
    """Brightness centroid, in normalised coordinates."""
    pixels = np.asarray(image.convert("L"), dtype=float)
    total = pixels.sum()
    assert total > 0, "marker disappeared from the image"
    ys, xs = np.mgrid[0 : pixels.shape[0], 0 : pixels.shape[1]]
    return float((pixels * xs).sum() / total / (pixels.shape[1] - 1)), float(
        (pixels * ys).sum() / total / (pixels.shape[0] - 1)
    )


def keypoints_at(x_norm: float, y_norm: float) -> np.ndarray:
    return np.array([[x_norm, y_norm, 2.0]], dtype=np.float32)


class TestDeterminism:
    def test_same_identifier_gives_the_same_seed(self) -> None:
        assert derive_seed("frame.jpg", 11) == derive_seed("frame.jpg", 11)

    def test_different_salts_decorrelate_transforms(self) -> None:
        assert derive_seed("frame.jpg", 11) != derive_seed("frame.jpg", 23)

    def test_different_images_get_different_seeds(self) -> None:
        assert derive_seed("a.jpg", 11) != derive_seed("b.jpg", 11)

    def test_repeated_application_is_identical(self) -> None:
        pipeline = build_pipeline(output_size=(SIZE, SIZE))
        image, keypoints = marked_image(20, 30), keypoints_at(20 / SIZE, 30 / SIZE)
        first_image, first_kp = pipeline(image, keypoints, image_id="frame.jpg")
        second_image, second_kp = pipeline(image, keypoints, image_id="frame.jpg")
        np.testing.assert_array_equal(np.asarray(first_image), np.asarray(second_image))
        np.testing.assert_array_equal(first_kp, second_kp)

    def test_augmentation_does_not_mutate_its_input(self) -> None:
        pipeline = build_pipeline(output_size=(SIZE, SIZE))
        keypoints = keypoints_at(0.3, 0.4)
        original = keypoints.copy()
        pipeline(marked_image(20, 26), keypoints, image_id="frame.jpg")
        np.testing.assert_array_equal(keypoints, original)


class TestLabelConsistency:
    def test_horizontal_flip_moves_pixels_and_label_together(self) -> None:
        flip = HorizontalFlip(p=1.0)
        image, keypoints = marked_image(10, 32), keypoints_at(10 / (SIZE - 1), 32 / (SIZE - 1))
        flipped_image, flipped_kp = flip(image, keypoints, image_id="frame.jpg")
        pixel_x, pixel_y = marker_centre(flipped_image)
        assert pixel_x == pytest.approx(float(flipped_kp[0, 0]), abs=0.02)
        assert pixel_y == pytest.approx(float(flipped_kp[0, 1]), abs=0.02)

    def test_vertical_flip_moves_pixels_and_label_together(self) -> None:
        flip = VerticalFlip(p=1.0)
        image, keypoints = marked_image(32, 12), keypoints_at(32 / (SIZE - 1), 12 / (SIZE - 1))
        flipped_image, flipped_kp = flip(image, keypoints, image_id="frame.jpg")
        pixel_x, pixel_y = marker_centre(flipped_image)
        assert pixel_x == pytest.approx(float(flipped_kp[0, 0]), abs=0.02)
        assert pixel_y == pytest.approx(float(flipped_kp[0, 1]), abs=0.02)

    def test_crop_jitter_moves_pixels_and_label_together(self) -> None:
        crop = CropJitter(output_size=(SIZE, SIZE), scale_range=(0.7, 0.7), shift_range=(0.0, 0.0))
        image, keypoints = marked_image(34, 28), keypoints_at(34 / (SIZE - 1), 28 / (SIZE - 1))
        cropped_image, cropped_kp = crop(image, keypoints, image_id="frame.jpg")
        assert cropped_kp[0, 2] > 0, "landmark should remain inside a centred 70% crop"
        pixel_x, pixel_y = marker_centre(cropped_image)
        assert pixel_x == pytest.approx(float(cropped_kp[0, 0]), abs=0.04)
        assert pixel_y == pytest.approx(float(cropped_kp[0, 1]), abs=0.04)

    def test_flip_at_probability_zero_is_the_identity(self) -> None:
        flip = HorizontalFlip(p=0.0)
        keypoints = keypoints_at(0.2, 0.8)
        _, moved = flip(marked_image(13, 51), keypoints, image_id="frame.jpg")
        np.testing.assert_array_equal(moved, keypoints)

    def test_double_flip_returns_the_original_label(self) -> None:
        flip = HorizontalFlip(p=1.0)
        start = keypoints_at(0.2, 0.8)
        _, once = flip(marked_image(13, 51), start, image_id="frame.jpg")
        _, twice = flip(marked_image(13, 51), once, image_id="frame.jpg")
        np.testing.assert_allclose(twice, start, atol=1e-7)


class TestVisibility:
    def test_landmark_cropped_out_is_marked_invisible_not_clamped(self) -> None:
        crop = CropJitter(output_size=(SIZE, SIZE), scale_range=(0.4, 0.4), shift_range=(0.0, 0.0))
        _, moved = crop(marked_image(2, 2), keypoints_at(0.02, 0.02), image_id="frame.jpg")
        assert moved[0, 2] == 0.0
        assert moved[0, 0] == 0.0 and moved[0, 1] == 0.0

    def test_invisible_landmarks_are_left_alone_by_flips(self) -> None:
        invisible = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        _, moved = HorizontalFlip(p=1.0)(marked_image(10, 10), invisible, image_id="frame.jpg")
        np.testing.assert_array_equal(moved, invisible)


class TestPipeline:
    def test_duplicate_salts_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="distinct salts"):
            AugmentationPipeline([HorizontalFlip(salt=5), VerticalFlip(salt=5)])

    def test_builder_honours_the_toggles(self) -> None:
        assert len(build_pipeline(crop_jitter=False, color_jitter=False)) == 2
        assert len(build_pipeline()) == 4
        assert (
            len(
                build_pipeline(
                    crop_jitter=False, color_jitter=False, horizontal_flip=False, vertical_flip=False
                )
            )
            == 0
        )

    def test_colour_jitter_changes_pixels_but_not_labels(self) -> None:
        jitter = ColorJitter()
        keypoints = keypoints_at(0.5, 0.5)
        image = Image.fromarray(np.full((SIZE, SIZE, 3), 128, dtype=np.uint8))
        adjusted, moved = jitter(image, keypoints, image_id="frame.jpg")
        np.testing.assert_array_equal(moved, keypoints)
        assert not np.array_equal(np.asarray(adjusted), np.asarray(image))
