"""Tests for the reported metrics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vitvert.metrics import mean_pixel_error, pck, per_image_pixel_error  # noqa: E402


class TestPerImagePixelError:
    def test_matches_a_hand_computed_distance(self) -> None:
        pred = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
        target = torch.tensor([[[0.6, 0.5, 2.0], [0.5, 0.7, 2.0]]])
        # 22.4 px and 44.8 px; the per-image value is their mean.
        assert float(per_image_pixel_error(pred, target)[0]) == pytest.approx(33.6, abs=1e-3)

    def test_aggregates_per_image_not_per_landmark(self) -> None:
        """One image with two landmarks must not outvote an image with one."""
        pred = torch.tensor([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.0, 0.0]]])
        target = torch.tensor([[[0.6, 0.5, 2.0], [0.6, 0.5, 2.0]], [[0.9, 0.5, 2.0], [0.0, 0.0, 0.0]]])
        errors = per_image_pixel_error(pred, target)
        assert float(errors[0]) == pytest.approx(22.4, abs=1e-3)
        assert float(errors[1]) == pytest.approx(89.6, abs=1e-3)
        assert mean_pixel_error(pred, target) == pytest.approx(56.0, abs=1e-3)

    def test_invisible_landmarks_are_excluded(self) -> None:
        pred = torch.tensor([[[0.5, 0.5], [0.9, 0.9]]])
        target = torch.tensor([[[0.6, 0.5, 2.0], [0.1, 0.1, 0.0]]])
        assert float(per_image_pixel_error(pred, target)[0]) == pytest.approx(22.4, abs=1e-3)

    def test_image_with_no_visible_landmark_is_nan_not_zero(self) -> None:
        """A flattering zero here would silently improve the reported mean."""
        pred = torch.tensor([[[0.5, 0.5]]])
        target = torch.tensor([[[0.0, 0.0, 0.0]]])
        assert torch.isnan(per_image_pixel_error(pred, target)[0])

    def test_mean_ignores_nan_images(self) -> None:
        pred = torch.tensor([[[0.5, 0.5]], [[0.5, 0.5]]])
        target = torch.tensor([[[0.6, 0.5, 2.0]], [[0.0, 0.0, 0.0]]])
        assert mean_pixel_error(pred, target) == pytest.approx(22.4, abs=1e-3)

    def test_scales_linearly_with_image_size(self) -> None:
        pred = torch.tensor([[[0.5, 0.5]]])
        target = torch.tensor([[[0.6, 0.5, 2.0]]])
        small = float(per_image_pixel_error(pred, target, image_size=100.0)[0])
        large = float(per_image_pixel_error(pred, target, image_size=200.0)[0])
        assert large == pytest.approx(2 * small, rel=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            per_image_pixel_error(torch.zeros(2, 2, 2), torch.zeros(3, 2, 3))


class TestPCK:
    def test_counts_only_landmarks_within_the_threshold(self) -> None:
        pred = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
        target = torch.tensor([[[0.51, 0.5, 2.0], [0.9, 0.5, 2.0]]])  # 2.24 px and 89.6 px
        assert pck(pred, target, threshold_px=5.0) == pytest.approx(50.0)
        assert pck(pred, target, threshold_px=100.0) == pytest.approx(100.0)
        assert pck(pred, target, threshold_px=1.0) == pytest.approx(0.0)

    def test_invisible_landmarks_are_not_counted(self) -> None:
        pred = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
        target = torch.tensor([[[0.5, 0.5, 2.0], [0.9, 0.5, 0.0]]])
        assert pck(pred, target, threshold_px=5.0) == pytest.approx(100.0)

    def test_no_visible_landmark_is_nan(self) -> None:
        pred = torch.tensor([[[0.5, 0.5]]])
        target = torch.tensor([[[0.5, 0.5, 0.0]]])
        assert pck(pred, target, threshold_px=5.0) != pck(pred, target, threshold_px=5.0)  # nan

    def test_non_positive_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold_px"):
            pck(torch.zeros(1, 1, 2), torch.zeros(1, 1, 3), threshold_px=0.0)
