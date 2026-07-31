"""Tests for the keypoint objectives.

Losses are the component where a silent error is most expensive: a wrong objective
still produces a smooth curve and a plausible number. These check closed-form values,
gradient flow, visibility handling, and the registry contract.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vitvert.loss_names import LOSS_NAMES, available_losses, canonical_loss_name  # noqa: E402
from vitvert.losses import (  # noqa: E402
    LOSS_REGISTRY,
    euclidean_loss,
    get_loss_function,
    huber_loss,
    l1_loss,
    mse_loss,
    multi_threshold_loss,
    wing_loss,
)


def pair(pred_xy, target_xy, visibility=2.0):
    pred = torch.tensor([[list(pred_xy)]], dtype=torch.float32)
    target = torch.tensor([[[*target_xy, visibility]]], dtype=torch.float32)
    return pred, target


class TestRegistry:
    def test_registry_matches_the_torch_free_name_list(self) -> None:
        assert sorted(LOSS_REGISTRY) == sorted(LOSS_NAMES) == available_losses()

    @pytest.mark.parametrize(
        "alias, canonical",
        [
            ("mae", "l1"),
            ("smooth_l1", "huber"),
            ("multi", "multithreshold"),
            ("exp", "exponential"),
            ("awing", "adaptive_wing"),
        ],
    )
    def test_aliases_resolve(self, alias: str, canonical: str) -> None:
        assert canonical_loss_name(alias) == canonical
        assert get_loss_function(alias) is LOSS_REGISTRY[canonical]

    def test_names_are_case_and_whitespace_insensitive(self) -> None:
        assert get_loss_function("  L1 ") is LOSS_REGISTRY["l1"]

    def test_unknown_name_raises_instead_of_substituting_a_default(self) -> None:
        """The original code warned and silently ran the Euclidean loss instead."""
        with pytest.raises(KeyError, match="unknown loss"):
            get_loss_function("l2")


class TestClosedForms:
    def test_euclidean_equals_the_hand_computed_distance(self) -> None:
        # (0.5, 0.5) vs (0.6, 0.5) at 224 px is 22.4 px.
        pred, target = pair((0.5, 0.5), (0.6, 0.5))
        assert float(euclidean_loss(pred, target)) == pytest.approx(22.4, abs=1e-3)

    def test_l1_is_the_mean_absolute_coordinate_error(self) -> None:
        pred, target = pair((0.5, 0.5), (0.6, 0.7))
        # |0.1| * 224 = 22.4 and |0.2| * 224 = 44.8; mean over both axes is 33.6.
        assert float(l1_loss(pred, target)) == pytest.approx(33.6, abs=1e-3)

    def test_mse_is_the_mean_squared_coordinate_error(self) -> None:
        pred, target = pair((0.5, 0.5), (0.6, 0.5))
        assert float(mse_loss(pred, target)) == pytest.approx(22.4**2 / 2, abs=1e-2)

    def test_huber_is_quadratic_inside_delta(self) -> None:
        pred, target = pair((0.5, 0.5), (0.5 + 2.0 / 224.0, 0.5))
        # dx = 2 px < delta = 5 px, so 0.5 * dx^2 / delta = 0.4, averaged over 2 axes.
        assert float(huber_loss(pred, target)) == pytest.approx(0.2, abs=1e-3)

    def test_zero_error_gives_zero_loss(self) -> None:
        pred, target = pair((0.42, 0.31), (0.42, 0.31))
        for name, fn in LOSS_REGISTRY.items():
            if name == "cosine":
                continue  # 1 - cos(v, v) = 0 too, but floating error dominates the assert below
            assert float(fn(pred, target)) == pytest.approx(0.0, abs=1e-3), name

    def test_multi_threshold_is_piecewise_linear(self) -> None:
        # 7 px sits in the (5, 10] band at weight 2.0: 2.0 * (7 - 5) = 4.0.
        pred, target = pair((0.5, 0.5), (0.5 + 7.0 / 224.0, 0.5))
        assert float(multi_threshold_loss(pred, target)) == pytest.approx(4.0, abs=1e-2)

    def test_wing_is_logarithmic_for_small_errors(self) -> None:
        pred, target = pair((0.5, 0.5), (0.5 + 4.0 / 224.0, 0.5))
        import math

        expected = 10.0 * math.log1p(4.0 / 2.0)  # x-axis only; y contributes 0
        assert float(wing_loss(pred, target)) == pytest.approx(expected, abs=1e-3)


class TestMonotonicity:
    @pytest.mark.parametrize("name", sorted(LOSS_REGISTRY))
    def test_larger_error_is_never_cheaper(self, name: str) -> None:
        fn = LOSS_REGISTRY[name]
        near, target = pair((0.50, 0.50), (0.52, 0.50))
        far_pred = torch.tensor([[[0.70, 0.50]]])
        assert float(fn(far_pred, target)) >= float(fn(near, target)) - 1e-6


class TestGradients:
    @pytest.mark.parametrize("name", sorted(LOSS_REGISTRY))
    def test_gradient_flows_to_the_prediction(self, name: str) -> None:
        pred = torch.tensor([[[0.4, 0.6], [0.7, 0.2]]], requires_grad=True)
        target = torch.tensor([[[0.5, 0.5, 2.0], [0.6, 0.3, 2.0]]])
        loss = LOSS_REGISTRY[name](pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all(), f"{name} produced non-finite gradients"
        assert float(pred.grad.abs().sum()) > 0.0, f"{name} produced a zero gradient"

    @pytest.mark.parametrize("name", sorted(LOSS_REGISTRY))
    def test_perfect_prediction_still_differentiates(self, name: str) -> None:
        """sqrt(0) is the classic NaN-gradient trap; the epsilon in _distances guards it."""
        pred = torch.tensor([[[0.5, 0.5]]], requires_grad=True)
        target = torch.tensor([[[0.5, 0.5, 2.0]]])
        LOSS_REGISTRY[name](pred, target).backward()
        assert torch.isfinite(pred.grad).all(), f"{name} produced NaN gradients at zero error"


class TestVisibility:
    @pytest.mark.parametrize("name", sorted(LOSS_REGISTRY))
    def test_invisible_landmarks_are_ignored(self, name: str) -> None:
        fn = LOSS_REGISTRY[name]
        pred = torch.tensor([[[0.5, 0.5], [0.9, 0.9]]])
        visible_only = torch.tensor([[[0.5, 0.5, 2.0], [0.1, 0.1, 0.0]]])
        just_the_visible = torch.tensor([[[0.5, 0.5, 2.0]]])
        assert float(fn(pred, visible_only)) == pytest.approx(
            float(fn(pred[:, :1], just_the_visible)), abs=1e-4
        ), f"{name} is influenced by an invisible landmark"

    @pytest.mark.parametrize("name", sorted(LOSS_REGISTRY))
    def test_all_invisible_gives_a_differentiable_zero(self, name: str) -> None:
        pred = torch.tensor([[[0.5, 0.5]]], requires_grad=True)
        target = torch.tensor([[[0.0, 0.0, 0.0]]])
        loss = LOSS_REGISTRY[name](pred, target)
        assert float(loss.detach()) == 0.0
        loss.backward()  # must not raise: the graph has to stay connected
        assert pred.grad is not None

    def test_targets_without_a_visibility_channel_are_all_visible(self) -> None:
        pred = torch.tensor([[[0.5, 0.5]]])
        target = torch.tensor([[[0.6, 0.5]]])
        assert float(euclidean_loss(pred, target)) == pytest.approx(22.4, abs=1e-3)


class TestValidation:
    def test_rank_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="rank-3"):
            l1_loss(torch.zeros(2, 2), torch.zeros(1, 2, 3))

    def test_batch_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            l1_loss(torch.zeros(2, 2, 2), torch.zeros(3, 2, 3))

    def test_unknown_reduction_raises(self) -> None:
        pred, target = pair((0.5, 0.5), (0.6, 0.5))
        with pytest.raises(ValueError, match="reduction"):
            euclidean_loss(pred, target, reduction="median")

    def test_mismatched_threshold_and_weight_lengths_raise(self) -> None:
        pred, target = pair((0.5, 0.5), (0.6, 0.5))
        with pytest.raises(ValueError, match="equal length"):
            multi_threshold_loss(pred, target, thresholds_px=(1.0, 2.0), weights=(1.0,))

    def test_unsorted_thresholds_raise(self) -> None:
        pred, target = pair((0.5, 0.5), (0.6, 0.5))
        with pytest.raises(ValueError, match="increasing"):
            multi_threshold_loss(pred, target, thresholds_px=(5.0, 2.0), weights=(1.0, 2.0))


class TestSumReduction:
    def test_sum_scales_with_the_batch(self) -> None:
        pred = torch.tensor([[[0.5, 0.5]], [[0.5, 0.5]]])
        target = torch.tensor([[[0.6, 0.5, 2.0]], [[0.6, 0.5, 2.0]]])
        mean = float(euclidean_loss(pred, target, reduction="mean"))
        total = float(euclidean_loss(pred, target, reduction="sum"))
        assert total == pytest.approx(2 * mean, rel=1e-6)
