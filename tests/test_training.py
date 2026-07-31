"""Tests for the training loop, on a synthetic task that needs no dataset.

A two-parameter model learning a fixed landmark position is enough to exercise
everything that matters: that the loop optimises, that best-checkpoint selection picks
the right epoch, that a frozen model is rejected rather than silently no-op trained,
and that empty loaders surface as ``nan`` rather than as a flattering zero.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from vitvert.losses import l1_loss  # noqa: E402
from vitvert.training import evaluate, train_fold, train_one_epoch  # noqa: E402

TARGET = (0.62, 0.38)


class ConstantPredictor(nn.Module):
    """Predicts one learnable landmark, ignoring the input entirely."""

    def __init__(self, start: float = 0.1) -> None:
        super().__init__()
        self.backbone = nn.Linear(1, 1)  # present so freeze_backbone has something to freeze
        self.raw = nn.Parameter(torch.full((1, 2), start))

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = pixel_values.shape[0]
        return {"keypoints": torch.sigmoid(self.raw).expand(batch, 2).view(batch, 1, 2)}


class DictLoader:
    """Wraps tensors into the ``{"pixel_values", "keypoints"}`` batch contract."""

    def __init__(self, n: int = 8, batch_size: int = 4, target: tuple[float, float] = TARGET) -> None:
        pixels = torch.randn(n, 1)
        keypoints = torch.tensor([[[*target, 2.0]]]).repeat(n, 1, 1)
        self._loader = DataLoader(TensorDataset(pixels, keypoints), batch_size=batch_size)

    def __iter__(self):  # noqa: D105
        for pixels, keypoints in self._loader:
            yield {"pixel_values": pixels, "keypoints": keypoints}

    def __len__(self) -> int:  # noqa: D105
        return len(self._loader)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


class TestEpoch:
    def test_training_reduces_the_loss(self, device: torch.device) -> None:
        torch.manual_seed(0)
        model = ConstantPredictor()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.2)
        losses = [
            train_one_epoch(
                model, DictLoader(), loss_fn=l1_loss, optimizer=optimizer, device=device, image_size=224.0
            )[0]
            for _ in range(15)
        ]
        assert losses[-1] < losses[0] / 2, "the loop failed to optimise a trivially learnable target"

    def test_empty_loader_yields_nan_not_zero(self, device: torch.device) -> None:
        model = ConstantPredictor()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        loss, error = train_one_epoch(
            model, [], loss_fn=l1_loss, optimizer=optimizer, device=device, image_size=224.0
        )
        assert loss != loss and error != error  # both nan

    def test_gradient_clipping_can_be_disabled(self, device: torch.device) -> None:
        model = ConstantPredictor()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        loss, _ = train_one_epoch(
            model,
            DictLoader(),
            loss_fn=l1_loss,
            optimizer=optimizer,
            device=device,
            image_size=224.0,
            grad_clip=None,
        )
        assert loss == loss


class TestEvaluate:
    def test_reports_the_per_image_summary_with_intervals(self, device: torch.device) -> None:
        model = ConstantPredictor(start=0.0)  # sigmoid(0) = 0.5
        loss, summary = evaluate(model, DictLoader(n=12), loss_fn=l1_loss, device=device, image_size=224.0)
        assert summary.n == 12
        assert loss > 0
        # Every image is identical, so the spread -- and the interval -- must be zero.
        assert summary.mean_ci_half_width == pytest.approx(0.0, abs=1e-9)
        expected = ((0.5 - TARGET[0]) ** 2 + (0.5 - TARGET[1]) ** 2) ** 0.5 * 224.0
        assert summary.mean == pytest.approx(expected, rel=1e-4)

    def test_evaluation_does_not_leave_the_model_in_train_mode(self, device: torch.device) -> None:
        model = ConstantPredictor()
        model.train()
        evaluate(model, DictLoader(), loss_fn=l1_loss, device=device, image_size=224.0)
        assert not model.training


class TestTrainFold:
    def test_keeps_the_state_dict_from_the_best_epoch(self, device: torch.device) -> None:
        torch.manual_seed(0)
        model = ConstantPredictor()
        result = train_fold(
            model,
            DictLoader(),
            DictLoader(n=4),
            loss_fn=l1_loss,
            num_epochs=6,
            learning_rate=0.2,
            device=device,
            scheduler_type="none",
        )
        assert len(result.epochs) == 6
        assert result.best_epoch is not None
        assert result.best_valid_loss == pytest.approx(min(result.valid_losses))
        best = next(e for e in result.epochs if e.epoch == result.best_epoch)
        assert best.valid_loss == result.best_valid_loss

    def test_best_state_dict_is_detached_from_the_live_model(self, device: torch.device) -> None:
        torch.manual_seed(0)
        model = ConstantPredictor()
        result = train_fold(
            model,
            DictLoader(),
            DictLoader(n=4),
            loss_fn=l1_loss,
            num_epochs=3,
            learning_rate=0.3,
            device=device,
            scheduler_type="none",
        )
        snapshot = result.best_state_dict["raw"].clone()
        with torch.no_grad():
            model.raw.add_(5.0)
        torch.testing.assert_close(result.best_state_dict["raw"], snapshot)

    def test_epoch_callback_fires_once_per_epoch(self, device: torch.device) -> None:
        seen: list[int] = []
        train_fold(
            ConstantPredictor(),
            DictLoader(),
            DictLoader(n=4),
            loss_fn=l1_loss,
            num_epochs=4,
            learning_rate=0.1,
            device=device,
            scheduler_type="none",
            on_epoch_end=lambda e: seen.append(e.epoch),
        )
        assert seen == [1, 2, 3, 4]

    @pytest.mark.parametrize("scheduler", ["plateau", "cosine", "step", "none"])
    def test_every_documented_scheduler_runs(self, scheduler: str, device: torch.device) -> None:
        result = train_fold(
            ConstantPredictor(),
            DictLoader(),
            DictLoader(n=4),
            loss_fn=l1_loss,
            num_epochs=2,
            learning_rate=0.1,
            device=device,
            scheduler_type=scheduler,
        )
        assert len(result.epochs) == 2

    def test_unknown_scheduler_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="scheduler_type"):
            train_fold(
                ConstantPredictor(),
                DictLoader(),
                DictLoader(n=4),
                loss_fn=l1_loss,
                num_epochs=1,
                learning_rate=0.1,
                device=device,
                scheduler_type="onecycle",
            )

    def test_non_positive_epochs_raise(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="num_epochs"):
            train_fold(
                ConstantPredictor(),
                DictLoader(),
                DictLoader(n=4),
                loss_fn=l1_loss,
                num_epochs=0,
                learning_rate=0.1,
                device=device,
            )

    def test_fully_frozen_model_is_rejected(self, device: torch.device) -> None:
        """Training a model with nothing trainable burns GPU hours for nothing."""
        model = ConstantPredictor()
        for parameter in model.parameters():
            parameter.requires_grad = False
        with pytest.raises(ValueError, match="no trainable parameters"):
            train_fold(
                model,
                DictLoader(),
                DictLoader(n=4),
                loss_fn=l1_loss,
                num_epochs=1,
                learning_rate=0.1,
                device=device,
            )

    def test_frozen_parameters_are_excluded_from_the_optimiser(self, device: torch.device) -> None:
        """A frozen tensor must not move, not even via weight decay or momentum."""
        torch.manual_seed(0)
        model = ConstantPredictor()
        model.backbone.weight.requires_grad = False
        frozen_before = model.backbone.weight.detach().clone()
        train_fold(
            model,
            DictLoader(),
            DictLoader(n=4),
            loss_fn=l1_loss,
            num_epochs=3,
            learning_rate=0.5,
            device=device,
            scheduler_type="none",
        )
        torch.testing.assert_close(model.backbone.weight.detach(), frozen_before)
