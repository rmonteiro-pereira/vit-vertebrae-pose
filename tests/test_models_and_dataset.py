"""Tests for the model registry, the prediction heads and the dataset wiring.

Backbone weights are not downloaded here: CI has no network budget for 1.6 GB of
checkpoints, and a test that needs the Hub is a test that fails for reasons unrelated
to this repository. What *is* tested is everything around them -- the registry
contract, the head arithmetic, and the images/labels wiring end to end against
synthetic noise images written to a temporary directory.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from vitvert.data.augment import build_pipeline  # noqa: E402
from vitvert.data.dataset import KeypointDataset  # noqa: E402
from vitvert.models import MODEL_NAMES, available_models, build_model  # noqa: E402
from vitvert.models.backbones import VITPOSE_SPECS  # noqa: E402
from vitvert.models.heads import ClassificationHead, KeypointHead  # noqa: E402


class TestRegistry:
    def test_seven_variants_are_registered(self) -> None:
        assert len(MODEL_NAMES) == 7
        assert set(available_models()) == set(MODEL_NAMES)

    def test_largest_variants_are_ordered_last(self) -> None:
        assert available_models()[-2:] == ["vitpose-l", "vitpose++-l"]

    def test_unverified_backbones_are_not_registered(self) -> None:
        """hrformer-b and transpose-b appear in the results but ship no builder."""
        assert "hrformer-b" not in MODEL_NAMES
        assert "transpose-b" not in MODEL_NAMES

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown model"):
            build_model("resnet50")

    def test_registry_import_does_not_pull_in_torch_hub(self) -> None:
        """``vitvert.models`` must stay importable without a network round-trip."""
        import importlib

        module = importlib.import_module("vitvert.models")
        assert not hasattr(module, "__annotations__") or "PlainViT" not in vars(module)

    def test_lazy_attribute_access_resolves_the_backbones(self) -> None:
        import vitvert.models as models

        assert models.ViTPose is not None
        assert models.KeypointHead is KeypointHead
        with pytest.raises(AttributeError):
            _ = models.NotAThing


class TestSpecs:
    def test_both_families_map_to_the_same_checkpoints(self) -> None:
        """The central caveat of the benchmark, asserted rather than only documented."""
        for size in ("s", "b", "l"):
            assert VITPOSE_SPECS[f"vitpose-{size}"] == VITPOSE_SPECS[f"vitpose++-{size}"]

    def test_hidden_sizes_follow_the_published_scales(self) -> None:
        assert [VITPOSE_SPECS[f"vitpose-{s}"].hidden_size for s in ("s", "b", "l")] == [384, 768, 1024]


class TestHeads:
    def test_keypoint_head_outputs_normalised_coordinates(self) -> None:
        head = KeypointHead(hidden_size=32, num_keypoints=2)
        output = head(torch.randn(4, 32))
        assert output.shape == (4, 2, 2)
        assert bool(((output >= 0.0) & (output <= 1.0)).all()), (
            "sigmoid must confine predictions to the frame"
        )

    def test_keypoint_head_is_differentiable(self) -> None:
        head = KeypointHead(hidden_size=16, num_keypoints=3)
        features = torch.randn(2, 16, requires_grad=True)
        head(features).sum().backward()
        assert features.grad is not None and torch.isfinite(features.grad).all()

    def test_classification_head_shape(self) -> None:
        assert ClassificationHead(hidden_size=16, num_classes=1)(torch.randn(5, 16)).shape == (5, 1)

    def test_dropout_is_inactive_in_eval_mode(self) -> None:
        head = KeypointHead(hidden_size=16, num_keypoints=1).eval()
        features = torch.randn(3, 16)
        torch.testing.assert_close(head(features), head(features))


class FakeProcessor:
    """Stands in for a HuggingFace image processor; no download, no config."""

    def __call__(self, images, return_tensors="pt"):
        array = np.asarray(images.resize((32, 32)), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return {"pixel_values": tensor}


@pytest.fixture
def split(tmp_path: Path) -> Path:
    """A three-image split of synthetic noise with matching labels."""
    root = tmp_path / "train"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(3):
        Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(
            root / "images" / f"synthetic_{index}.png"
        )
        (root / "labels" / f"synthetic_{index}.txt").write_text(
            f"0 0.5 0.5 0.3 0.4 0.4{index} 0.5 2 0.6 0.7 2\n", encoding="utf-8"
        )
    return root


class TestDataset:
    def test_yields_the_batch_contract(self, split: Path) -> None:
        dataset = KeypointDataset(split, processor=FakeProcessor())
        assert len(dataset) == 3
        sample = dataset[0]
        assert set(sample) == {"pixel_values", "keypoints", "labels"}
        assert sample["pixel_values"].shape == (3, 32, 32)
        assert sample["keypoints"].shape == (2, 3)
        assert sample["labels"].tolist() == [0]

    def test_ordering_is_stable(self, split: Path) -> None:
        first = KeypointDataset(split, processor=FakeProcessor())
        second = KeypointDataset(split, processor=FakeProcessor())
        assert [p.name for p in first.image_paths] == [p.name for p in second.image_paths]

    def test_augmentation_is_applied_deterministically(self, split: Path) -> None:
        dataset = KeypointDataset(
            split, processor=FakeProcessor(), augment=build_pipeline(output_size=(48, 48))
        )
        torch.testing.assert_close(dataset[0]["keypoints"], dataset[0]["keypoints"])

    def test_extra_keypoint_slots_are_padded_invisible(self, split: Path) -> None:
        dataset = KeypointDataset(split, processor=FakeProcessor(), max_keypoints=4)
        keypoints = dataset[0]["keypoints"]
        assert keypoints.shape == (4, 3)
        assert float(keypoints[2:].abs().sum()) == 0.0

    def test_missing_images_directory_points_at_the_docs(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=re.escape("docs/dataset.md")):
            KeypointDataset(tmp_path / "nowhere", processor=FakeProcessor())

    def test_empty_split_raises_rather_than_yielding_zero_batches(self, tmp_path: Path) -> None:
        (tmp_path / "train" / "images").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="no images"):
            KeypointDataset(tmp_path / "train", processor=FakeProcessor())

    def test_missing_label_file_raises(self, split: Path) -> None:
        from vitvert.data.annotations import AnnotationError

        (split / "labels" / "synthetic_0.txt").unlink()
        dataset = KeypointDataset(split, processor=FakeProcessor())
        with pytest.raises(AnnotationError, match="cannot read"):
            _ = dataset[0]
