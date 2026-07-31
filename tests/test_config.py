"""Tests for experiment configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vitvert.config import AugmentationConfig, ConfigError, ExperimentConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
MINIMAL = {"model_name": "vitpose-s"}


class TestValidation:
    def test_minimal_configuration_is_valid(self) -> None:
        config = ExperimentConfig.from_mapping(MINIMAL)
        assert config.loss_type == "l1"
        assert config.config_label == "Fine-tuned / Control"

    def test_unknown_model_is_rejected_at_load_time(self) -> None:
        with pytest.raises(ConfigError, match="unknown model_name"):
            ExperimentConfig.from_mapping({"model_name": "resnet50"})

    def test_unknown_loss_is_rejected_at_load_time(self) -> None:
        with pytest.raises(ConfigError, match="unknown loss_type"):
            ExperimentConfig.from_mapping({**MINIMAL, "loss_type": "l2"})

    def test_loss_aliases_are_accepted(self) -> None:
        assert ExperimentConfig.from_mapping({**MINIMAL, "loss_type": "mae"}).loss_type == "mae"

    def test_unknown_scheduler_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="scheduler_type"):
            ExperimentConfig.from_mapping({**MINIMAL, "scheduler_type": "onecycle"})

    @pytest.mark.parametrize("field", ["num_epochs", "batch_size", "k_folds", "num_keypoints"])
    def test_non_positive_counts_are_rejected(self, field: str) -> None:
        with pytest.raises(ConfigError, match=field):
            ExperimentConfig.from_mapping({**MINIMAL, field: 0})

    @pytest.mark.parametrize("rate", [0.0, 1.0, -1e-4, 12.0])
    def test_learning_rate_must_be_a_fraction(self, rate: float) -> None:
        with pytest.raises(ConfigError, match="learning_rate"):
            ExperimentConfig.from_mapping({**MINIMAL, "learning_rate": rate})

    def test_typo_in_a_key_is_rejected_rather_than_ignored(self) -> None:
        """A silently ignored ``freeze_backone`` would run the wrong experiment."""
        with pytest.raises(ConfigError, match="unknown configuration keys"):
            ExperimentConfig.from_mapping({**MINIMAL, "freeze_backone": True})

    def test_typo_in_an_augmentation_key_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="unknown augmentation keys"):
            ExperimentConfig.from_mapping({**MINIMAL, "augmentation_params": {"enable_augmentaton": True}})

    def test_non_mapping_augmentation_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="must be a mapping"):
            ExperimentConfig.from_mapping({**MINIMAL, "augmentation_params": ["yes"]})


class TestLabels:
    @pytest.mark.parametrize(
        ("frozen", "augmented", "expected"),
        [
            (False, False, "Fine-tuned / Control"),
            (False, True, "Fine-tuned / Aug"),
            (True, False, "Frozen / Control"),
            (True, True, "Frozen / Aug"),
        ],
    )
    def test_labels_match_the_published_grid(self, frozen: bool, augmented: bool, expected: str) -> None:
        config = ExperimentConfig.from_mapping(
            {**MINIMAL, "freeze_backbone": frozen, "augmentation_params": {"enable_augmentation": augmented}}
        )
        assert config.config_label == expected


class TestRoundTrip:
    def test_yaml_round_trip_preserves_every_field(self, tmp_path: Path) -> None:
        original = ExperimentConfig.from_mapping(
            {
                "model_name": "vitpose++-b",
                "loss_type": "huber",
                "num_epochs": 40,
                "freeze_backbone": True,
                "augmentation_params": {"enable_augmentation": True, "vertical_flip_enabled": False},
            }
        )
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(original.as_dict()), encoding="utf-8")
        assert load_config(path) == original

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "absent.yaml")

    def test_non_mapping_document_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="expected a YAML mapping"):
            load_config(path)


class TestShippedExperiments:
    """Every committed experiment file must load. A broken example is worse than none."""

    @pytest.mark.parametrize(
        "path",
        sorted((REPO_ROOT / "experiments").glob("*.yaml")),
        ids=lambda p: p.name,
    )
    def test_shipped_experiment_loads(self, path: Path) -> None:
        assert isinstance(load_config(path), ExperimentConfig)

    def test_at_least_one_experiment_ships(self) -> None:
        assert list((REPO_ROOT / "experiments").glob("*.yaml"))


class TestDefaults:
    def test_augmentation_defaults_are_off_but_individually_enabled(self) -> None:
        defaults = AugmentationConfig()
        assert defaults.enable_augmentation is False
        assert defaults.crop_jitter_enabled is True
