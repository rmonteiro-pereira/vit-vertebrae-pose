"""Declarative experiment configuration.

An experiment is a YAML file; a run is one point in the grid it describes.  The
configuration is validated at load time rather than at first use, because the failure
this guards against -- a typo silently selecting a different objective or an
unregistered model -- costs GPU-hours before it becomes visible in a loss curve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from vitvert.loss_names import available_losses, canonical_loss_name
from vitvert.models.names import available_models

__all__ = ["AugmentationConfig", "ConfigError", "ExperimentConfig", "load_config"]

_KNOWN_SCHEDULERS = frozenset({"plateau", "cosine", "step", "none"})


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass(frozen=True)
class AugmentationConfig:
    """Which augmentations the ``Aug`` policy enables."""

    enable_augmentation: bool = False
    crop_jitter_enabled: bool = True
    color_jitter_enabled: bool = True
    horizontal_flip_enabled: bool = True
    vertical_flip_enabled: bool = True

    @classmethod
    def from_mapping(cls, payload: Any) -> AugmentationConfig:
        """Build from a decoded YAML mapping, rejecting unknown keys.

        Raises:
            ConfigError: if the payload is not a mapping or carries unknown keys.
        """
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ConfigError(f"augmentation_params must be a mapping, got {type(payload).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(f"unknown augmentation keys: {sorted(unknown)}; expected {sorted(known)}")
        return cls(**payload)


@dataclass(frozen=True)
class ExperimentConfig:
    """One training configuration.

    Raises:
        ConfigError: from :meth:`validate` on any inconsistent field.
    """

    model_name: str
    loss_type: str = "l1"
    num_keypoints: int = 2
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 2e-4
    k_folds: int = 5
    scheduler_type: str = "plateau"
    freeze_backbone: bool = False
    seed: int = 42
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def __post_init__(self) -> None:
        """Validate eagerly, so a bad configuration fails before any GPU time is spent."""
        self.validate()

    def validate(self) -> None:
        """Check every field against the registries and numeric ranges."""
        if self.model_name not in available_models():
            raise ConfigError(
                f"unknown model_name {self.model_name!r}; available: {', '.join(available_models())}"
            )
        try:
            canonical_loss_name(self.loss_type)
        except KeyError:
            raise ConfigError(
                f"unknown loss_type {self.loss_type!r}; available: {', '.join(available_losses())}"
            ) from None
        if self.scheduler_type not in _KNOWN_SCHEDULERS:
            raise ConfigError(
                f"unknown scheduler_type {self.scheduler_type!r}; expected one of {sorted(_KNOWN_SCHEDULERS)}"
            )
        for name, value, minimum in (
            ("num_keypoints", self.num_keypoints, 1),
            ("num_epochs", self.num_epochs, 1),
            ("batch_size", self.batch_size, 1),
            ("k_folds", self.k_folds, 1),
        ):
            if value < minimum:
                raise ConfigError(f"{name} must be >= {minimum}, got {value}")
        if not 0.0 < self.learning_rate < 1.0:
            raise ConfigError(f"learning_rate must lie in (0, 1), got {self.learning_rate}")

    @property
    def config_label(self) -> str:
        """Grid label, matching the published ``Fine-tuned / Aug`` style."""
        backbone = "Frozen" if self.freeze_backbone else "Fine-tuned"
        policy = "Aug" if self.augmentation.enable_augmentation else "Control"
        return f"{backbone} / {policy}"

    def as_dict(self) -> dict[str, Any]:
        """Serialise to the mapping shape that :meth:`from_mapping` accepts."""
        payload = asdict(self)
        payload["augmentation_params"] = payload.pop("augmentation")
        return payload

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ExperimentConfig:
        """Build a configuration from a decoded YAML mapping.

        Raises:
            ConfigError: on unknown keys or a failed field check.
        """
        data = dict(payload)
        augmentation = AugmentationConfig.from_mapping(
            data.pop("augmentation_params", data.pop("augmentation", None))
        )
        known = {f.name for f in fields(cls)} - {"augmentation"}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown configuration keys: {sorted(unknown)}; expected {sorted(known)}")
        return cls(augmentation=augmentation, **data)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ConfigError: if the document is not a mapping or fails validation.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"{config_path}: expected a YAML mapping, got {type(payload).__name__}")
    return ExperimentConfig.from_mapping(payload)
