#!/usr/bin/env python3
"""Train one configuration on a dataset you supply.

Requires the licensed images (see ``docs/dataset.md``); nothing in this repository
will download them for you.  Everything else in the repository runs without them.

Example::

    uv run --extra train python scripts/train.py \
        --config experiments/vitpose_s_aug.yaml \
        --data-root /path/to/dataset/fold1 \
        --output runs/vitpose-s-aug

The output directory receives ``history.json`` (per-epoch metrics), ``summary.json``
(configuration plus final validation summary with intervals) and ``best.pt`` (the
state dict with the lowest validation loss).  Checkpoints are written atomically via
a temporary file and a rename, so an interrupted run leaves either the previous
checkpoint or the new one, never a truncated file.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vitvert.config import ExperimentConfig, load_config
from vitvert.data.augment import build_pipeline
from vitvert.data.dataset import KeypointDataset
from vitvert.losses import get_loss_function
from vitvert.models import build_model
from vitvert.training import train_fold

logger = logging.getLogger("vitvert.train")


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch. Full determinism also needs cuDNN settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


Batch = dict[str, torch.Tensor]


def build_loaders(
    config: ExperimentConfig,
    data_root: Path,
    processor: object,
    num_workers: int,
) -> tuple[DataLoader[Batch], DataLoader[Batch]]:
    """Build train/valid loaders from ``<data_root>/train`` and ``<data_root>/valid``.

    Raises:
        FileNotFoundError: if either split is missing.
    """
    augment = (
        build_pipeline(
            crop_jitter=config.augmentation.crop_jitter_enabled,
            color_jitter=config.augmentation.color_jitter_enabled,
            horizontal_flip=config.augmentation.horizontal_flip_enabled,
            vertical_flip=config.augmentation.vertical_flip_enabled,
        )
        if config.augmentation.enable_augmentation
        else None
    )

    train_set = KeypointDataset(
        data_root / "train",
        processor=processor,
        augment=augment,
        max_keypoints=config.num_keypoints,
    )
    valid_set = KeypointDataset(
        data_root / "valid",
        processor=processor,
        augment=None,  # validation is never augmented
        max_keypoints=config.num_keypoints,
    )

    def make(dataset: KeypointDataset, *, shuffle: bool) -> DataLoader[Batch]:
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            # Windows spawns a fresh interpreter per worker; without this the dataset
            # is reconstructed every epoch. See docs/engineering.md.
            persistent_workers=num_workers > 0,
        )

    return make(train_set, shuffle=True), make(valid_set, shuffle=False)


def atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    """Write via ``path.tmp`` then rename, so readers never see a partial file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    write(temporary)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, required=True, help="fold directory holding train/ and valid/"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None, help="override the configured epoch budget")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(args.config)
    epochs = args.epochs if args.epochs is not None else config.num_epochs
    set_seed(config.seed)

    if not args.data_root.is_dir():
        print(f"data root not found: {args.data_root}. See docs/dataset.md.", file=sys.stderr)
        return 2

    from transformers import AutoImageProcessor

    processor_id = (
        "google/vit-base-patch16-224"
        if config.model_name == "vit-base"
        else "usyd-community/vitpose-plus-base"
    )
    processor = AutoImageProcessor.from_pretrained(processor_id)  # type: ignore[no-untyped-call]

    train_loader, valid_loader = build_loaders(config, args.data_root, processor, args.num_workers)
    n_train, n_valid = len(train_loader.dataset), len(valid_loader.dataset)  # type: ignore[arg-type]
    logger.info(
        "model=%s loss=%s frozen=%s train=%d valid=%d",
        config.model_name,
        config.loss_type,
        config.freeze_backbone,
        n_train,
        n_valid,
    )

    model = build_model(config.model_name, num_keypoints=config.num_keypoints)
    if config.freeze_backbone:
        frozen = model.freeze_backbone()
        logger.info("froze %d backbone tensors", frozen)

    result = train_fold(
        model,
        train_loader,
        valid_loader,
        loss_fn=get_loss_function(config.loss_type),
        num_epochs=epochs,
        learning_rate=config.learning_rate,
        device=torch.device(args.device),
        scheduler_type=config.scheduler_type,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write(
        args.output / "history.json",
        lambda p: p.write_text(
            json.dumps([asdict(e) for e in result.epochs], indent=2) + "\n", encoding="utf-8"
        ),
    )
    atomic_write(
        args.output / "summary.json",
        lambda p: p.write_text(
            json.dumps(
                {
                    "config": config.as_dict(),
                    "best_epoch": result.best_epoch,
                    "best_valid_loss": result.best_valid_loss,
                    "validation_error": result.validation_error.as_dict()
                    if result.validation_error
                    else None,
                    "n_train": n_train,
                    "n_valid": n_valid,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        ),
    )
    if result.best_state_dict is not None:
        atomic_write(args.output / "best.pt", lambda p: torch.save(result.best_state_dict, p))

    logger.info("best epoch %s, valid loss %.4f", result.best_epoch, result.best_valid_loss or float("nan"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
