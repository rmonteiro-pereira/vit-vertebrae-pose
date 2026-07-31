"""Model names as plain data.

Kept free of ``torch`` so that configuration loading, results aggregation and the
figure scripts can validate a model name on a machine that has no deep-learning stack
installed.
"""

from __future__ import annotations

from typing import Final

__all__ = ["LARGE_VARIANTS", "MODEL_NAMES", "available_models"]

MODEL_NAMES: Final[tuple[str, ...]] = (
    "vit-base",
    "vitpose-s",
    "vitpose-b",
    "vitpose-l",
    "vitpose++-s",
    "vitpose++-b",
    "vitpose++-l",
)

#: Variants ordered last in sweeps because they are the ones that exhaust GPU memory.
LARGE_VARIANTS: Final[tuple[str, ...]] = ("vitpose-l", "vitpose++-l")


def available_models() -> list[str]:
    """Registered model names, with the 430M-parameter variants last.

    The ordering is not cosmetic: an experiment sweep iterates this list, so putting
    the largest models at the end means a sweep produces most of its results before
    it reaches the runs most likely to fail on memory.
    """
    return sorted(n for n in MODEL_NAMES if n not in LARGE_VARIANTS) + list(LARGE_VARIANTS)
