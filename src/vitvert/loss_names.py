"""Loss names and aliases as plain data, importable without ``torch``.

:mod:`vitvert.losses` builds its registry from these names, and
``tests/test_losses.py`` asserts the two never drift apart.  The split exists so that
configuration validation works on a machine with no deep-learning stack.
"""

from __future__ import annotations

from typing import Final

__all__ = ["LOSS_ALIASES", "LOSS_NAMES", "available_losses", "canonical_loss_name"]

LOSS_NAMES: Final[tuple[str, ...]] = (
    "adaptive_wing",
    "cosine",
    "euclidean",
    "exponential",
    "focal",
    "huber",
    "l1",
    "mse",
    "multithreshold",
    "wing",
)

LOSS_ALIASES: Final[dict[str, str]] = {
    "mae": "l1",
    "smooth_l1": "huber",
    "multi": "multithreshold",
    "exp": "exponential",
    "awing": "adaptive_wing",
}


def available_losses() -> list[str]:
    """Canonical loss names, sorted."""
    return sorted(LOSS_NAMES)


def canonical_loss_name(name: str) -> str:
    """Resolve an alias to its canonical name.

    Raises:
        KeyError: if the name is neither canonical nor a known alias.  The original
            code printed a warning and silently substituted the Euclidean loss, so a
            typo in a config produced a full training run under the wrong objective
            with no trace in the archived metadata.
    """
    key = name.strip().lower()
    key = LOSS_ALIASES.get(key, key)
    if key not in LOSS_NAMES:
        raise KeyError(f"unknown loss {name!r}; available: {', '.join(available_losses())}")
    return key
