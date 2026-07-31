"""Vision-Transformer benchmark for cervical-vertebra landmark localisation.

The package is deliberately split so that the *analysis* half (statistics, results
aggregation, tables) has no heavyweight dependencies and runs without the dataset,
while the *training* half (models, data, losses) needs ``torch`` and the licensed
images.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
