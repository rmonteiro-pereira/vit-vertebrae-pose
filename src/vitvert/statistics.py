"""Interval estimation for validation error summaries.

Every ``±`` printed by this project comes from :func:`error_summary`.  The intervals
describe the *sampling variability of a summary statistic on a fixed validation
split*.  They are **not** calibrated predictive uncertainty and must never be read as
such -- see ``docs/limitations.md``.

Method
------
mean
    Student-*t* interval.  ``t`` is obtained from the exact inverse CDF
    (:func:`student_t_ppf`), which converges to the normal quantile as ``n`` grows,
    so there is no discontinuity at any sample size.
median
    Percentile bootstrap with a fixed seed, so a rerun reproduces the interval
    bit-for-bit.

Fixed defect
------------
The original research code approximated the *t* quantile with
``min(2.0 + (30 - n) * 0.05, 2.576)`` for ``n < 30`` and switched to ``1.96`` at
``n >= 30``.  That expression is not an approximation of anything -- at ``n = 10`` it
yields 2.576 against a true ``t(9, 0.975) = 2.2622``, an interval 14% too wide, and it
is discontinuous at ``n = 30``, where *more* data produced a *wider* interval
(2.05 at ``n = 29``, 1.96 at ``n = 30``).  It is replaced here by the exact quantile,
which is monotone in ``n``.  Every interval published from
this project was computed on the primary validation split with ``n = 297``, i.e. on
the ``n >= 30`` branch, where the old code used ``1.96`` and the new code uses
``t(296, 0.975) = 1.9680``; the published half-widths are therefore reproduced to
within 0.4% and no reported conclusion changes.  ``tests/test_statistics.py`` pins
both facts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 1000

__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "ErrorSummary",
    "error_summary",
    "student_t_ppf",
]


def _regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)`` via the Lentz continued fraction (Numerical Recipes 6.4)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(log_front)

    # The continued fraction converges quickly only for x < (a + 1) / (a + b + 2);
    # otherwise use the symmetry I_x(a, b) = 1 - I_{1-x}(b, a).
    if x >= (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_incomplete_beta(b, a, 1.0 - x)

    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d

    for m in range(1, 300):
        m2 = 2 * m
        # even step
        num = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + num * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + num / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        # odd step
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + num * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + num / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break

    return front * h / a


def _student_t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * _regularised_incomplete_beta(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def student_t_ppf(p: float, df: float) -> float:
    """Inverse CDF of Student's *t* with ``df`` degrees of freedom.

    Implemented by bisection on :func:`_student_t_cdf` so the package keeps a
    numpy-only dependency footprint.  Accurate to ~1e-10 over the range this project
    uses.

    Raises:
        ValueError: if ``p`` is outside ``(0, 1)`` or ``df`` is not positive.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p!r}")
    if df <= 0:
        raise ValueError(f"df must be positive, got {df!r}")
    if p == 0.5:
        return 0.0

    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class ErrorSummary:
    """Point estimates and 95% intervals for a vector of per-image errors."""

    n: int
    mean: float | None = None
    mean_ci_low: float | None = None
    mean_ci_high: float | None = None
    mean_ci_half_width: float | None = None
    median: float | None = None
    median_ci_low: float | None = None
    median_ci_high: float | None = None
    median_ci_half_width: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, for JSON serialisation."""
        return asdict(self)


def error_summary(
    errors: Sequence[float] | Iterable[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    random_state: int = BOOTSTRAP_SEED,
) -> ErrorSummary:
    """Summarise per-image errors with 95% confidence intervals.

    Args:
        errors: one error value per validation image (non-finite entries are dropped).
        confidence: two-sided confidence level.
        bootstrap_replicates: resamples used for the median interval.
        random_state: seed making the bootstrap reproducible.

    Returns:
        An :class:`ErrorSummary`.  With no finite input every field except ``n`` is
        ``None`` -- callers must handle that rather than receive a fabricated zero.

    Raises:
        ValueError: if ``confidence`` is outside ``(0, 1)`` or
            ``bootstrap_replicates`` is not positive.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    if bootstrap_replicates <= 0:
        raise ValueError(f"bootstrap_replicates must be positive, got {bootstrap_replicates!r}")

    values = np.asarray(list(errors), dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return ErrorSummary(n=0)

    mean = float(np.mean(values))
    if n == 1:
        return ErrorSummary(
            n=1,
            mean=mean,
            mean_ci_low=mean,
            mean_ci_high=mean,
            mean_ci_half_width=0.0,
            median=mean,
            median_ci_low=mean,
            median_ci_high=mean,
            median_ci_half_width=0.0,
        )

    alpha = 1.0 - confidence
    std = float(np.std(values, ddof=1))
    standard_error = std / math.sqrt(n)
    half_width = student_t_ppf(1.0 - alpha / 2.0, df=n - 1) * standard_error

    # Draw one resample at a time. Vectorising this into a single (R, n) draw is
    # faster but consumes the RNG stream in a different order, which would silently
    # change every interval relative to the archived runs. Fidelity wins here.
    rng = np.random.default_rng(random_state)
    boot_medians = np.array(
        [np.median(rng.choice(values, size=n, replace=True)) for _ in range(bootstrap_replicates)]
    )
    median_low = float(np.percentile(boot_medians, 100.0 * alpha / 2.0))
    median_high = float(np.percentile(boot_medians, 100.0 * (1.0 - alpha / 2.0)))

    return ErrorSummary(
        n=n,
        mean=mean,
        mean_ci_low=mean - half_width,
        mean_ci_high=mean + half_width,
        mean_ci_half_width=float(half_width),
        median=float(np.median(values)),
        median_ci_low=median_low,
        median_ci_high=median_high,
        median_ci_half_width=float((median_high - median_low) / 2.0),
    )
