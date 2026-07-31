"""Tests for interval estimation, including the defect that was fixed."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vitvert.statistics import BOOTSTRAP_SEED, error_summary, student_t_ppf


class TestStudentT:
    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (1, 12.706205),
            (2, 4.302653),
            (9, 2.262157),
            (29, 2.045230),
            (60, 2.000298),
            (296, 1.968010),
        ],
    )
    def test_matches_published_critical_values(self, df: int, expected: float) -> None:
        """t(df, 0.975) against a standard table, to six decimals."""
        assert student_t_ppf(0.975, df) == pytest.approx(expected, abs=1e-5)

    def test_converges_to_the_normal_quantile(self) -> None:
        """T decreases monotonically toward z = 1.959964 and reaches it in the limit."""
        normal_quantile = 1.9599639845
        values = [student_t_ppf(0.975, df) for df in (100, 1_000, 10_000, 1_000_000)]
        assert values == sorted(values, reverse=True)
        assert all(v > normal_quantile for v in values)
        assert values[-1] == pytest.approx(normal_quantile, abs=1e-5)

    def test_is_symmetric(self) -> None:
        assert student_t_ppf(0.025, 12) == pytest.approx(-student_t_ppf(0.975, 12), abs=1e-9)

    @pytest.mark.parametrize(("p", "df"), [(0.0, 5), (1.0, 5), (-0.1, 5), (0.5, 0), (0.5, -3)])
    def test_rejects_invalid_arguments(self, p: float, df: float) -> None:
        with pytest.raises(ValueError):
            student_t_ppf(p, df)


class TestFixedDefect:
    """The original code approximated t with ``2.0 + (30 - n) * 0.05`` below n = 30.

    These tests pin both halves of the story: the old rule was materially wrong at
    small n, and it was harmless at the n = 297 used for every published interval.
    """

    @staticmethod
    def _legacy_multiplier(n: int) -> float:
        if 1 < n < 30:
            return min(2.0 + (30 - n) * 0.05, 2.576)
        return 1.96

    def test_legacy_rule_was_materially_wrong_at_small_n(self) -> None:
        # The raw expression gives 3.0 at n = 10; the 2.576 cap clips it to 2.576.
        assert pytest.approx(3.0) == 2.0 + (30 - 10) * 0.05
        legacy = self._legacy_multiplier(10)
        exact = student_t_ppf(0.975, 9)
        assert legacy == pytest.approx(2.576)
        assert exact == pytest.approx(2.262157, abs=1e-5)
        # The legacy interval was ~14% too wide even after the cap.
        assert legacy / exact > 1.13

    def test_legacy_rule_was_discontinuous_at_the_threshold(self) -> None:
        below = self._legacy_multiplier(29)
        above = self._legacy_multiplier(30)
        assert below > above  # a wider interval from *more* data
        assert student_t_ppf(0.975, 28) > student_t_ppf(0.975, 29)  # the exact rule is monotone

    def test_published_intervals_are_unaffected(self) -> None:
        """Every published interval used n = 297, where old and new agree to <0.5%."""
        n = 297
        legacy = self._legacy_multiplier(n)
        exact = student_t_ppf(0.975, n - 1)
        assert abs(exact - legacy) / legacy < 0.005


class TestErrorSummary:
    def test_reproduces_a_hand_computed_interval(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        summary = error_summary(values)
        assert summary.n == 5
        assert summary.mean == pytest.approx(3.0)
        expected_half = student_t_ppf(0.975, 4) * (math.sqrt(2.5) / math.sqrt(5))
        assert summary.mean_ci_half_width == pytest.approx(expected_half, rel=1e-12)
        assert summary.mean_ci_low == pytest.approx(3.0 - expected_half)
        assert summary.mean_ci_high == pytest.approx(3.0 + expected_half)

    def test_drops_non_finite_values_rather_than_propagating_nan(self) -> None:
        summary = error_summary([1.0, float("nan"), 3.0, float("inf")])
        assert summary.n == 2
        assert summary.mean == pytest.approx(2.0)

    def test_empty_input_yields_none_not_zero(self) -> None:
        summary = error_summary([])
        assert summary.n == 0
        assert summary.mean is None
        assert summary.mean_ci_half_width is None

    def test_single_observation_has_a_degenerate_interval(self) -> None:
        summary = error_summary([7.5])
        assert summary.n == 1
        assert summary.mean == summary.median == pytest.approx(7.5)
        assert summary.mean_ci_half_width == 0.0

    def test_bootstrap_is_reproducible_across_calls(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.gamma(2.0, 2.0, size=200)
        first = error_summary(values, random_state=BOOTSTRAP_SEED)
        second = error_summary(values, random_state=BOOTSTRAP_SEED)
        assert first.median_ci_low == second.median_ci_low
        assert first.median_ci_high == second.median_ci_high

    def test_bootstrap_seed_actually_changes_the_interval(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.gamma(2.0, 2.0, size=200)
        assert (
            error_summary(values, random_state=1).median_ci_low
            != error_summary(values, random_state=2).median_ci_low
        )

    def test_median_interval_brackets_the_median(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.normal(10.0, 2.0, size=500)
        summary = error_summary(values)
        assert summary.median_ci_low <= summary.median <= summary.median_ci_high

    def test_interval_narrows_as_n_grows(self) -> None:
        rng = np.random.default_rng(11)
        small = error_summary(rng.normal(0, 1, size=30))
        large = error_summary(rng.normal(0, 1, size=3000))
        assert large.mean_ci_half_width < small.mean_ci_half_width

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"confidence": 0.0}, "confidence"),
            ({"confidence": 1.0}, "confidence"),
            ({"bootstrap_replicates": 0}, "bootstrap_replicates"),
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            error_summary([1.0, 2.0], **kwargs)
