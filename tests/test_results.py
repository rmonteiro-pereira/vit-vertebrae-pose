"""Tests for the archived-run loader and the published aggregation rules.

The most important test in the suite is
:meth:`TestPublishedNumbers.test_readme_headline_numbers`: every number quoted in the
README is re-derived here from ``results/runs.jsonl``.  If a README figure is ever
edited by hand, this fails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from vitvert.results import RunRecord, aggregate_runs, load_runs
from vitvert.results.aggregate import (
    baseline_row,
    config_label,
    image_diagonal_px,
    relative_improvement,
    top_fine_tuned,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_PATH = REPO_ROOT / "results" / "runs.jsonl"
AGGREGATED_PATH = REPO_ROOT / "results" / "aggregated_metrics.json"


@pytest.fixture(scope="module")
def runs() -> list[RunRecord]:
    return list(load_runs(RUNS_PATH))


@pytest.fixture(scope="module")
def rows(runs: list[RunRecord]):
    return aggregate_runs(runs)


def _record(**overrides) -> RunRecord:
    payload = {
        "model_hash": "abc123",
        "config": {"model_name": "vitpose-b", "loss_type": "l1", "num_epochs": 100},
        "history": {"valid_losses": [5.0] * 100, "train_losses": [4.0] * 100},
        "is_complete": True,
    }
    payload.update(overrides)
    return RunRecord.from_json(payload)


class TestRunRecord:
    def test_config_label_covers_the_full_grid(self) -> None:
        assert config_label(_record()) == "Fine-tuned / Control"
        assert (
            config_label(
                _record(config={"model_name": "m", "augmentation_params": {"enable_augmentation": True}})
            )
            == "Fine-tuned / Aug"
        )
        assert (
            config_label(_record(config={"model_name": "m", "freeze_backbone": True})) == "Frozen / Control"
        )

    def test_expand_beats_aug_when_both_are_set(self) -> None:
        record = _record(
            config={
                "model_name": "m",
                "augmentation_params": {"enable_augmentation": True},
                "dataset_expansion": {"enabled": True},
            }
        )
        assert config_label(record) == "Fine-tuned / Expand"

    def test_infinite_losses_are_excluded_from_the_best(self) -> None:
        record = _record(history={"valid_losses": [float("inf"), 3.0, float("inf")]})
        assert record.best_valid_loss == 3.0

    def test_all_infinite_losses_give_none_not_inf(self) -> None:
        assert _record(history={"valid_losses": [float("inf")]}).best_valid_loss is None

    def test_pixel_error_falls_back_to_the_per_epoch_minimum(self) -> None:
        record = _record(history={"valid_losses": [1.0], "valid_pixel_errors": [9.0, 4.0, 6.0]})
        assert record.mean_pixel_error == 4.0

    def test_ci_half_width_rejects_an_unknown_statistic(self) -> None:
        with pytest.raises(ValueError, match="mean' or 'median"):
            _record().ci_half_width("mode")

    def test_missing_model_hash_is_an_error(self) -> None:
        with pytest.raises(KeyError):
            RunRecord.from_json({"config": {}})


class TestLoader:
    def test_archive_is_present_and_non_trivial(self, runs: list[RunRecord]) -> None:
        assert len(runs) == 169

    def test_missing_file_names_the_recovery_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=re.escape("export_runs.py")):
            load_runs(tmp_path / "absent.jsonl")

    def test_malformed_line_reports_its_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "runs.jsonl"
        path.write_text('{"model_hash": "a", "config": {}}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match=":2:"):
            load_runs(path)

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "runs.jsonl"
        path.write_text('\n{"model_hash": "a", "config": {}}\n\n', encoding="utf-8")
        assert len(load_runs(path)) == 1


class TestAggregation:
    def test_produces_the_published_grid_size(self, rows) -> None:
        assert len(rows) == 48

    def test_only_l1_runs_reach_the_grid(self, runs: list[RunRecord], rows) -> None:
        assert all(row.loss_type == "l1" for row in rows)
        assert any(run.loss_type != "l1" for run in runs), "fixture should contain other losses"

    def test_output_is_deterministically_ordered(self, runs: list[RunRecord]) -> None:
        first = [(r.model, r.config) for r in aggregate_runs(runs)]
        second = [(r.model, r.config) for r in aggregate_runs(list(reversed(runs)))]
        assert first == second

    def test_short_runs_are_discarded(self) -> None:
        short = _record(
            config={"model_name": "m", "loss_type": "l1", "num_epochs": 100},
            history={"valid_losses": [5.0] * 9},
        )
        assert aggregate_runs([short]) == []

    def test_incomplete_runs_are_discarded(self) -> None:
        assert aggregate_runs([_record(is_complete=False)]) == []

    def test_best_loss_spans_the_cell_but_metrics_come_from_one_run(self) -> None:
        good = _record(
            model_hash="good",
            history={"valid_losses": [3.0] * 100},
            pixel_error={"mean": 5.0, "median": 4.0},
        )
        worse = _record(
            model_hash="worse",
            history={"valid_losses": [2.5] + [9.0] * 99},
            pixel_error={"mean": 42.0, "median": 40.0},
        )
        (row,) = aggregate_runs([good, worse])
        assert row.best_loss == 2.5  # minimum across the cell
        assert row.hash == "worse"  # the run that achieved it supplies the metrics
        assert row.best_pixel_error == 42.0

    def test_baseline_cell_exists(self, rows) -> None:
        baseline = baseline_row(rows)
        assert baseline is not None
        assert (baseline.model, baseline.config) == ("vitpose-b", "Fine-tuned / Control")

    def test_top_fine_tuned_excludes_frozen_cells(self, rows) -> None:
        assert all(row.config.startswith("Fine-tuned") for row in top_fine_tuned(rows, limit=20))

    def test_relative_improvement_is_none_rather_than_fabricated(self) -> None:
        assert relative_improvement(None, 6.0) is None
        assert relative_improvement(5.0, None) is None
        assert relative_improvement(5.0, 0.0) is None
        assert relative_improvement(3.0, 6.0) == pytest.approx(50.0)

    def test_image_diagonal_differs_between_families(self) -> None:
        assert image_diagonal_px("vit-base") == pytest.approx(316.78, abs=0.01)
        assert image_diagonal_px("vitpose-s") == pytest.approx(320.0)


class TestPublishedNumbers:
    """Re-derive every headline number the README quotes."""

    def test_committed_aggregate_matches_a_fresh_derivation(self, rows) -> None:
        committed = json.loads(AGGREGATED_PATH.read_text(encoding="utf-8"))
        derived = [row.as_dict() for row in rows]
        assert committed["cells"] == derived
        assert committed["n_cells"] == 48
        assert committed["n_runs_scanned"] == 169

    def test_readme_headline_numbers(self, rows) -> None:
        baseline = baseline_row(rows)
        assert baseline.best_pixel_error == pytest.approx(6.31, abs=0.005)

        best = top_fine_tuned(rows, limit=1)[0]
        assert (best.model, best.config) == ("vitpose-s", "Fine-tuned / Aug")
        assert best.best_pixel_error == pytest.approx(5.18, abs=0.005)
        assert best.best_pixel_error_ci_half_width == pytest.approx(0.40, abs=0.005)
        assert relative_improvement(best.best_pixel_error, baseline.best_pixel_error) == pytest.approx(
            17.8, abs=0.05
        )
        assert best.hash == "bd486fc68aa1"

    def test_frozen_backbone_is_worse_in_every_cell_where_both_exist(self, rows) -> None:
        fine = {
            (r.model, r.config.split(" / ")[1]): r.best_pixel_error
            for r in rows
            if r.config.startswith("Fine-tuned")
        }
        frozen = {
            (r.model, r.config.split(" / ")[1]): r.best_pixel_error
            for r in rows
            if r.config.startswith("Frozen")
        }
        shared = [k for k in fine if k in frozen and fine[k] and frozen[k]]
        assert len(shared) >= 15
        assert all(frozen[k] > fine[k] for k in shared), "the frozen-backbone claim must hold universally"

    def test_worst_frozen_penalty_matches_the_readme(self, rows) -> None:
        fine = {r.model: r.best_pixel_error for r in rows if r.config == "Fine-tuned / Control"}
        frozen = {r.model: r.best_pixel_error for r in rows if r.config == "Frozen / Control"}
        penalties = {m: (frozen[m] - fine[m]) / fine[m] * 100 for m in fine if m in frozen}
        assert max(penalties.values()) == pytest.approx(223.6, abs=0.5)
        assert max(penalties, key=penalties.__getitem__) == "transpose-b"

    @staticmethod
    def _identical_weight_pairs(rows) -> list[float]:
        """Signed error differences for cells whose two labels load the same checkpoint."""
        by_key = {(r.model, r.config): r for r in rows}
        differences = []
        for size in ("s", "b", "l"):
            for config in {r.config for r in rows}:
                plain = by_key.get((f"vitpose-{size}", config))
                plus = by_key.get((f"vitpose++-{size}", config))
                if plain and plus and plain.best_pixel_error and plus.best_pixel_error:
                    differences.append(plus.best_pixel_error - plain.best_pixel_error)
        return differences

    def test_identically_weighted_labels_show_no_systematic_difference(self, rows) -> None:
        """``vitpose-x`` and ``vitpose++-x`` load the same checkpoint.

        Their mean paired difference must sit at zero: a systematic gap would mean the
        two labels are not in fact the same network, contradicting the parameter audit.
        """
        differences = self._identical_weight_pairs(rows)
        assert len(differences) == 15
        assert abs(sum(differences) / len(differences)) < 0.2

    def test_run_to_run_noise_floor_exceeds_the_best_versus_baseline_gap(self, rows) -> None:
        """The headline ranking is not resolvable at one seed per cell.

        Pairs that load identical weights differ by up to 1.20 px purely from run-to-run
        variation, while the best fine-tuned cell beats the baseline by 1.13 px. This is
        the central caveat of the benchmark and it is asserted, not merely asserted in
        prose. See docs/limitations.md.
        """
        noise_floor = max(abs(d) for d in self._identical_weight_pairs(rows))
        baseline = baseline_row(rows)
        best = top_fine_tuned(rows, limit=1)[0]
        headline_gap = baseline.best_pixel_error - best.best_pixel_error

        assert noise_floor == pytest.approx(1.196, abs=0.01)
        assert headline_gap == pytest.approx(1.124, abs=0.01)
        assert noise_floor > headline_gap
