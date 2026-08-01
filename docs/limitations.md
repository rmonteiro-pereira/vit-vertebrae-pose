# Limitations

What this benchmark does not establish. Read this before quoting any number from it.

---

## 1. The run-to-run noise floor exceeds the headline effect

This is the most important limitation and it invalidates the *ranking*, not the study.

`vitpose-x` and `vitpose++-x` load the **same checkpoint** (§2). Their cells are
therefore repeated measurements of one configuration under two labels. Across the 15
cells where both exist:

- mean paired difference: **−0.135 px** — no systematic effect, confirming they are the
  same network;
- largest paired difference: **1.196 px** (`vitpose-l` vs `vitpose++-l`, Fine-tuned /
  Control: 7.23 vs 6.04).

The headline result — best fine-tuned cell 5.18 px against a 6.31 px baseline — is a
gap of **1.124 px**. The noise floor is larger than the effect.

**Therefore:** the direction of the frozen-backbone finding (§4) is safe, since it
holds in 23 of 23 cells with penalties up to +558%. The *ranking among fine-tuned
configurations is not*. `vitpose-s / Aug` at 5.18 px cannot be said to beat
`vitpose++-l / Aug` at 5.52 px; the two are indistinguishable at one seed per cell.

Fixing this needs 3–5 seeds per cell and a between-seed variance component
(`sqrt(within² + between²)`). That was planned in the original project and executed for
only a small subset of cells; the published grid is single-seed.

`tests/test_results.py::test_run_to_run_noise_floor_exceeds_the_best_versus_baseline_gap`
asserts this rather than leaving it to prose.

## 2. The two ViTPose families are the same network

Original ViTPose checkpoints are not published on the Hub, so `vitpose-x` and
`vitpose++-x` both resolve to `usyd-community/vitpose-plus-x`. Parameter counts at
equal scale are byte-identical:

| Scale | `vitpose-*` | `vitpose++-*` |
|---|---:|---:|
| small | 30,895,749 | 30,895,749 |
| base | 121,820,421 | 121,820,421 |
| large | 429,944,837 | 429,944,837 |

Any "ViTPose vs ViTPose++" comparison in this repository, or in the article derived
from it, is a comparison of a model with itself.

## 3. `hrformer-b` and `transpose-b` are not HRFormer and TransPose

The original repository's parameter audit reports **9,536** trainable parameters for
its `hrformer-b` builder and **9,600** for `transpose-b` — three to four orders of
magnitude below the published architectures, and below the size of the shared
prediction head alone. Whatever those runs trained, it was not HRFormer-B or
TransPose-B.

Their rows are retained in `results/runs.jsonl` and in the grid figure because deleting
measured data is worse than labelling it, and the public package ships no builder for
them ([ADR 0002](adr/0002-exclude-unverified-backbones.md)).

They are **excluded from the model ranking** — README §2 and the top-five table in
[`results.md`](results.md) are restricted to the seven shipped backbones — and
**marked in place wherever else they appear**: the frozen-backbone claim (README §1)
spans all nine labels and its worst cell, +558%, is `transpose-b / Expand`; the
augmentation table (README §3) carries `hrformer-b`; the full grid (README §5) shows
both. No claim in this repository depends on treating either label as the
architecture it names.

Note that `transpose-b / Fine-tuned / Expand` ranks third in the top-five table of the
original article at 5.31 px. That ranking should not be relied on.

## 4. Validation-only, one split, one site

- No held-out test set. Every number is a **validation** number, and the best epoch was
  selected on the same split it is reported on. Reported errors are optimistic by an
  unmeasured amount.
- One institution, one acquisition setup. No external validation, no demographics, no
  scanner metadata, no acquisition-year stratification.
- `k_folds` is inconsistent across the archive: some runs used 5-fold
  cross-validation, others a single split. The grid mixes them. A standardised
  benchmark would retrain everything under one protocol.

## 5. Replicate selection is optimistic

Where a cell contains several runs, `best_loss` is the minimum over the cell and every
other metric comes from the run that achieved it. Taking a minimum over replicates
biases downward. The rule is reproduced unchanged from the published analysis so the
numbers here match the article; it is not defended.

## 6. Intervals are not predictive uncertainty

The `±` values are 95% confidence intervals on a **summary statistic** (mean or median
per-image error) over a fixed validation split. They describe how much that summary
would move under resampling of the same 297 frames. They say nothing about how far a
prediction on a new frame is likely to fall from the landmark, and they are not
calibrated model uncertainty.

The interval arithmetic itself was corrected during this release
(`vitvert/statistics.py`), but the fix does not change any published number: all
published intervals used n = 297, where the old and new multipliers agree to 0.4%.

## 7. Pixels, not millimetres

No per-study pixel-spacing table exists in the archive, so pixel errors cannot be
converted to millimetres. Without that conversion there is **no clinical tolerance
band, no non-inferiority claim against manual annotation, and no clinical claim of any
kind**. A 5.18 px error on a 256×192 input is 1.6% of the frame diagonal; what that
means anatomically is not established here.

## 8. Two landmarks, not a spine model

Only the inferior endplates of C2 and C4 are localised. This is not vertebral
segmentation, not full-spine landmarking, and not a curvature or alignment measurement.

## 9. Inter-rater agreement was gated but never quantified

The annotation protocol routes disagreements above a threshold to a third rater, so
the labels are filtered for consistency — but the threshold, the disagreement
distribution and the resulting agreement statistic are not in the archive. The label
noise floor is therefore unknown, which matters given §1: some of the 1.2 px
run-to-run spread may be label noise rather than optimisation noise.

## 10. The public training loop is a re-implementation

`vitvert/training.py` is a distilled replacement for three overlapping orchestrators in
the private research code. It reproduces the *procedure* — same objective, optimiser,
scheduler, best-checkpoint criterion and per-epoch metric — not the archived numbers
bit-for-bit. Running it will not regenerate `results/runs.jsonl`.

The **aggregation** path is different: it is verified. The reimplementation in
`vitvert/results/` reproduces all 48 published cells and every metric to the last
decimal against the original export, and `scripts/aggregate_results.py --check`
enforces it in CI ([ADR 0005](adr/0005-distilled-training-loop.md)).

## 11. Loss-comparison runs are thin, and one set is mislabelled

Of 169 archived runs, 143 use L1. The remaining 26 spread across eight other
objectives, 1–7 runs each — too few for any loss-function conclusion, which is why the
grid is L1-only.

Three of them, archived as `loss_type: cosine_similarity`, actually optimised the
Euclidean loss: that name was never in the original registry and the resolver silently
substituted a default. See [`engineering.md`](engineering.md).

## 12. `Expand` is under-documented

The `Expand` data policy draws on an offline-expanded frame set. The archive records
that the flag was on but not what the expansion consisted of, so `Expand` cells cannot
be reproduced from this repository even with the dataset in hand.
