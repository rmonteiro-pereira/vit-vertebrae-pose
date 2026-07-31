# Results

Every number here is derived from `results/runs.jsonl` by
`scripts/aggregate_results.py`. Nothing is transcribed by hand, and CI re-derives the
whole grid on every push.

Read [`limitations.md`](limitations.md) alongside this page. In particular: the
run-to-run noise floor is larger than the gap between the best cell and the baseline,
so the *ranking* below is not resolvable at one seed per cell.

---

## The evidence chain

```
private research repo                    this repository
─────────────────────                    ───────────────
models/<hash>/summary.json      ┐
models/<hash>/run_<hash>/       ├─ tools/export_runs.py ─→ results/runs.jsonl   (169 runs)
  results/experiment_results.json ┘                              │
                                                                 │ scripts/aggregate_results.py
                                                                 ▼
                                                    results/aggregated_metrics.json  (48 cells)
                                                                 │
                                              ┌──────────────────┴──────────────────┐
                                              ▼                                     ▼
                                        figures/*.png                      README claims
                                    scripts/make_figures.py            tests/test_results.py
```

`tools/export_runs.py` copies an allowlisted set of fields and drops everything else,
including the absolute paths the training code wrote into its own artefacts.
`tests/test_no_patient_data.py` re-checks the archive's shape on every run.

## What is in the archive

| | |
|---|---:|
| Runs | 169 |
| Epochs trained, total | 20,098 |
| Runs using L1 | 143 |
| Other objectives | 26, across 8 losses |
| Architectures | 9 labels (7 shipped as builders) |
| Validation frames | 297 in every run |
| Training frames | 1,191 (30 in the few-shot runs) |
| Cells after aggregation | 48 |

## Aggregation rules

Reproduced unchanged from the published analysis, in `vitvert/results/aggregate.py`:

1. **L1 only.** Architecture is compared at a fixed objective; the loss study is
   separate and too thin to support conclusions.
2. **Completed runs only**, and only those that reached
   `max(5, 10% of the configured epoch budget)` epochs.
3. **Cell key** is `(model, configuration)` where configuration is
   `{Fine-tuned, Frozen} × {Control, Aug, Expand}`. `Expand` takes precedence over
   `Aug` when both flags are set.
4. `best_loss` is the **minimum over the cell**.
5. Every other metric comes from the **single run with the lowest best validation
   loss**, and that run's hash is carried into the output so each published number
   points at exactly one run.

Rule 5 is optimistic. It is retained for consistency with the article and flagged in
[`limitations.md`](limitations.md) §5.

## Primary endpoint

Mean, over the 297 validation frames, of the mean Euclidean distance between the two
predicted and annotated landmarks, in pixels of the network input.

Aggregating **per frame** before averaging is deliberate: it makes the resampling unit
one patient frame, so the confidence intervals do not treat two landmarks on the same
frame as independent observations.

**Baseline:** `vitpose-b`, Fine-tuned / Control, run `231898b6645d`, **6.31 px**.

## Top fine-tuned cells

| Model | Configuration | Mean px error | vs baseline | Run |
|---|---|---:|---:|---|
| `vitpose-s` | Fine-tuned / Aug | 5.18 ± 0.40 | −17.8% | `bd486fc68aa1` |
| `vitpose++-s` | Fine-tuned / Aug | 5.21 ± 0.35 | −17.5% | `0fac3a90118c` |
| `transpose-b`* | Fine-tuned / Expand | 5.31 ± 0.40 | −15.8% | `2801012a5201` |
| `vitpose-b` | Fine-tuned / Expand | 5.33 ± 0.39 | −15.5% | `02c3c400d8ee` |
| `vitpose++-l` | Fine-tuned / Aug | 5.52 ± 0.38 | −12.6% | `616b3bcf4094` |

\* `transpose-b` is not TransPose — see [`limitations.md`](limitations.md) §3. It is
shown because it appears in the published table, not because the row is trustworthy.

The spread across this table is 0.34 px, against a measured noise floor of 1.20 px. The
ordering within it carries no information.

## Frozen backbone

Across all 23 cells where the same model and data policy were run both fine-tuned and
frozen, freezing was worse in **23 of 23**.

| | Penalty |
|---|---:|
| Smallest | +35.7% (`vitpose-l`, Control) |
| Median | +104.1% |
| Largest | +558.1% (`transpose-b`, Expand) |

This is the one finding whose magnitude clears the noise floor comfortably, and the
only architectural claim in the README that survives §1 of the limitations.

Mechanistically it is unsurprising: the shared head sees a single pooled vector, so a
frozen backbone reduces the model to a linear-ish map from a fixed representation onto
two coordinates. What is worth reporting is the *size* of the effect — a 2× median
error increase — and that it is universal across nine architectures.

## Augmentation

Per-model change in mean pixel error from enabling online augmentation, fine-tuned:

| Model | Control | Aug | Δ |
|---|---:|---:|---:|
| `vitpose-l` | 7.23 | 5.79 | **−20.0%** |
| `vitpose++-s` | 6.13 | 5.21 | −15.1% |
| `vitpose-s` | 5.75 | 5.18 | −9.8% |
| `vitpose++-l` | 6.04 | 5.52 | −8.6% |
| `transpose-b`* | 8.23 | 8.27 | +0.6% |
| `vitpose++-b` | 6.00 | 6.14 | +2.3% |
| `vitpose-b` | 6.31 | 6.86 | +8.7% |
| `vit-base` | 5.71 | 6.54 | +14.5% |
| `hrformer-b`* | 9.65 | 15.10 | +56.5% |

The sign flips. Several individual deltas sit inside the noise floor, so the honest
reading is not "augmentation helps model X and hurts model Y" but **"a single pooled
augmentation effect would have been meaningless here"** — a pooled average over this
column is near zero and would have hidden a ±20% spread.

The augmentation policy is crop jitter, horizontal flip, vertical flip and colour
jitter, all deterministic per image (`vitvert/data/augment.py`). Vertical flip on a
lateral cervical radiograph is anatomically implausible as a real acquisition, which
may explain part of the negative results; the archive does not isolate the individual
transforms, so this stays a hypothesis.

## Scale

Best fine-tuned cell per model:

| Model | Parameters | Best px error |
|---|---:|---:|
| `vitpose-s` | 30,895,749 | 5.18 |
| `vitpose++-s` | 30,895,749 | 5.21 |
| `vitpose-b` | 121,820,421 | 5.33 |
| `vitpose++-l` | 429,944,837 | 5.52 |
| `vitpose-l` | 429,944,837 | 5.79 |
| `vitpose++-b` | 121,820,421 | 5.92 |

Monotonically *worse* with scale across the ViTPose family, though every gap except
small-vs-large sits within the noise floor. The defensible statement is the negative
one: **14× more parameters bought nothing measurable** on ~1,200 training frames with
two landmarks. The parameter counts come from the original repository's model audit.

This contradicts the conclusion in the original Portuguese README, which named
ViTPose++ Large the optimal configuration. That conclusion predates the corrected
pixel-error aggregation; the numbers above supersede it.

## Full grid

`figures/error_grid.png`, and `results/aggregated_metrics.json → cells` for the values.
Blank cells were never run.

## Reproducing

```bash
uv run python scripts/aggregate_results.py          # rewrite aggregated_metrics.json
uv run python scripts/aggregate_results.py --check   # verify it matches (CI runs this)
uv run python scripts/make_figures.py                # regenerate figures/
uv run pytest tests/test_results.py                  # 27 assertions over the numbers above
```
