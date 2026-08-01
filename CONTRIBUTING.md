# Contributing

## Setup

```bash
git clone https://github.com/rmonteiro-pereira/vit-vertebrae-pose
cd vit-vertebrae-pose
uv sync --all-extras
```

`uv sync` installs CPU-only PyTorch wheels by default — this repository analyses
archived results, and nothing in its test suite trains. If you are training, install
the CUDA build yourself.

## The loop

```bash
uv run pytest                                        # 243 tests, ~30 s
uv run ruff check . && uv run ruff format .          # lint and format
uv run mypy                                          # strict, on Python 3.12
uv run python scripts/aggregate_results.py --check   # numbers still match the evidence
uv run python scripts/make_figures.py                # figures still regenerate
```

CI runs exactly this on Python 3.11 and 3.12. If it passes locally it passes there.

## Three rules that are not negotiable

### 1. No patient data. Ever.

No image, no video, no array derived from one, no filename that identifies a study or a
frame. Not in a test fixture, not in a docstring, not in a figure.

This rule has exactly one exception and it is not yours to extend: the presentation in
`docs/presentation/`, whose figures were redacted and published on the data owner's
instruction (see [`SECURITY.md`](SECURITY.md)). A contribution that adds imagery will
be refused; if you believe yours is a second exception, open an issue and do not commit
it in the meantime.

`tests/test_no_patient_data.py` enforces this and a second CI job scans the full git
history. Read [`SECURITY.md`](SECURITY.md) before touching `tools/export_runs.py`,
`.gitignore` or anything under `figures/`.

If you need image-like data for a test, generate synthetic noise into `tmp_path` — see
`tests/test_models_and_dataset.py`.

### 2. No number that does not trace to a committed artifact

Every figure in the README and in `docs/` derives from `results/runs.jsonl` through
`scripts/aggregate_results.py`. If you add a claim, add the assertion that checks it in
`tests/test_results.py`.

A measured result that looks bad is a finding. An invented one that looks good is
fraud. If a number came from a run that cannot be reproduced, say so where the number
appears — `docs/engineering.md` shows the pattern.

### 3. Tests must be able to fail

A test that passes whatever the code does is worse than no test: it costs runtime and
buys false confidence. Prefer:

- closed-form expected values over "it returns something";
- error paths over happy paths — roughly half the suite is `pytest.raises`;
- properties over snapshots — e.g. `test_augment.py` finds the landmark in the pixels
  and asserts the label agrees, rather than comparing against a stored array.

## Where things go

| Adding | Goes in | Also update |
|---|---|---|
| A loss function | `src/vitvert/losses.py` | `src/vitvert/loss_names.py`, `tests/test_losses.py` |
| A model | `src/vitvert/models/backbones.py` | `src/vitvert/models/names.py`, `build_model` |
| A figure | `scripts/make_figures.py` | commit the PNG, reference it from `README.md` or `docs/` |
| An experiment config | `experiments/*.yaml` | nothing — `tests/test_config.py` picks it up automatically |
| A decision | `docs/adr/NNNN-slug.md` | `docs/adr/README.md` |

The dependency rule described in [`docs/architecture.md`](docs/architecture.md) is real:
`statistics`, `config`, `results`, `loss_names`, `models/names`, `data/annotations` and
`data/augment` must stay importable **without torch**. If you add a torch import to one
of them, the analysis pipeline stops working on a machine that has no deep-learning
stack, and that is the property this repository's reviewability rests on.

## Decisions get an ADR

Anything a future reader would otherwise have to reverse-engineer from the diff: a
dependency added, an interface changed, a component deliberately not shipped.

The format is short and the important part is the middle section — **the alternative you
rejected and the condition that would reverse the decision**. See
[`docs/adr/`](docs/adr/) for five worked examples.

## Commits and pull requests

Conventional-commit style, and the body explains *why*:

```
fix(statistics): use the exact Student-t quantile

The previous multiplier `min(2.0 + (30 - n) * 0.05, 2.576)` was 14% too wide at
n = 10 and discontinuous at n = 30, where more data produced a wider interval.
No published number changes: all published intervals used n = 297.
```

In a pull request, state what you measured, not only what you changed. If it affects a
published number, say which and by how much.
