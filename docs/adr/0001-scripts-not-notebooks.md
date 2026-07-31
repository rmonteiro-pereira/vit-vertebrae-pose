# ADR 0001 — Analysis ships as scripts, not notebooks

**Status:** accepted · 2026-07 · supersedes the original repository's `notebooks/`

## Context

The private research repository committed three notebooks: `eda.ipynb` (1.0 MB),
`visualize_training_curves.ipynb` (3.4 MB) and `visualize_runs.ipynb`. Between them
they held 1,266 stored outputs, six of which were base64-encoded PNG images.

The dataset behind the project is licensed patient imaging. A committed notebook output
is a base64 blob inside a JSON document: it renders as an image in a browser but is
opaque to `git diff`, to code review, and to any grep-based check. There is no way to
look at a notebook diff and know whether a radiograph just entered the repository.

Inspecting the six images resolved this particular case — all six were charts. But the
mechanism remains: the next cell execution can embed anything, and nobody reviewing the
diff will see it.

## Decision

No notebooks in this repository. Analysis is `scripts/aggregate_results.py` and
`scripts/make_figures.py`, both of which read `results/runs.jsonl` and write PNGs to
`figures/`.

Two things enforce it:

- `*.ipynb` in `.gitignore`;
- `tests/test_no_patient_data.py::test_no_notebooks_are_committed`, so that deleting
  the ignore rule still fails CI.

## Alternatives rejected

**`nbstripout` in a pre-commit hook.** Strips outputs on commit, keeping the notebook
authoring experience. Rejected because it fails open: a hook that is not installed, or
is bypassed with `--no-verify`, leaves the blob in the commit, and the failure is
invisible until someone opens the file. A guard for patient data has to fail closed.

**Notebooks committed with outputs cleared by convention.** Same failure mode, minus
the tooling.

**`jupytext` paired scripts.** Genuinely good, and it would have preserved the
narrative-with-output authoring style. Rejected on cost: it adds a dependency and a
sync step to serve an authoring workflow this repository does not need, since the
analysis is four functions over one JSONL file.

## Consequences

Lost: inline narrative around outputs, and the ability to re-run one cell.

Gained: the figure pipeline became testable and CI-enforced. `make_figures.py` runs in
CI on every push, so a figure can never drift from the data that produced it — which is
the property the notebooks did not have and the one that actually matters here.

## What would reverse this

Nothing plausible while the dataset is patient imaging. If the project ever moves to a
fully public, non-clinical dataset, the trade-off changes and `jupytext` becomes
attractive again.
