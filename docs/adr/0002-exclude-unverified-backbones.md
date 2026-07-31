# ADR 0002 — Ship no builder for `hrformer-b` and `transpose-b`, keep their results

**Status:** accepted · 2026-07

## Context

The published grid contains nine model labels. Two of them, `hrformer-b` and
`transpose-b`, are hand-rolled partial re-implementations rather than wrappers around a
published checkpoint — roughly 900 lines of convolutional stem, transition blocks and
fallback paths, with `NotImplementedError` on several branches.

The original repository's parameter audit reports:

| Label | Trainable parameters | Published architecture |
|---|---:|---:|
| `hrformer-b` | 9,536 | ~43 M |
| `transpose-b` | 9,600 | ~17 M |

9,536 parameters is smaller than the shared prediction head alone. Whatever those runs
trained, it was not HRFormer-B or TransPose-B. Local `.pth` weight files for both exist
in the private tree (167 MB and 67 MB), so the most likely explanation is that the
loader fell through to a placeholder path — but the archive does not record which code
path each run took, and this cannot be settled after the fact.

Meanwhile `transpose-b / Fine-tuned / Expand` ranks **third** in the published top-five
table at 5.31 px.

## Decision

Two separate calls.

**The builders are not ported.** `vitvert.models` registers seven variants:
`vit-base` and six ViTPose. `build_model("hrformer-b")` raises `KeyError`.

**The results are retained.** `results/runs.jsonl` keeps every `hrformer-b` and
`transpose-b` run, they appear in `figures/error_grid.png`, and they are excluded from
every claim in the README. [`limitations.md`](../limitations.md) §3 states why, and
`tests/test_models_and_dataset.py::test_unverified_backbones_are_not_registered` pins
the exclusion.

## Alternatives rejected

**Port the builders as they are.** Cheapest, and it would keep the model list at nine.
Rejected because shipping code labelled `HRFormerBase` that instantiates 9,536
parameters invites a reader to treat it as a faithful implementation. In a repository
whose value proposition is trustworthy evidence, that is the wrong trade.

**Delete the rows from `runs.jsonl`.** Makes the grid uniform and the top-five table
defensible without a footnote. Rejected outright: deleting measured results because they
are inconvenient is exactly the practice this repository is trying to demonstrate the
opposite of. The rows are data; the labels are the problem.

**Debug and fix the builders.** The honest ideal. Rejected on scope: without a GPU, the
weights, and the dataset, a fixed builder could not be validated, and an unvalidated
"fix" would be worse than the documented exclusion.

**Rename them to `hrformer-b-stub` / `transpose-b-stub`.** Rejected because it silently
rewrites history: the archive says `hrformer-b`, the article says `hrformer-b`, and a
relabelled row would no longer match either.

## Consequences

The published top-five table has an asterisk on row three, which is the correct outcome
and is stated wherever the table appears.

A reader wanting HRFormer or TransPose numbers on this task will have to produce them.
That is the honest position.

## What would reverse this

A builder that instantiates the published parameter count, verified by an assertion in
`tests/`, and a fresh run under the same protocol. At that point the models go back in
and the archived rows are superseded rather than footnoted.
