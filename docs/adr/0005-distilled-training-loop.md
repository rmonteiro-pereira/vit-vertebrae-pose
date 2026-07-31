# ADR 0005 — Distil the training loop; verify the aggregation instead

**Status:** accepted · 2026-07

## Context

The private repository's training path is about 2,900 lines across three modules that
overlap heavily and mutate the same run directory:

| Module | Lines |
|---|---:|
| `training_kfold.py` | 1,286 |
| `training_orchestrator.py` | 867 |
| `training_manager.py` | 743 |

Around them sit a further ~2,300 lines of supporting machinery: a checkpoint manager
(1,044), a file-operation interceptor that monkey-patches `open`, `torch.save`,
`pickle` and `json` (638), a profiling layer (364), an asynchronous checkpoint queue
(346), a preprocessing cache (334) and a staged-IO layer (149).

Almost all of it exists because of one Windows workstation: `spawn`-based dataloader
workers, 1.6 GB checkpoints saturating a spinning disk, and CUDA-synchronisation
artifacts in the profiler. Real problems, well diagnosed
([`engineering.md`](../engineering.md)) — and specific to a machine that is not part of
the published benchmark.

Porting it verbatim would have meant carrying ~5,200 lines whose behaviour cannot be
tested here (no GPU, no dataset, no weights) into a repository whose value proposition
is that its evidence can be checked.

## Decision

**Distil the training loop.** `vitvert/training.py` is 250 lines: one fold, one
optimiser, one scheduler, best-checkpoint tracking, per-epoch metrics. Same objective,
same optimiser (`AdamW` over trainable parameters only), same scheduler options, same
best-checkpoint criterion, same per-epoch metric as the original. Atomic writes are kept
because they are correctness, not tuning; the profiler, the interceptor, the async queue
and the staged-IO layer are dropped.

**State plainly that it is not bit-compatible.** The archived numbers were produced by
the original code and ship as data in `results/runs.jsonl`. Running this loop will not
regenerate them. The README and [`limitations.md`](../limitations.md) §10 say so.

**Verify the path that actually matters.** The aggregation — the code that turns 169
archived runs into the 48 published cells — *is* a faithful reimplementation, and it is
proven:

- all 48 cells reproduced against the original paper export;
- every metric matching to the last decimal: `best_loss`, `best_pixel_error`,
  `median_pixel_error`, both interval half-widths, and the traceable run hash;
- `scripts/aggregate_results.py --check` enforces it in CI on every push.

So the honesty split is explicit: **the numbers are verified, the trainer is not.** That
is the correct place to spend the verification budget, because the numbers are what a
reader is being asked to believe.

## Alternatives rejected

**Port all 5,200 lines verbatim.** Maximum fidelity, zero interpretation. Rejected: the
result is a repository that cannot be reviewed, cannot be tested, and whose bulk is
Windows-specific I/O plumbing irrelevant to the finding. It would also have carried the
silent-fallback loss resolver and the two-call-site augmentation with it.

**Ship no training code at all — publish only the analysis.** Cleanest, and defensible
given that the dataset is unavailable anyway. Rejected because "here are results from
code you cannot see" is a weaker artifact than "here is the procedure, honestly labelled
as a re-implementation". A reader can read 250 lines and judge whether the procedure is
sound.

**Port verbatim, then refactor incrementally with tests.** The right answer with a GPU
and the dataset in hand. Rejected here on the same ground as ADR 0002: a refactor that
cannot be validated is not a refactor, it is a rewrite with extra steps.

## Consequences

`vitvert/training.py` reaches 99% line coverage, because a 250-line loop can be
exercised against a two-parameter synthetic model — including the paths that matter:
best-checkpoint selection, detachment of the saved state dict, frozen parameters staying
frozen, empty loaders producing `nan` rather than `0.0`, and a fully-frozen model being
rejected rather than trained pointlessly.

None of that was testable in the original design.

The Windows throughput work is preserved as documentation instead of code, which is
where its transferable value was: the diagnosis (`spawn` re-runs `Dataset.__init__`;
host-side timers absorb pending CUDA work) outlives the specific fix.

## What would reverse this

Reproducing an archived run bit-for-bit becoming a requirement — for a rebuttal, an
audit, or a reviewer request. Then the original code has to be resurrected under its
original dependency pins, and this loop stays as the maintained path.
