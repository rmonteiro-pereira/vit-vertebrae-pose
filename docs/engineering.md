# Engineering notes

Consolidated from about twenty scratch analysis files in the private repository. The
measured findings are kept; the projections are kept **and labelled as projections**,
because that distinction is the point.

> **Provenance and honesty.** The profiler exports and the file-operation log that
> produced these numbers stayed in the private repository — they were written to
> gitignored directories. They are therefore quoted, not committed, and are the one
> class of number in this repository that does **not** trace to a committed artifact.
> They are reported here because the diagnostic reasoning is transferable; treat the
> figures as a recorded observation on one machine, not as a reproducible benchmark.
>
> A second point, stated plainly: **no doc in the original set contains a post-fix
> re-measurement.** Every "after" figure below was an expectation. None was verified.

---

## Part 1 — Three defects found while preparing this release

These are the findings with the strongest evidence, because the evidence is in the
committed archive.

### 1.1 An unknown loss name silently trained the wrong objective

The original loss resolver responded to an unrecognised name by printing a warning and
returning the Euclidean loss:

```python
if loss_type in loss_functions:
    return loss_functions[loss_type]
else:
    print(f"⚠️ Loss type '{loss_type}' not found. Using 'euclidean' as default.")
    return loss_functions['euclidean']
```

`results/runs.jsonl` contains **three completed 100-epoch runs** recorded as
`loss_type: cosine_similarity` — a name that was never a key in that dictionary. Those
runs optimised the Euclidean loss under a label saying otherwise: about 300 epochs of
mislabelled evidence.

The archive corroborates it. Their best validation losses are 5.15, 6.05 and 7.34,
sitting squarely in the range of the runs actually labelled `euclidean` (5.57, 7.55,
8.30). A cosine objective of the form `0.7·(1 − cos) + 0.3·(d / diagonal)` is bounded
near 2.0 and cannot reach those values.

**Fix.** `get_loss_function` raises `KeyError` with the available names. `ConfigError`
does the same for unknown configuration keys — including near-miss typos like
`freeze_backone`, which the original `dict.get` pattern would have ignored while
running the opposite experiment.

*The general lesson: a default-on-unknown in a configuration path converts a typo into
a silent, expensive, undetectable wrong answer. Fail loudly at the boundary.*

### 1.2 The confidence-interval multiplier was not an approximation of anything

```python
t_approx = 2.0 + (30 - n) * 0.05
margin = min(t_approx * se, 2.576 * se)   # n < 30
margin = 1.96 * se                        # n >= 30
```

Two problems. At n = 10 this yields 2.576 against a true `t(9, 0.975) = 2.2622` — an
interval 14% too wide. And it is **discontinuous at n = 30**: 2.05 at n = 29 versus
1.96 at n = 30, so collecting one more observation widened the interval.

**Fix.** An exact Student-*t* inverse CDF (`vitvert/statistics.py`), monotone in n and
converging to the normal quantile.

**No published number changes.** Every published interval used n = 297, i.e. the
`n >= 30` branch, where the old multiplier was 1.96 and the exact value is
`t(296, 0.975) = 1.9680` — a 0.4% difference. `tests/test_statistics.py` pins both
halves: that the old rule was materially wrong, and that it was harmless here.

*Worth saying explicitly: the defect was real and the impact was nil. Reporting both is
what makes the first half credible.*

### 1.3 Augmentation geometry was computed in two places

`DeterministicCropJitter.__call__` cropped the image. A separate branch in
`Dataset._apply_single_transform_with_keypoints` re-seeded the same RNG, re-drew the
same scale and shift, recomputed the same crop box, and used it to move the landmarks.
Two independent copies of the same arithmetic, required to agree exactly.

They did agree. But a change to one — a clamp, a rounding rule, a default — would have
mislabelled the training set while every loss curve continued to look healthy, because
a systematically shifted label is still a learnable target.

**Fix.** Each transform maps image and landmarks in one call and returns both
([ADR 0003](adr/0003-single-call-site-augmentation.md)). `tests/test_augment.py` locates
a synthetic marker by brightness centroid and asserts the pixels and the label agree
after every transform.

---

## Part 2 — The training-throughput investigation

### The symptom

Production training ran far slower than the reference notebook on the same machine and
data: **~50.3 s/epoch against ~12.9 s/epoch**.

### Profiling, and what it got wrong first

An early pass showed **235 s of a 250 s epoch unaccounted for** by the instrumented
timers, and reported an implausible 0.276 s for a single tensor comparison
(`target_vis > 0.1`).

Both symptoms had the same cause: **the timing contexts were absorbing pending CUDA
work.** CUDA kernels are asynchronous, so a timer that happens to wrap the first
host-side read after a queue of GPU work attributes the whole queue to itself. The
0.276 s comparison was not slow; it was the first synchronisation point after it.

*This is the most transferable item on the page. A GPU profile taken with host-side
wall-clock timers and no explicit synchronisation does not measure what it appears to
measure.*

### The measured profile

Instrumented at full granularity, 3 epochs, batch size 128:

| | |
|---|---:|
| `epoch_train`, 3 epochs | 150.78 s (~50.26 s/epoch) |
| `dataloader_get_next_batch` | 134.26 s — **89.1%** of batch iteration |
| `dataset_vit_processor` (HF `ViTImageProcessor`) | 114.41 s CPU, 4,592 calls, 24.9 ms/image |
| `dataset_apply_transform` | 43.04 s CPU, 9.4 ms/image |
| `dataset_image_open` | 9.51 s |
| loss computation | 8.67 s, of which `loss.item()` synchronisation 6.64 s |
| First batch of the run | 46.41 s (CUDA warm-up) |

**Read the units carefully.** Dataset CPU time sums to 170.84 s while the wall-clock
cost was 134.26 s, because the workers run in parallel; the component percentages sum
to ~104%. The original doc noted this and it is worth preserving, because "75.9% of the
epoch" mixes CPU time against wall time and overstates the case. The defensible
statement is the ordering: **HuggingFace image preprocessing dominated the input
pipeline, by a wide margin, and the GPU was starved.**

### What was done about it

| Change | Rationale | Claimed effect |
|---|---|---|
| Disk cache of preprocessed tensors, keyed on `md5(path, mtime, model, transform)` | preprocessing is deterministic when augmentation is off | 70–76% epoch reduction — **projection, never verified** |
| Auto-disable the cache when random transforms are present | a cached augmented tensor is a silently frozen augmentation | correctness, not speed |
| `persistent_workers=True`, lazy `__init__`, module-level `worker_init_fn` | Windows uses `spawn`: every worker re-imports the module and re-runs `Dataset.__init__` each epoch | 25–30 s/epoch — **derived from a hypothetical, never measured** |
| `pin_memory` whenever CUDA is available; `non_blocking=True` transfers; explicit next-batch prefetch | overlap host-to-device copies with compute | qualitative only |
| Explicit 3-iteration CUDA warm-up | removes the 46.4 s first-batch outlier from the measurement | measured symptom, plausible fix |
| Profiling behind an env var, no-op by default | instrumentation that changes the thing it measures | — |

The Windows `spawn` analysis is the piece worth keeping in full: on Windows there is no
`fork`, so each `DataLoader` worker starts a fresh interpreter, re-imports the module
and re-constructs the `Dataset`. Anything expensive in `__init__` — validating paths,
opening a cache, loading an image processor — is paid per worker per epoch unless
workers persist. `vitvert/data/dataset.py` keeps `__init__` to a directory listing and
takes the processor as a constructor argument for this reason.

Two claims in the original docs are worth flagging as unsupported. The headline
"29 seconds of per-epoch worker overhead" originates as *"if `Dataset.__init__` takes
7 seconds, that's 28+ seconds with 4 workers"* and hardens into a measured fact in
later documents. And a "lambda wrapper prevents PyTorch JIT compilation" diagnosis does
not survive scrutiny — eager PyTorch does not JIT through or around a lambda. The
warm-up and precomputed-constant changes bundled with it are defensible on their own
terms; the JIT explanation is not.

---

## Part 3 — Checkpoint and I/O reliability

### What the instrumentation found

A file-operation interceptor wrapping `open`, `torch.save`/`load`, `pickle`, `json`,
`PIL.Image.open` and `cv2.imread` was run over a training session that had driven the
disk to 100% utilisation. Over ~19 minutes:

| | |
|---|---:|
| File operations | 3,855 |
| Written | 24.10 GB |
| Read | 5.89 GB |
| `experiment_results.json` writes | **101** (once per epoch, plus one) |
| `torch.save` checkpoints | 15, at 1.61 GB each |
| Time spent inside `torch.save` | 8–15 s each, ~2.38 min total |
| `TORCH_SAVE_ERROR` log entries | 101 |
| Label-file reads | 0 (the in-memory annotation cache was working) |

Two findings stand out.

**A 13 KB results file was read, parsed, mutated and rewritten inside the epoch loop —
101 times — even when the checkpoint interval was set to 10.** The interval throttled
the 1.6 GB checkpoints and not the small file that was actually in the hot path.

**Checkpoints cost 2.38 minutes of a 19-minute run**, synchronously, on the training
thread.

### The 101 errors, and a contradiction worth naming

Two documents disagree about what the 101 `TORCH_SAVE_ERROR` entries were.
`DISK_BOTTLENECK_ANALYSIS.md` calls them *"failed attempts to save `last_epoch.pt` …
wasted I/O"*. `ASYNC_JSON_AND_TEMPFILE_FIXES.md` says the *logger* choked on the
`tempfile._TemporaryFileWrapper` used by atomic saves, and the saves themselves
succeeded.

The second explanation fits the mechanism: an atomic write is write-to-temp-then-rename,
and the interceptor tried to read a filename from an object that has none. **Whether any
checkpoint was actually lost is not recorded anywhere, and this repository does not
claim to know.** It is listed here as an unresolved item rather than quietly resolved in
favour of the convenient answer.

### Fixes applied

- Background checkpoint saver with a priority queue (best models first), a bounded
  queue with synchronous fallback, and a state-dict copy to CPU before the handoff.
- Checkpoint interval with always-save rules: first epochs, last epoch, and any epoch
  that improves the best metric.
- Results JSON moved to a single-worker executor; **the final save stays synchronous**,
  so process exit cannot race the last write. This is the right call and worth noting:
  asynchrony everywhere is not the goal, asynchrony off the hot path is.
- Atomic writes throughout, temp-file plus rename.

All "after" numbers for this section (writes 101 → ~15, errors 101 → 0, 8–25 minutes
saved per run) were projections. The last of those does not even reconcile with the
measured 2.38 minutes of total save time in the observed run, and is quoted here only
to be retired.

### What survives into this repository

`scripts/train.py` writes atomically (temp file, then rename) and writes results once
at the end rather than once per epoch. The asynchronous saver, the priority queue, the
staged-IO layer and the file-operation interceptor are **not** ported: they solved one
machine's disk contention at the cost of several hundred lines in the critical path of
a benchmark whose value is its evidence.
[ADR 0005](adr/0005-distilled-training-loop.md).

---

## Part 4 — A silent configuration bug caught by counting parameters

During a retraining review, four model variants — `vitpose-s`, `vitpose-l`,
`vitpose++-s`, `vitpose++-l` — were found to have trained with **121,820,550
parameters**, the *base* backbone size, rather than the 30.9 M and 429.9 M their labels
implied. The variant key was not reaching the backbone constructor, so every size
resolved to base.

Nothing in the loss curves indicated this. A small model and a large model both train,
both converge, and both produce a plausible number. The bug was found by **auditing
parameter counts against the published architectures**, which is now the same check
that exposed `hrformer-b` at 9,536 parameters
([`limitations.md`](limitations.md) §3).

*Cheap, boring, and it catches a class of bug that no loss curve will ever reveal:
assert the size of the thing you think you built.*
