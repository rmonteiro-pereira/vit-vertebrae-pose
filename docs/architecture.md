# Architecture

## The dependency rule that shapes everything

The package is split so that **the analysis half never imports `torch`**.

```
                     no torch, no dataset, no network
        ┌────────────────────────────────────────────────────┐
        │  statistics.py      confidence intervals            │
        │  loss_names.py      loss registry keys              │
        │  models/names.py    model registry keys             │
        │  config.py          validated experiment configs    │
        │  results/           run archive + aggregation       │
        │  data/annotations   YOLO keypoint parsing           │
        │  data/augment       deterministic augmentation      │◄── needs PIL + numpy only
        └────────────────────────────────────────────────────┘
                                  ▲
                                  │ lazy import
        ┌─────────────────────────┴──────────────────────────┐
        │  models/backbones   ViT + ViTPose wrappers          │
        │  models/heads       shared prediction heads         │
        │  losses.py          10 objectives                   │◄── needs torch
        │  metrics.py         pixel error, PCK                │
        │  data/dataset       torch Dataset                   │
        │  training.py        single-fold loop                │
        └────────────────────────────────────────────────────┘
```

Why it matters: this repository's *product* is the evidence, not the model. A reviewer
should be able to clone it, run `pytest`, regenerate every figure and verify every
published number without a 2 GB CUDA download. `pyproject.toml` puts `torch` behind the
`train` extra for exactly this reason.

The rule is kept by two mechanisms:

- Registry **names** live in torch-free modules (`loss_names.py`, `models/names.py`),
  and the torch-dependent registries are built from them.
  `tests/test_losses.py::test_registry_matches_the_torch_free_name_list` fails if the
  two drift apart.
- `vitvert.models.__getattr__` and `vitvert.data.__getattr__` resolve the heavyweight
  symbols lazily, so `import vitvert.config` costs nothing.

## Data flow

```
image + label file
      │
      ├─ annotations.parse_annotation ──→ (K, 3) normalised x, y, visibility
      │       strict: truncated lines and partial triplets raise
      │
      ├─ augment.AugmentationPipeline ──→ image AND landmarks, one call
      │       deterministic: seed = sha256(image name) + per-transform salt
      │
      ├─ HuggingFace image processor ───→ (C, H, W) pixel values
      │
      ▼
KeypointDataset  ──→  {"pixel_values", "keypoints", "labels"}
      │
      ▼
model(pixel_values) ──→ {"keypoints": (B, K, 2) in [0, 1], "classification"}
      │
      ├─ losses.get_loss_function(name)  ──→ scalar, in pixels
      └─ metrics.per_image_pixel_error   ──→ (B,) px, nan where nothing is visible
                    │
                    ▼
          statistics.error_summary ──→ mean, median, 95% intervals
```

## The four decisions worth explaining

### One call site for augmentation

Each transform maps image **and** landmarks in a single call and returns both. The
original code applied the transform to the image in one place and re-derived the same
random crop geometry in a second place to move the landmarks — two copies of the same
arithmetic that had to agree by inspection. Divergence would have mislabelled the
training set while every loss curve stayed healthy.
[ADR 0003](adr/0003-single-call-site-augmentation.md).

Tests find the landmark in the pixels (brightness centroid of a synthetic marker) and
assert it went where the label says it went.

### Determinism from content, not from RNG state

Augmentation parameters derive from `sha256(image_name) + salt`, never from global RNG
state. Consequences: the same image gets the same augmentation on every worker process,
in every epoch, on every machine; a preprocessing cache is sound; and a rerun
reproduces a run. Distinct salts per transform are enforced at pipeline construction,
because equal salts would silently correlate a crop with a flip.

Note the trade-off, since it is a real one: fixed-per-image augmentation is *not* the
usual stochastic-per-epoch augmentation. The model sees one augmented view of each
image, not a fresh one each epoch. That is a weaker regulariser and may contribute to
the mixed augmentation results in [`results.md`](results.md).

### Padding is invisible, and invisible means excluded

A padded landmark slot is `(0, 0, 0)`: coordinate zero **and** visibility zero. Every
loss and every metric filters on visibility, so padding contributes nothing. In the
original code only six of eleven losses honoured the visibility channel; the rest
trained on the `(0, 0)` padding, pulling predictions toward the top-left corner.
[ADR 0004](adr/0004-uniform-visibility-masking.md).

Where a metric has nothing to average, it returns `nan`, not `0.0` — a flattering zero
would silently improve the reported mean.

### Unknown names raise

`get_loss_function("l2")` raises `KeyError`. An unrecognised config key raises
`ConfigError` at load time. The original resolver warned and substituted a default; the
archive contains three completed 100-epoch runs whose recorded objective is a name that
was never in the registry. See [`engineering.md`](engineering.md).

## Shared head, swapped backbone

Every model is `backbone → pooled vector → KeypointHead`, with the head identical
across all seven variants: `Linear(hidden, 512) → ReLU → Dropout(0.3) →
Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 2K) → sigmoid`.

Holding the head fixed is what makes the comparison a comparison of representations. It
also caps what the benchmark can say: a backbone whose strength is *spatial* structure
is being asked to express it through a single globally-pooled vector.

Pooling differs by family and is not a free choice: `vit-base` uses the `[CLS]` token,
which its pretraining optimised; the ViTPose backbones have no `[CLS]`, so their
feature map is mean-pooled over the sequence.

## Modules

| Module | Responsibility | torch |
|---|---|:-:|
| `statistics.py` | Student-*t* and bootstrap intervals | no |
| `loss_names.py` | loss registry keys and aliases | no |
| `config.py` | experiment config, validated at load | no |
| `results/records.py` | typed access to `runs.jsonl` | no |
| `results/aggregate.py` | published aggregation rules | no |
| `data/annotations.py` | strict YOLO keypoint parsing | no |
| `data/augment.py` | deterministic keypoint-aware transforms | no |
| `losses.py` | 10 objectives, uniform visibility masking | yes |
| `metrics.py` | per-image pixel error, PCK | yes |
| `models/heads.py` | shared keypoint and classification heads | yes |
| `models/backbones.py` | ViT and ViTPose wrappers | yes |
| `data/dataset.py` | torch `Dataset` over a split directory | yes |
| `training.py` | single-fold loop, best-checkpoint tracking | yes |

## What was left behind

The private repository carries roughly 6,700 further lines: a profiling interceptor, a
file-operation logger that wraps `open`/`torch.save`/`json.dump`, an asynchronous
checkpoint queue, a staged-IO layer, a preprocessing cache, and three overlapping
training orchestrators. That machinery was a response to one Windows workstation's
disk and GPU behaviour, and it is not part of a published benchmark. The *measurements*
it produced are preserved in [`engineering.md`](engineering.md), which is the part with
transferable value. [ADR 0005](adr/0005-distilled-training-loop.md).
