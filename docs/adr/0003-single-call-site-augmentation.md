# ADR 0003 — Transforms map image and landmarks in one call

**Status:** accepted · 2026-07

## Context

The original augmentation pipeline was built on `torchvision`-style transforms with the
signature `transform(image) -> image`. Landmarks are not images, so they were handled
separately: `Dataset._apply_single_transform_with_keypoints` inspected each transform's
type, **re-seeded the same RNG with the same salt**, re-drew the same random parameters,
recomputed the same crop box, and applied the resulting geometry to the landmarks.

Concretely, this arithmetic existed twice:

```python
scale = max(min(rng.uniform(min_scale, max_scale), 1.0), 0.5)
crop_w = max(int(width * scale), 1)
offset_x = int(width / 2 + shift_x * max_dx - crop_w / 2)
offset_x = max(0, min(offset_x, max_dx))
```

once in `DeterministicCropJitter.__call__` for the pixels, and once in the dataset for
the labels. They had to agree exactly, forever, by inspection.

The failure mode this creates is specific and nasty. A change to one copy — a different
clamp, a rounding change, an added parameter draw that shifts the RNG stream — produces
a **systematically shifted label**. That is still a learnable target: the loss curve
converges, the validation error looks plausible, and nothing anywhere reports a
problem. The error only surfaces when someone overlays a prediction on an image, which
is precisely the artifact this project cannot publish.

## Decision

The transform interface takes and returns both:

```python
def __call__(self, image, keypoints, *, image_id) -> tuple[Image, np.ndarray]:
```

Each transform draws its parameters once and applies them to pixels and labels in the
same function body. There is no second copy to keep in sync.

Verified rather than asserted: `tests/test_augment.py` renders a synthetic marker,
locates it in the output by brightness centroid, and checks it landed where the label
says it did — for every geometric transform.

## Alternatives rejected

**Albumentations.** It has first-class keypoint support and solves this correctly, and
it was already a dependency of the original project. Rejected for two reasons. The
project needs *content-addressed* determinism — parameters derived from the image name,
identical across worker processes and epochs — which does not fit Albumentations'
global-RNG model without fighting it. And a dependency here would pull OpenCV into a
package that otherwise needs only PIL and numpy, for four transforms totalling about
sixty lines.

**Keep two call sites, add a consistency test.** Cheapest change, and the test would
have caught divergence. Rejected because it treats a structural problem as a testing
problem: the duplication remains, and the guarantee is only as good as the test's
coverage of parameter combinations. Removing the second copy makes the class of bug
unrepresentable.

**Have transforms return their sampled parameters for the caller to apply.** Keeps the
image-only signature and removes the duplicated draw. Rejected as a half-measure — the
caller still has to know how to turn a crop box into a coordinate mapping, so the
geometry knowledge stays split across two modules.

## Consequences

The transforms are no longer drop-in `torchvision` components, so they cannot be
composed with `transforms.Compose`. `AugmentationPipeline` replaces it in about ten
lines, and additionally rejects duplicate salts at construction — two transforms
sharing a salt would draw identical parameter sequences for the same image, correlating
a crop with a flip in a way that is invisible in any metric.

## What would reverse this

Needing a transform library's full catalogue (elastic deformation, grid distortion,
domain-specific medical augmentations). At that point Albumentations' keypoint pipeline
is worth the dependency, and the determinism requirement would have to be met by
seeding per sample inside the dataset instead.
