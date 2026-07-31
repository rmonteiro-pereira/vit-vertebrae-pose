# Architecture Decision Records

Each record states the decision, the alternative that was rejected and why, and the
condition that would reverse it.

| # | Decision | Notable because |
|---|---|---|
| [0001](0001-scripts-not-notebooks.md) | Analysis ships as scripts, not notebooks | Notebook outputs are an unreviewable channel for patient imagery |
| [0002](0002-exclude-unverified-backbones.md) | No builder for `hrformer-b` / `transpose-b`, but keep their results | Splits "code I will not ship" from "data I will not delete" |
| [0003](0003-single-call-site-augmentation.md) | Transforms map image and landmarks in one call | Removes a label-corruption failure mode by construction |
| [0004](0004-uniform-visibility-masking.md) | Every loss honours the visibility channel | An intentional behaviour change, declared rather than hidden |
| [0005](0005-distilled-training-loop.md) | Distil the trainer, verify the aggregation | The numbers are proven; the trainer is honestly labelled as not |
