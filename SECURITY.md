# Security and data handling

## The one thing that matters here

This project was built on **licensed videofluoroscopy of real patients**. The code and
the aggregate metrics are publishable. The frames carry burned-in acquisition metadata
(study date, time, technical parameters, and a device and site banner) that constitutes
an indirect identifier when attached to identifiable anatomy.

**No radiograph and no patient identifier reaches this repository through the data
path — not in `results/`, not in `figures/`, not in a test fixture, and not in the
history.**

There is exactly one deliberate exception, and it is not on that path:
[`docs/presentation/`](docs/presentation/) holds the talk this work was presented as,
and eight of its figures are built on frames from the corpus. They are published at
the author's instruction, as the party who holds the relationship with the data
provider. Before committing, every burned-in identifier was removed from them: study
date, acquisition time, technical parameters and the device and site banner, including
the copy of that banner showing through the translucent plot legend, which is invisible
at normal contrast and legible after a stretch. The redaction fills each cleared region
with a single flat value rather than its local background, so contrast-stretching
recovers no glyph silhouette. The anatomy is unaltered — the redaction changes **zero**
pixels inside the field of view.

The anatomy itself is not de-identified. A lateral fluoroscopy frame shows a facial
profile and dentition. Anyone re-using those figures should treat them as identifiable
imagery published with permission, not as an anonymised set.

Everything else here is a claim, so it is enforced rather than asserted.

## How it is enforced

### The bridge is an allowlist

`tools/export_runs.py` is the only path from the private research repository to this
one. It copies a fixed list of metric fields and drops everything else — including the
absolute filesystem paths the original training code wrote into its own artefacts. A
new field cannot appear in `results/runs.jsonl` without editing that allowlist and the
test that mirrors it.

### CI fails the build

`tests/test_no_patient_data.py` runs on every push and every pull request:

| Check | What it catches |
|---|---|
| No `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.dcm`, `.nii`, `.npy`, `.npz` anywhere | a frame or an array of one |
| No `.avi`, `.mp4`, `.mov`, `.mkv` | a source study |
| Committed PNGs are charts, not photographs | a frame pasted in as a "figure" — a tone-distribution heuristic |
| No `.ipynb` | the unreviewable base64 channel (ADR 0001) |
| No `.pt`, `.pth`, `.ckpt`, `.safetensors` | weights that could be inverted or that carry licence obligations |
| Run archive matches a fixed key allowlist | an export widened without review |
| Run archive contains no path, filename, video id, frame id, date or "patient" | an identifier leaking through a metric field |
| Every array in the archive is per-epoch length, not per-image | per-frame error vectors, which are patient-linked |
| No file over 5 MB | a blob nobody looked at |

A second CI job scans the **entire git history** for the same file types, so a leak
committed and later deleted still fails.

### `.gitignore` fails closed

`.gitignore` blocks `data/`, `Dados/`, `dataset/`, `**/images/`, `**/labels/`, every
raster format except PNG, every video format, and all weight formats. Deleting a rule
from it does not help: the CI checks above are independent of it, and one of them
asserts the rules are present.

## Verifying it yourself

```bash
uv run pytest tests/test_no_patient_data.py -v

# every file type that should never appear, across all history
git log --all --diff-filter=A --name-only --format= | sort -u \
  | grep -Ei '\.(jpe?g|bmp|tiff?|dcm|avi|mp4|pt|pth|ipynb)$'
```

The second command should print nothing.

## Reporting a problem

If you believe this repository contains an image, an identifier, or anything derived
from patient data:

**Do not open a public issue.** Contact the repository owner privately through GitHub.
Include the file path and, if you can, the commit. It will be treated as urgent.

The same applies to any secret, credential, hostname or private IP address. None should
be present; if one is, report it privately.

## Scope

There is no deployed service and no released package here, so there is no vulnerability
surface in the usual sense. This document is about **data disclosure**, which is the
only security property this repository has to get right.

For anything else — a dependency advisory, an unsafe deserialisation path — a normal
GitHub issue is fine.

## Note on `torch.load`

Nothing in this repository loads a checkpoint from an untrusted source. If you adapt
`scripts/train.py` to resume from a downloaded checkpoint, remember that
`torch.load` executes pickle by default; pass `weights_only=True`.
