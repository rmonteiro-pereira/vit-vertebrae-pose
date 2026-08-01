# The dataset

**Not distributed with this repository. Not obtainable from it.** This page describes
what the data is, what its licence permits, what layout the code expects, and how to
obtain it yourself.

## What the images are

Frames extracted from **lateral videofluoroscopy studies** — X-ray video of the head
and neck, acquired clinically. Frames were contrast-enhanced with CLAHE, exported in
YOLO keypoint format, and annotated with **two landmarks**: the inferior endplate of
**C2** and the inferior endplate of **C4**.

Annotation followed a written protocol executed in ImageJ at the research group. Its
material points, preserved here because they govern how the labels should be read:

- C1 is located by its characteristic anatomical profile; C2 through C5 are then
  counted downward from it.
- Vertebrae only partially inside the frame are skipped unless the structure is
  unambiguous.
- Each frame is annotated independently by **two raters**. Where the two markings
  disagree by more than a pre-set distance threshold, the frame goes to a **third**
  rater for re-annotation.
- Inter-rater agreement was used as an inclusion gate. It was **not** quantified into
  a published statistic, so no rater-agreement figure appears anywhere in this
  repository — see [`limitations.md`](limitations.md).

Split sizes recorded in every archived run: **1,191 training frames, 297 validation
frames** for the primary fold.

## What the licence permits

The source video material comes from a repository whose data is governed by the
**Creative Commons BY-NC-SA 3.0** licence together with the host's ground rules. The
two clauses that decide what this repository may contain, quoted verbatim:

> "Password-protected data cannot be posted on other websites or servers or shared
> with anyone who does not already have password access."

> "This license precludes the incorporation of the data in commercial products,
> including systems such as large language models (LLMs) such as ChatGPT."

So: **use** is permitted for non-commercial research with attribution and share-alike.
**Re-posting** is not. So the corpus is not here: no frame of it is committed as data,
and neither the annotations nor the splits are published.

Two further constraints apply on top of the licence and would apply even if the
licence were permissive:

1. **The frames carry burned-in acquisition metadata.** Study date, acquisition time
   and technical parameters are rendered into the image corners by the fluoroscopy
   unit, alongside a device and site banner. A study date attached to an identifiable
   anatomy is an indirect identifier.
2. **A licence to use is not a licence to republish patient imaging.** Publishing
   clinical images in a portfolio is a separate judgement from the copyright question.

Both constraints were engaged once, deliberately, for the talk under
[`presentation/`](presentation/): eight of its figures are built on frames. The first
constraint was answered by removing every burned-in identifier before committing; the
second was answered by the author, who holds the relationship with the data provider
and instructed that the deck be published. Nothing else changed — every figure under
[`../figures/`](../figures/) is still a chart over aggregate metrics, and the data path
still carries no imagery at all. [`../SECURITY.md`](../SECURITY.md) records what was
removed and what was not.

## Obtaining the data

There is no download link in this repository, and running any script here will not
fetch anything.

1. **Source video material.** The underlying videofluoroscopy studies are held in a
   public research repository of swallow studies, hosted within the TalkBank family of
   corpora and developed by a university swallow-disorders research laboratory. Access
   to the clinical corpora is granted by the repository maintainers on request, under
   the ground rules quoted above. Start from <https://talkbank.org/> and its data
   ground rules at <https://talkbank.org/0share/rules.html>.
2. **Frame selection, CLAHE enhancement and landmark annotation** were performed by
   the research group that produced this study; the annotated derivative is not the
   host's to distribute and is not published. Reproducing the annotation from the
   source video requires re-executing the protocol summarised above.

> **Open item — the attribution here is incomplete.** The exact corpus name and the
> citation it requires are still to be confirmed with the research group and the corpus
> maintainers, and this section updated with the precise citation. The description above
> is what the archived material supports; it deliberately stops short of naming a corpus
> this repository cannot verify, because a wrong attribution would be worse than one
> marked incomplete.
>
> What this release publishes is metrics, plus the eight redacted frames illustrating
> the talk under [`presentation/`](presentation/). It does not publish the corpus or
> the annotated derivative: no annotation, no split, and no quantity of frames from
> which either could be reconstructed. Whether the annotated derivative could be
> shared is a separate question, and this repository does not answer it: it does not
> share it.

## Expected layout

`scripts/train.py --data-root` points at a **fold directory**:

```
<fold>/
├── train/
│   ├── images/
│   │   ├── frame_0001.jpg
│   │   └── ...
│   └── labels/
│       ├── frame_0001.txt
│       └── ...
└── valid/
    ├── images/
    └── labels/
```

Image and label share a stem; `images` maps to `labels` and the suffix becomes `.txt`
(`vitvert.data.annotations.label_path_for`).

## Annotation format

One line per image:

```
<class_id> <cx> <cy> <w> <h> <x1> <y1> <v1> <x2> <y2> <v2>
```

| Field | Meaning |
|---|---|
| `class_id` | integer class (a single class is used) |
| `cx cy w h` | bounding box, normalised to `[0, 1]` |
| `xi yi` | landmark *i*, normalised to `[0, 1]` |
| `vi` | visibility: `0` absent, `1` occluded but labelled, `2` clearly visible |

Rules the parser enforces, each of which corresponds to a way the original parser
could fail quietly:

- Fewer than 8 fields is an **error**, not an unannotated image. A truncated label file
  previously trained as a blank, contributing a zero-visibility target with no warning.
- Keypoint fields must arrive in complete triplets; a trailing partial triplet is an
  error rather than being discarded.
- Visible landmarks must lie inside `[0, 1]`. Invisible landmarks may sit anywhere,
  since `(0, 0, 0)` is the padding convention.
- An empty file is legitimate and means "no annotated landmark".

## Pretrained weights

Not distributed here either, and not required to run the analysis.

| Model | Source | Notes |
|---|---|---|
| `vit-base` | `google/vit-base-patch16-224` | fetched from the Hub on first use |
| `vitpose-{s,b,l}` | `usyd-community/vitpose-plus-{small,base,large}` | see the warning below |
| `vitpose++-{s,b,l}` | `usyd-community/vitpose-plus-{small,base,large}` | identical to the row above |

> **Both ViTPose families load the same checkpoints.** Original ViTPose weights are not
> published on the Hub. The consequence for the benchmark is set out in
> [`limitations.md`](limitations.md).

Check the licence attached to each checkpoint on the Hub before use; the MIT licence
covering this repository's code does not extend to them.
