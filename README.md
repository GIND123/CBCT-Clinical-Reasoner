# CBCT Clinical Reasoner

A reproducible ToothFairy4 starter repository for ODIN 2026 Task 1: generating a
maxillofacial and surgical-planning report from one 3D cone-beam CT volume.

This repository supplies an executable retrieval baseline, strict data and output
validation, local development metrics, the official Grand Challenge container contract,
tests, CI, and a detailed competition analysis. It is a research scaffold—not a medical
device, diagnostic system, or clinically validated model.

## What is implemented

- Dataset discovery for CBCT NIfTI volumes and one-to-many English reports.
- Bounded-memory 3D feature extraction with geometry, intensity, histogram, and pooled
  projection features.
- A deterministic nearest-neighbor report-retrieval baseline with a safe, versioned
  `.npz` artifact.
- Exact challenge output: `/output/diagnostic-imaging-report.json` containing
  `{"report": "Generated report text."}`.
- Lightweight BLEU-4, METEOR-style, and clinical-term development diagnostics.
- A privacy-preserving leaderboard snapshot with participant and team names removed.
- Docker, command-line tools, unit tests, linting, and GitHub Actions.

The baseline is deliberately transparent and runnable on CPU. It establishes plumbing
and a reproducible lower bound; it does **not** infer tooth-level pathology or provide
clinical reasoning. The recommended competition model is an anatomy-to-ontology-to-text
pipeline described in [docs/method.md](docs/method.md).

## Challenge at a glance

| Item | ToothFairy4 Task 1 |
|---|---|
| Input | One preprocessed 3D jaw CBCT per case |
| Training format | One compressed NIfTI (`.nii.gz`) volume per case |
| Output | One English textual clinical report |
| Public data | 625 stated patients and 1,001 reports per language |
| Hidden test | 50 cases from an independent center |
| Final submissions | Two per team; the better submission is ranked |
| Public evaluator | BLEU-4 and METEOR, with local/off-platform RadFact support |
| Clinical review | Blinded expert assessment is described for top systems |
| Prizes | EUR 1,500 / 1,000 / 500 |

The benchmark emphasizes dental status, bone quality and quantity, critical anatomy,
anatomical variation, and procedure-related risks for implant placement, extraction,
sinus lift, and other maxillofacial workflows. The released data contain subsets P, F,
S, and A; the external-center design makes cross-center validation central rather than
optional.

One source-quality warning matters: the [dataset page](https://ditto.ing.unimore.it/toothfairy4/)
states 625 total patients but lists subset counts of 417 + 63 + 52 + 100 = 632. This
repository does not guess which number is wrong. It discovers actual case directories,
fails on incomplete cases, and records the observed count in a manifest.

## Anonymized test-phase leaderboard

Snapshot supplied for this repository on **16 August 2026**. Participant usernames,
team names, profile links, team links, and evaluation links have been removed. Anonymous
entry IDs identify rows only and are not stable identity hashes. Public algorithm labels
are retained for method-level analysis.

| Place | Anonymous entry | Algorithm label | Created | Mean position |
|---:|---|---|---:|---:|
| 1st | Entry-01 | tf4 baseline | 14 Aug 2026 | 1.5 |
| 1st | Entry-02 | Stage Seg GPU | 14 Aug 2026 | 1.5 |
| 3rd | Entry-03 | ToothFairy4 Retrieval Baseline | 14 Aug 2026 | 3.0 |
| 4th | Entry-04 | SINUS | 14 Aug 2026 | 4.5 |
| 4th | Entry-05 | NaiveBaseline | 16 Aug 2026 | 4.5 |
| 6th | Entry-06 | cprvlm | 10 Aug 2026 | 6.0 |
| 7th | Entry-07 | baseline1 | 13 Aug 2026 | 7.0 |
| 8th | Entry-08 | ODIN2026 Task1 SK | 11 Aug 2026 | 8.0 |
| 9th | Entry-09 | TF4 baseline2 | 15 Aug 2026 | 9.5 |
| 10th | Entry-10 | CBCT Report Retrieval Baseline | 9 Aug 2026 | 10.0 |

The machine-readable copy is
[data/leaderboard_test_phase_anonymized.csv](data/leaderboard_test_phase_anonymized.csv).
Use [scripts/anonymize_leaderboard.py](scripts/anonymize_leaderboard.py) for a later
snapshot; never commit the identity-bearing input CSV.

### What the snapshot does—and does not—show

- There are 10 displayed submissions across eight calendar days. Two pairs are tied,
  which explains fractional mean positions of 1.5 and 4.5.
- Six labels explicitly contain “baseline” or “naive.” A labeled retrieval baseline is
  third, so retrieval is a serious sanity check and competition floor at this snapshot.
- The shared lead between a generic baseline label and a segmentation-oriented label
  supports anatomy-aware modeling as a promising direction, but names alone do not
  establish implementation details.
- Only aggregate mean position is visible. There are no per-metric values, confidence
  intervals, case-level errors, runtime data, or clinical-review results. A 1.5 versus
  3.0 mean position is therefore not evidence of a clinically meaningful gap.
- The board is time-sensitive and should be treated as an interim snapshot, not the
  final ODIN 2026 result.

See [docs/competition_analysis.md](docs/competition_analysis.md) for the full analysis,
risk register, ablation plan, and recommended submission strategy.

## Pipeline

```text
CBCT (.nii.gz locally / .mha on Grand Challenge)
        |
        v
bounded 3D sampling -> geometry + intensity + projection features
        |
        v
robust feature scaling -> nearest training case
        |
        v
consensus reference report -> schema validation -> {"report": "..."}
```

This baseline never uses a network call and can run in an offline evaluation container.
Retrieval provenance is available locally for error analysis but is never written to the
challenge output.

## Quick start

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest
```

Download ToothFairy4 through the organizer’s authenticated
[dataset portal](https://ditto.ing.unimore.it/toothfairy4/). Do not commit volumes,
reports, trained artifacts, or patient-derived outputs.

Expected local layout:

```text
/path/to/toothfairy4/
├── A001/
│   ├── cbct/volume.nii.gz
│   └── reports_en/
│       ├── report_1.txt
│       └── report_2.txt
├── F001/
├── P001/
└── S001/
```

Validate and inventory the data:

```bash
cbct-reasoner index \
  --data /path/to/toothfairy4 \
  --output artifacts/manifest.jsonl
```

Fit the baseline:

```bash
cbct-reasoner train \
  --data /path/toothfairy4 \
  --output model/retrieval_model.npz
```

Predict one case and inspect local-only provenance:

```bash
cbct-reasoner predict \
  --model model/retrieval_model.npz \
  --input /path/to/case/cbct/volume.nii.gz \
  --output predictions/diagnostic-imaging-report.json \
  --show-provenance

cbct-reasoner validate-output predictions/diagnostic-imaging-report.json
```

## Local evaluation

Prepare JSON Lines with one case per line:

```json
{"case_id":"A001","prediction":"...","reference":"..."}
```

Then run:

```bash
cbct-reasoner evaluate --pairs artifacts/validation_pairs.jsonl \
  --output artifacts/metrics.json
```

Every metric emitted by this command is labeled as a development proxy. For an official
comparison, use the organizer’s pinned evaluator. Its current open implementation uses
BLEU-4 and METEOR and can optionally run RadFact; the challenge states that submissions
are re-evaluated locally because not all metrics run on Grand Challenge.

Use leave-one-center-out validation and report each center separately. A random
patient-level split measures in-domain interpolation and will likely overestimate hidden
external-center performance. Multiple reports for one patient must remain on the same
side of every split.

## Grand Challenge container

The implementation follows the organizer’s published interface:

| Interface | Container path |
|---|---|
| `cbct-image` | `/input/images/cbct/*.mha` |
| `diagnostic-imaging-report` | `/output/diagnostic-imaging-report.json` |
| model artifact | `/opt/ml/model/retrieval_model.npz` |

Train the artifact before building:

```bash
docker build -t cbct-clinical-reasoner:0.1.0 .
```

Local smoke test:

```bash
docker run --rm \
  -v "$PWD/test/input/images/cbct:/input/images/cbct:ro" \
  -v "$PWD/test/output:/output" \
  cbct-clinical-reasoner:0.1.0
```

The image is CPU-compatible. If the feature encoder is replaced by a GPU model, pin its
CUDA runtime, test without internet access, preserve the same input/output paths, and
include every checkpoint in the saved image.

## Repository map

```text
.
├── configs/                 experiment intent and validation policy
├── data/                    anonymized, non-clinical snapshots only
├── docs/                    analysis, data, method, and submission guides
├── model/                   local trained artifact location (ignored by Git)
├── scripts/                 privacy-preserving maintenance utilities
├── src/cbct_reasoner/       package, CLI, model, metrics, and GC entrypoint
├── tests/                   synthetic unit and contract tests
├── Dockerfile               offline inference image
└── pyproject.toml           package and development tooling
```

## Recommended competition direction

The strongest defensible system is a staged structured predictor:

1. Normalize orientation and physical spacing while preserving scanner metadata.
2. Segment and label teeth, jaws, mandibular canals, maxillary sinuses, and other
   report-relevant structures.
3. Predict ontology-grounded findings with tooth number, laterality, measurement,
   uncertainty, and evidence coordinates.
4. Render fluent English from only those structured findings, with constrained tooth
   numbering and explicit negation handling.
5. Ensemble at the finding level, not by unconstrained prose voting.
6. Calibrate on held-out centers and audit hallucinations, missed critical structures,
   laterality swaps, and measurement errors with clinicians.

This design aligns lexical overlap with factual correctness while retaining traceability.
It also lets segmentation labels from earlier ToothFairy releases contribute without
making segmentation itself the final output.

## Reproducibility and safety

- Record dataset release/checksum, subset counts, split manifest, package lock, source
  commit, checkpoint checksums, GPU/runtime, and decoding parameters for every run.
- Declare every public training dataset, pretrained model, and external annotation.
  The challenge forbids undisclosed private data and requires fully automatic inference.
- Never infer “normal” merely because a detector is uncertain. Separate `absent`,
  `present`, `uncertain`, and `not assessable` states.
- Treat generated text as a draft for qualified clinician review. Do not use this code
  for patient care.
- Check dataset and model licenses independently of this repository’s MIT code license.

## Primary sources

- [ODIN 2026 challenge overview](https://odin2026.grand-challenge.org/)
- [Official ToothFairy4 dataset page](https://ditto.ing.unimore.it/toothfairy4/)
- [Official submission policy and Task 1 contract](https://odin2026.grand-challenge.org/how-to-submit/)
- [Organizer repository](https://github.com/AImageLab-zip/ToothFairy/tree/65b7f93796ac8f61046585dd5a52964b09890d0f/ODIN2026/ToothFairy4)
- [ODIN 2026 challenge document, DOI 10.5281/zenodo.19727377](https://doi.org/10.5281/zenodo.19727377)
- [ODIN 2026 workshop challenge page](https://odin-workshops.org/2026/challenges/)

Challenge facts and interface details were checked on 16 August 2026. The anonymized
leaderboard values come from the snapshot supplied with this repository and may change.

## License

Code is released under the [MIT License](LICENSE). Dataset access, clinical reports,
model weights, third-party code, and challenge submissions remain subject to their own
terms.
