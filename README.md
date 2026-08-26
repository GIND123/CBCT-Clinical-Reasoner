# CBCT Clinical Reasoner

CBCT Clinical Reasoner is an end-to-end pipeline for ODIN 2026 Task 1,
ToothFairy4. It turns one 3D cone-beam CT volume into a structured
maxillofacial and surgical-planning report. The repository includes data
inspection, preprocessing, label construction, model training, metric-matched
evaluation, report calibration, Hugging Face artifact synchronization, and a
Grand Challenge submission container.

> This is benchmark research software, not a medical device. Generated reports
> require review by a qualified clinician and must not be used directly for
> patient care.

## Project snapshot

| Item | Value |
|---|---:|
| Evaluated cases | 622 |
| English reference reports | 1,000 |
| Prototype statements | 989 |
| Reachable reference phrases | 89.0% |
| Cross-validation folds | 5 |
| Shallow model input features | 122 |
| Shallow model size | 52 KB |
| Neural encoder parameters | 29 million |
| Neural checkpoint total | 580 MB |
| Submission report | 8 core statements and 5 geometry-gated statements |

The real-data measurements use `toothfairy4_v03`. Patient-level grouping keeps
multiple reports from the same patient in the same split.

## Current measured results

The current adaptive submission report was measured over all 622 public cases.
The clinical values below use the repository's offline RadFact surrogate unless
an evaluation is explicitly run with `--radfact-lite`.

| System | Final | RadFact precision | RadFact recall | BLEU-4 | METEOR |
|---|---:|---:|---:|---:|---:|
| Adaptive submission, 8 core plus 5 gated | **0.4122** | 0.5220 | 0.4050 | 0.1268 | 0.3431 |
| Linear model, 122 features | 0.3544 | not reported | not reported | **0.1493** | 0.3064 |
| Corpus prior, no image features | 0.3575 | not reported | not reported | 0.1170 | 0.3082 |
| Fine-tuned encoder, 29M parameters | 0.3403 | not reported | not reported | 0.1331 | 0.2684 |
| Oracle within the prototype space | 0.6008 | not reported | not reported | 0.3410 | 0.4726 |

The challenge objective is:

```text
Final = 0.8 * RadFact-F1 + 0.2 * mean(BLEU-4, METEOR)
```

BLEU-4 matches NLTK to machine precision. METEOR matches the organizer's
`meteor_lite_score` implementation exactly. The surrogate is useful for local
ranking, but it is not a replacement for the official RadFact evaluation.

## Hugging Face resources

The primary inference artifacts are publicly downloadable. Access to the
submission and experiment repositories varies because some artifacts can
contain patient-derived report text:

- [Primary CBCT Clinical Reasoner artifacts](https://huggingface.co/GOVINDFROM/cbct-clinical-reasoner)
- [ODIN 2026 submission artifacts](https://huggingface.co/GOVINDFROM/cbct-clinical-reasoner-odin2026)
- [ToothFairy4 artifact backup](https://huggingface.co/GOVINDFROM/odin-toothfairy4-artifacts)
- [Experiment runs and metrics](https://huggingface.co/datasets/GOVINDFROM/cbct-clinical-reasoner-runs)
- [Qwen2.5 1.5B Instruct base model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

Set `HF_TOKEN` or a compatible token key in `.env` before using Hub commands.
New repositories created by this project are private by default.

## Published models: use and reproducibility

The primary [Hugging Face model repository](https://huggingface.co/GOVINDFROM/cbct-clinical-reasoner)
contains two evaluated model families. They share the same 989-statement
prototype bank and calibrated report decoder but are not interchangeable at
inference time:

| Artifact | Intended use | Runtime requirements |
|---|---|---|
| `bundle/` | Recommended ready-to-run model: a 122-feature linear predictor with a calibrated decoder and deterministic fallback | CPU; base package dependencies |
| `checkpoints_neural/fold*.pt` | Five-fold 29M-parameter encoder ensemble retained for comparison and further research; it scored below the linear bundle out of fold | PyTorch and `.[train]`; GPU recommended |
| `bundle/fallback_report.txt` | Image-independent safety output used only if volume processing or model inference fails | No learned-model runtime |

The Qwen configuration is disabled in the published experiment. No Qwen adapter
or other LLM is required to generate a report. RadFact may use an LLM only when
evaluation is explicitly run with `--radfact-lite`.

### Run the published CPU bundle

Install the source package, then download the exact Hub snapshot used below.
Pinning revision `cf0a77e14900e935ce3bb40bc6d5550bb089aeec`
prevents a later Hub update from silently changing the model:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[hub]"

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="GOVINDFROM/cbct-clinical-reasoner",
    revision="cf0a77e14900e935ce3bb40bc6d5550bb089aeec",
    allow_patterns=["bundle/*", "RESULTS.md", "results.json"],
    local_dir="artifacts/published",
)
PY

cbct-reasoner predict \
  --bundle artifacts/published/bundle \
  --input /path/to/volume.mha \
  --output predictions/diagnostic-imaging-report.json
cbct-reasoner validate-output predictions/diagnostic-imaging-report.json
```

The input must be one CBCT volume readable by SimpleITK, such as `.mha` or
`.nii.gz`. The output is the Grand Challenge JSON object
`{"report": "<non-empty report>"}`. Inference is local and does not upload the
volume. This remains benchmark research software: every generated report needs
qualified clinical review.

### Reproduce the reported experiments

The public snapshot is sufficient to reproduce inference, but not training: the
ToothFairy4 volumes and reports are access-controlled and are not redistributed
here. With an authorized copy of `toothfairy4_v03`, the linear model's
out-of-fold path can be rerun as follows:

```bash
python -m pip install -e ".[train,dev]"
export TOOTHFAIRY_DATA=/absolute/path/to/toothfairy4_v03

cbct-reasoner --config configs/toothfairy4.json prepare
cbct-reasoner --config configs/toothfairy4.json prototypes
cbct-reasoner --config configs/toothfairy4.json splits
python scripts/train_shallow.py --config configs/toothfairy4.json
cbct-reasoner --config configs/toothfairy4.json calibrate \
  --oof work/oof_shallow.npz
cbct-reasoner --config configs/toothfairy4.json evaluate \
  --oof work/oof_shallow.npz \
  --output artifacts/evaluation-shallow.json
```

For the neural comparison, replace the shallow training command with
`cbct-reasoner --config configs/toothfairy4.json train`, then run `calibrate`
and `evaluate` without `--oof`. The published run used 622 evaluated cases,
five patient-grouped folds, seed 2026, and the preprocessing/training settings
stored in `bundle/config.json`. The Hub snapshot's `results.json` records the
source commit and full metric provenance.

Reproduction means matching the data split and reported metrics within normal
numerical tolerance, not necessarily producing bit-identical neural weights.
Record the dataset release/checksum, source and Hub revisions, Python package
versions (`python -m pip freeze`), hardware, CUDA version, and RadFact model and
provider. BLEU-4 and METEOR are deterministic local implementations; the
offline RadFact surrogate is a ranking aid, while `--radfact-lite` results also
depend on the selected LLM backend.

## Quick start

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[train,hub,cloud,dev]"
cbct-reasoner doctor
```

Run the full workflow with a synthetic 40-case dataset:

```bash
cbct-reasoner synthetic --output /tmp/tf4 --cases 40
cbct-reasoner --config configs/fast.json --data /tmp/tf4 --work /tmp/ws run-all
```

Run against the real release placed under `data/raw/`:

```bash
cbct-reasoner inspect
cbct-reasoner --config configs/toothfairy4.json run-all
```

Train all 5 folds on Modal:

```bash
modal volume create cbct-toothfairy4
modal volume put cbct-toothfairy4 E:/datasets/toothfairy4 /raw
modal run modal_app/app.py --stage all --push
```

See [docs/runbook.md](docs/runbook.md) for environment setup, recovery steps,
and the complete training sequence.

## Pipeline

```text
CBCT volume
  -> orient to LPS, resample, and crop around the jaws
  -> extract geometry or learned image features
  -> predict probabilities for reportable statements
  -> calibrate per-statement decisions out of fold
  -> remove contradictions and duplicate assertions
  -> render a section-ordered clinical report
```

The production submission uses acquisition geometry for lightweight gating. It
does not need pixel decoding for those decisions. This reduces container risk
and still adapts field-of-view statements to each scan.

## Main commands

| Command | Purpose |
|---|---|
| `cbct-reasoner doctor` | Check packages, CUDA, credentials, and artifacts |
| `cbct-reasoner inspect` | Summarize the available dataset |
| `cbct-reasoner synthetic` | Create a ToothFairy4-shaped rehearsal dataset |
| `cbct-reasoner prepare` | Build the corpus and normalized voxel cache |
| `cbct-reasoner prototypes` | Construct the report statement space |
| `cbct-reasoner splits` | Build stratified or center-held-out splits |
| `cbct-reasoner train` | Train folds and write out-of-fold predictions |
| `cbct-reasoner calibrate` | Fit decoding thresholds on the target objective |
| `cbct-reasoner evaluate` | Compute captioning and clinical metrics |
| `cbct-reasoner package` | Assemble a deployable submission bundle |
| `cbct-reasoner predict --input volume.mha` | Generate one report |
| `cbct-reasoner hub push` | Upload artifacts to Hugging Face |
| `cbct-reasoner run-all` | Execute the pipeline in dependency order |

## Repository layout

```text
configs/                  full and fast experiment configurations
data/raw/                 local access-controlled release, ignored by Git
docs/                     method, strategy, runbook, and submission guidance
modal_app/app.py          parallel Modal training and evaluation app
scripts/                  analysis, selection, validation, and packaging tools
src/cbct_reasoner/        installable pipeline and command-line package
submission/               Grand Challenge image and offline entrypoint
tests/                    metrics, decoding, data, model, and end-to-end tests
```

Detailed documentation:

- [Method](docs/method.md)
- [Strategy and ablations](docs/strategy.md)
- [Data layout](docs/data.md)
- [Operations runbook](docs/runbook.md)
- [Submission guide](docs/submission.md)
- [Competition analysis](docs/competition_analysis.md)

## Validation and safety

- All reports belonging to one patient stay on one side of each split.
- The hidden test set comes from an independent center, so center-held-out
  validation is the most conservative local estimate.
- Thresholds are calibrated only on out-of-fold predictions.
- Container tests cover missing input, corrupt input, read-only execution,
  unavailable model directories, and different acquisition geometries.
- Volumes, reports, caches, and trained artifacts are excluded from Git.
- Hub repositories remain private unless publication is explicitly authorized.

## Submission

The final container runs without network access and includes layered fallbacks
so that an unusual or unreadable volume still produces a valid report. Build and
stress-test instructions are in [submission/README.md](submission/README.md).

Teams receive two final submissions. The recommended strategy is to pair the
adaptive entry with a prior-only fallback that does not require a GPU.

## Primary references

- [ODIN 2026 challenge](https://odin2026.grand-challenge.org/)
- [ToothFairy4 dataset](https://ditto.ing.unimore.it/toothfairy4/)
- [Organizer algorithm and evaluation code](https://github.com/AImageLab-zip/ToothFairy/tree/main/ODIN2026/ToothFairy4)
- [radfact-lite](https://pypi.org/project/radfact-lite/)
- [Microsoft RadFact](https://github.com/microsoft/RadFact)
- [ODIN 2026 challenge document](https://doi.org/10.5281/zenodo.19727377)

## License

The source code is available under the [MIT License](LICENSE). Dataset access,
clinical reports, model weights, third-party software, and challenge submissions
remain subject to their respective terms.
