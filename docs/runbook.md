# Runbook

From a fresh clone to an uploaded algorithm. Steps 0-1 are the only ones needing
a decision; everything after is mechanical.

## 0. Environment

```bash
python -m pip install -e ".[train,hub,cloud,dev]"
cp .env.example .env          # already present in this working copy
cbct-reasoner doctor
```

`doctor` prints package versions, CUDA availability, whether the Hugging Face
token resolved, and which pipeline artifacts already exist. Everything below
assumes it reports `ready to prepare: True`.

## 1. Paste the dataset

Put the ToothFairy4 release in `data/raw/` (git-ignored), or point at it:

```bash
# option A - in place
data/raw/{A001,F001,P001,S001,...}/cbct/volume.nii.gz + reports_en/*.txt

# option B - anywhere, via .env
TOOTHFAIRY_DATA=E:/datasets/toothfairy4

# option C - per command
cbct-reasoner --data E:/datasets/toothfairy4 inspect
```

Then check what the loader sees **before** starting a long run:

```bash
cbct-reasoner inspect
```

```text
dataset root      : E:\datasets\toothfairy4
detected layout   : per-case directories
complete cases    : 625
cases by center   : {'A': 100, 'F': 63, 'P': 417, 'S': 52}
reports per case  : {1: 250, 2: 371, 3: 3}
```

`inspect` never raises. If it reports `MISSING REPORTS` or `MISSING VOLUMES`,
fix those cases first - `prepare` treats them as errors rather than dropping
them silently.

> The dataset page states 625 patients but its subset counts sum to 632. This
> repository does not guess which is right: it reports the observed count and
> uses that.

## 2A. Run locally

```bash
cbct-reasoner --config configs/toothfairy4.json run-all
```

Or stage by stage, which is what you want the first time:

```bash
CFG="--config configs/toothfairy4.json"
cbct-reasoner $CFG prepare        # ~1-3 h on CPU for 625 volumes; resumable
cbct-reasoner $CFG prototypes     # builds the label space; inspect the output
cbct-reasoner $CFG splits         # add --strategy center for the honest estimate
cbct-reasoner $CFG train          # needs a GPU; see 2B otherwise
cbct-reasoner $CFG calibrate
cbct-reasoner $CFG evaluate
cbct-reasoner $CFG package
```

Sanity checks worth making at each gate:

| After | Check | Expected |
|---|---|---|
| `prepare` | `failures` is empty | no unreadable volumes |
| `prototypes` | `phrase_coverage` | > 0.85; below that, findings exist the decoder can never emit |
| `prototypes` | `top_prototypes` | recognisable clinical sentences, one finding each |
| `splits` | fold sizes | balanced, and centre proportions preserved |
| `train` | `mean_val_map` | above the prior; if not, the encoder is learning nothing |
| `calibrate` | `objective.final` vs `baseline.final` | strictly higher |
| `calibrate` | threshold ordering | rare statements stricter than common ones |

### No GPU, no dataset yet?

Rehearse the entire pipeline on synthetic data in about a minute:

```bash
cbct-reasoner synthetic --output /tmp/tf4 --cases 40
cbct-reasoner --config configs/fast.json --data /tmp/tf4 --work /tmp/ws run-all
```

## 2B. Run on Modal

**Preprocess locally, upload the cache, train remotely.** This is the recommended
path and the one used for the real release: the raw release is ~45 GB, while the
normalized voxel cache is ~5 GB, so preprocessing first cuts the upload by 9x.
Preprocessing is CPU-only and takes about 30 minutes for 622 volumes; nothing is
gained by paying to do it on a GPU machine.

```bash
modal setup                                    # once
modal volume create cbct-toothfairy4

# Build the cache and label space locally (CPU)
cbct-reasoner --config configs/toothfairy4.json prepare
cbct-reasoner --config configs/toothfairy4.json prototypes
cbct-reasoner --config configs/toothfairy4.json splits

# Upload only what training needs (~5 GB, not 45 GB)
modal volume put cbct-toothfairy4 work/cache          /work/cache
modal volume put cbct-toothfairy4 work/labels.npz     /work/labels.npz
modal volume put cbct-toothfairy4 work/corpus.jsonl   /work/corpus.jsonl
modal volume put cbct-toothfairy4 work/folds.json     /work/folds.json
modal volume put cbct-toothfairy4 artifacts/prototypes.json /artifacts/prototypes.json

# GPU work: folds train in parallel
modal run modal_app/app.py --stage train
# Everything after training: calibrate, evaluate, ablation, figures, package
modal run modal_app/app.py --stage post
```

> On Git Bash, prefix Modal commands with `MSYS_NO_PATHCONV=1`. Without it the
> remote path `/work/cache` is rewritten to a Windows path and the upload lands
> somewhere unexpected.

To do everything remotely instead, upload the raw release and use `--stage all`:

```bash
modal volume put cbct-toothfairy4 E:/datasets/toothfairy4 /raw
modal run modal_app/app.py --stage inspect     # confirm the upload
modal run modal_app/app.py --stage all
```

`--stage all` runs prepare → prototypes → splits → train → collect-oof →
calibrate → evaluate → package. Folds train **in parallel** via `Function.map`,
so wall-clock is roughly one fold regardless of fold count.

Individual stages and knobs:

```bash
modal run modal_app/app.py --stage prepare
modal run modal_app/app.py --stage train --folds 0,1
modal run modal_app/app.py --stage calibrate
modal run modal_app/app.py --stage all --prior-only     # no GPU at all
modal run modal_app/app.py --stage all --push           # push to the Hub when done

CBCT_GPU=A100-40GB modal run modal_app/app.py --stage train
```

Everything lives on one Volume (`/vol/raw`, `/vol/work`, `/vol/artifacts`), so
any stage can be re-run or resumed independently. The Hugging Face token is read
from your local `.env` and forwarded as an inline Modal secret - no
`modal secret create` step.

## 3. Push to Hugging Face

```bash
cbct-reasoner hub whoami
cbct-reasoner hub push
```

Creates three **private** repositories under your namespace:

| Repo | Contents |
|---|---|
| `<ns>/cbct-clinical-reasoner` | inference bundle, checkpoints, model card |
| `<ns>/cbct-clinical-reasoner-data` | manifest, corpus, folds, labels, OOF probabilities |
| `<ns>/cbct-clinical-reasoner-runs` | calibration and evaluation JSON |

> Private is the default deliberately. `prototypes.json` contains sentences
> copied verbatim from ToothFairy4 clinical reports and the corpus contains the
> reports themselves. `--public` prints a warning; check the data-use agreement
> before using it.

## 4. Verify with the real clinical metric

The surrogate score printed by `evaluate` is a ranking tool, not the challenge
metric. Before choosing a submission:

```bash
pip install '.[radfact]'

# Local model, reports never leave the machine (recommended for patient text)
ollama serve && ollama pull llama3.1
cbct-reasoner evaluate --radfact-lite --radfact-provider ollama --radfact-model llama3.1

# Or the organizer's default backend
OPENAI_API_KEY=sk-... cbct-reasoner evaluate --radfact-lite --radfact-model gpt-4o-mini
```

## 5. Build and test the container

```bash
# local training
cbct-reasoner package && cp -r artifacts/bundle/. submission/model/

# or pull the bundle down from Modal
modal run modal_app/app.py --stage download-bundle --output submission/model.tar.gz
tar -xzf submission/model.tar.gz -C submission

python submission/make_test_input.py --case data/raw/P001
bash submission/do_build.sh
bash submission/do_test_run.sh
bash submission/do_save.sh
```

`do_test_run.sh` runs with `--network none` (as the platform does) and asserts
the output is exactly `{"report": "<non-empty string>"}`.

## 6. Submit

Upload `submission/cbct-clinical-reasoner.tar.gz` as the algorithm container on
Grand Challenge, with input socket `cbct-image` and output socket
`diagnostic-imaging-report`.

Teams get **two** final submissions and the better one is ranked. Use them as a
hedge, not as two attempts at the same idea:

1. **Safe** - `--prior-only` bundle. No GPU, no checkpoint, cannot fail on an
   unusual volume, and on these metrics a calibrated prior is genuinely
   competitive.
2. **Best** - the full ensemble, chosen on out-of-fold score with real RadFact.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No ToothFairy4 cases found` | unexpected layout | `cbct-reasoner inspect`; extend `VOLUME_CANDIDATES` in `data/discovery.py` |
| `Invalid dataset cases: X: missing English report` | incomplete case | fix or remove the case; do not skip silently |
| `Cache ... was written by version N` | preprocessing config changed | `cbct-reasoner prepare --force` |
| `No out-of-fold predictions` | `train` not run | run `train`, or `calibrate --prior-only` |
| Prototype bank has 2-sentence entries | segmentation regression | check `tests/test_text_ontology.py` |
| Calibration does not beat baseline | thresholds fitted on in-sample data | confirm `oof.npz` came from `train`, not the prior |
| Modal: `raw is empty` | volume not populated | `modal volume put cbct-toothfairy4 <local> /raw` |
