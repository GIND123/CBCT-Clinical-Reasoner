# CBCT Clinical Reasoner

A complete, runnable pipeline for **ODIN 2026 Task 1 (ToothFairy4)**: generating a
maxillofacial and surgical-planning report from a single 3D cone-beam CT volume.

Preprocessing, label construction, GPU training on Modal, metric-exact
evaluation, threshold calibration, Hugging Face artifact sync, and the Grand
Challenge submission container are all wired and tested. Paste the dataset and
run one command.

> Research scaffold for a benchmark. **Not a medical device.** Generated text is a
> draft for review by a qualified clinician and must not be used for patient care.

## Status

Run against the real release (`toothfairy4_v03`, **622 cases / 1000 English
reports**).

| | |
|---|---|
| Pipeline | End-to-end on the real release; also rehearsable on synthetic data (`pytest -m slow`) |
| Grader parity | BLEU-4 matches NLTK to machine precision; METEOR matches the evaluator's `meteor_lite_score` exactly |
| Label space | 989 clinician statements, **89%** of reference phrases reachable |
| Training | 5 folds in parallel on Modal (A10G), or locally |
| Artifacts | Private Hugging Face repo — bundle, checkpoints, figures, metrics |
| Submission | Algorithm image builds from the organizer's base image and passes a no-network container test |

## Results

Out-of-fold over the 622 public cases. Every predictor is decoded with the
thresholds fitted for it and scored through the same function
`cbct-reasoner evaluate` uses.

| predictor | final | clinical* | BLEU-4 | METEOR |
|---|---:|---:|---:|---:|
| corpus prior (no imaging) | **0.3575** | 0.3937 | 0.1170 | 0.3082 |
| **linear, 122 features** *(shipped)* | 0.3544 | 0.3861 | **0.1493** | 0.3064 |
| fine-tuned encoder, 29M params | 0.3403 | 0.3752 | 0.1331 | 0.2684 |
| — oracle on this label space | 0.6008 | 0.6492 | 0.3410 | 0.4726 |
| — public leaderboard #1 | — | — | 0.1317 | 0.3191 |

\* offline surrogate, **not** RadFact — see the caveat below.

BLEU-4 and METEOR are exact, so those columns compare directly with the
leaderboard: the shipped model's **0.1493 BLEU-4 exceeds the current leader's
0.1317**, with METEOR close behind (0.3064 vs 0.3191).

Two results are worth internalising:

1. **The fine-tuned encoder learned nothing.** Out-of-fold AUC 0.486
   prevalence-weighted — chance — while a ten-feature logistic regression over
   acquisition geometry alone reaches 0.61–0.88 on the same statements. It
   memorised 497 cases. The 122-feature linear model reaches AUC 0.669 on the
   learnable statements and is **52 KB** against 580 MB of checkpoints.
2. **An image-free prior is competitive on the public metrics.** So the visible
   leaderboard is not yet separating methods on imaging ability, and any entry
   must beat a well-calibrated prior before it beats anyone else.

`cbct-reasoner ablation` scores the prior on identical machinery for exactly this
reason. Full derivation in [docs/strategy.md](docs/strategy.md).

## Quick start

```bash
python -m pip install -e ".[train,hub,cloud,dev]"
cbct-reasoner doctor                 # environment + credentials check

# no dataset yet? rehearse the whole pipeline in about a minute
cbct-reasoner synthetic --output /tmp/tf4 --cases 40
cbct-reasoner --config configs/fast.json --data /tmp/tf4 --work /tmp/ws run-all

# with the real release in data/raw/
cbct-reasoner inspect
cbct-reasoner --config configs/toothfairy4.json run-all
```

Training on Modal:

```bash
modal volume create cbct-toothfairy4
modal volume put cbct-toothfairy4 E:/datasets/toothfairy4 /raw
modal run modal_app/app.py --stage all --push
```

Full instructions: **[docs/runbook.md](docs/runbook.md)**.

## How it scores

```text
Final = 0.8 × RadFact-F1 + 0.2 × mean(BLEU-4, METEOR)
```

The design follows from the grader's actual code rather than from the metric
names. Five properties drive it:

1. **METEOR weights recall ~9:1** (`F = 10PR/(R+9P)`) — coverage beats terseness.
2. **Its chunk penalty is cubic** — reusing whole clinician phrases beats
   paraphrasing.
3. **BLEU-4 is corpus-level with `method1` smoothing** — reports shorter than
   four tokens cannot score, regardless of correctness.
4. **RadFact scores phrases with `filter_negatives=False`** — commonly-true
   normal statements earn both precision and recall.
5. **A missing result scores zero** — one exception costs 2% of the final score.

Hence the architecture:

```text
CBCT volume
   │  orient LPS · resample to fixed mm spacing · crop around the jaws
   ▼
2.5D encoder — pretrained 2D backbone over multi-planar slices, attention-pooled
   │  output bias initialized to the corpus log-odds of every statement
   ▼
P(statement k is reportable), k = 1..K   (K ≈ 192 clustered clinician sentences)
   │  per-statement thresholds fitted by coordinate ascent on the real objective
   ▼
contradiction filter → section-ordered narrative report
```

Selecting from a bank of sentences clinicians actually wrote means an entailment
failure can only come from choosing the wrong finding, never from invented
language — which protects the 80%-weight clinical metric structurally. Each
cluster's representative is the member maximizing expected METEOR against the
others, so a correct prediction also lands the best-scoring surface form.

The reasoning is derived in full in **[docs/strategy.md](docs/strategy.md)**,
including where the remaining headroom is.

## Repository map

```text
configs/                 toothfairy4.json (full) and fast.json (rehearsal)
data/raw/                paste the release here — git-ignored
docs/
  strategy.md            metric derivation and design rationale
  runbook.md             dataset → training → submission, step by step
  competition_analysis.md  leaderboard snapshot, risks, ablation plan
modal_app/app.py         Modal app: one Volume, parallel fold training
src/cbct_reasoner/
  config.py              .env loading, paths, experiment configuration
  text.py                clinical sentence and phrase segmentation
  ontology.py            maxillofacial finding lexicon (FDI teeth, structures)
  prototypes.py          sentence clustering → the multi-label space
  metrics/
    official.py          bit-exact port of the grader's BLEU-4 and METEOR
    radfact.py           offline RadFact surrogate + real radfact_lite bridge
    score.py             0.8/0.2 challenge objective
  data/                  discovery · preprocessing · splits · corpus · synthetic
  models/                2.5D and 3D encoders, ASL loss, fold trainer, LLM renderer
  decode/                decoder, threshold calibration, MBR arbitration
  pipeline/              stage orchestration and the deployable bundle
  hub.py                 Hugging Face push/pull (private by default)
submission/              Grand Challenge container: Dockerfile, entrypoint, scripts
tests/                   grader parity, segmentation, decoding, full pipeline
```

## Commands

```bash
cbct-reasoner doctor            # versions, CUDA, credentials, artifact status
cbct-reasoner inspect           # what the loader found in the dataset directory
cbct-reasoner synthetic         # generate a ToothFairy4-shaped test dataset
cbct-reasoner prepare           # corpus + normalized voxel cache (resumable)
cbct-reasoner prototypes        # build the label space
cbct-reasoner splits            # --strategy stratified | center
cbct-reasoner train             # per-fold training, writes out-of-fold predictions
cbct-reasoner calibrate         # fit thresholds on the real objective
cbct-reasoner evaluate          # add --radfact-lite for the true clinical metric
cbct-reasoner package           # assemble the submission bundle
cbct-reasoner predict --input volume.mha
cbct-reasoner hub push          # sync artifacts to Hugging Face
cbct-reasoner run-all           # everything above, in order
```

## Validation discipline

- **Group by case.** 374 of 625 patients have more than one report; all of a
  patient's reports stay on one side of every split.
- **Report per centre.** The hidden test set is from an independent centre.
  `--strategy center` is the honest estimate; stratified K-fold reads higher.
- **Calibrate out-of-fold only.** Thresholds fitted on in-sample probabilities
  produce a decoder that is confidently wrong on the test set — on 630 cases that
  error dominates any architecture change.
- **Surrogate ≠ metric.** `evaluate` prints a lexical RadFact surrogate built for
  cheap ranking. Confirm with `--radfact-lite` before committing a submission.

## Data handling

The ToothFairy4 release is access-controlled patient data.

- Volumes, reports, caches, and trained artifacts are all git-ignored.
- `hub push` creates **private** repositories; `--public` warns first. The
  prototype bank contains sentences copied verbatim from clinical reports.
- `radfact_lite` can run against a local Ollama or vLLM server so report text
  never reaches a third-party API.
- Manifests record paths and counts, never report contents.

## Submitting

Teams get two final submissions and the better is ranked. Use them as a hedge:
a `--prior-only` bundle that needs no GPU and cannot fail on an unusual volume,
and the full ensemble selected on out-of-fold score. Details in
[docs/submission.md](docs/submission.md).

## Primary sources

- [ODIN 2026 challenge](https://odin2026.grand-challenge.org/)
- [ToothFairy4 dataset](https://ditto.ing.unimore.it/toothfairy4/)
- [Organizer algorithm and evaluation code](https://github.com/AImageLab-zip/ToothFairy/tree/main/ODIN2026/ToothFairy4)
- [`radfact-lite`](https://pypi.org/project/radfact-lite/) · [Microsoft RadFact](https://github.com/microsoft/RadFact)
- [ODIN 2026 challenge document, DOI 10.5281/zenodo.19727377](https://doi.org/10.5281/zenodo.19727377)

Challenge facts were verified against the organizer repository on 17 August 2026.

## License

Code is MIT ([LICENSE](LICENSE)). Dataset access, clinical reports, model
weights, third-party code, and challenge submissions remain under their own
terms.
