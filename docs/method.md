# Method

The scoring rationale behind these choices is in [strategy.md](strategy.md).
This document describes what each component does and where to change it.

## 1. Volume normalization

[`data/preprocess.py`](../src/cbct_reasoner/data/preprocess.py)

Orient to LPS, resample to a fixed millimetre spacing, then crop a fixed-size
window centred on the dense-bone centroid.

Spacing is **preserved** rather than resizing the field of view to a fixed shape.
CBCT fields of view range from a single quadrant to the full skull, and the
reports quantify real distances ("2.1 mm of residual bone above the sinus
floor"). A network trained on scale-normalized volumes cannot learn those.
Truncated fields of view are padded with air rather than rescaled.

The original geometry - shape, spacing, physical extent, foreground fraction -
travels alongside the voxels as a 10-dimensional descriptor concatenated to the
pooled image embedding, so the model can condition on acquisition differences
between centres.

Caches are `float16` `.npy` with a JSON sidecar carrying a `cache_version`.
Changing the preprocessing config invalidates them explicitly instead of silently
mixing grids.

## 2. Text to labels

[`text.py`](../src/cbct_reasoner/text.py) → [`prototypes.py`](../src/cbct_reasoner/prototypes.py)

Reports are segmented into RadFact-style verifiable phrases: one finding each,
with recommendations, speculation, and scan-quality remarks separated out (the
grader's phrase parser discards those, so keeping them would only dilute
precision).

Segmentation details that matter:

- Units are **not** treated as abbreviations. "residual bone height is 6 mm."
  ends a sentence far more often than it continues one, and merging two findings
  into one phrase makes RadFact score them as a single all-or-nothing unit.
- Decimals (`3.5 mm`) and Italian-style tooth notation (`n. 48`) are protected.
- Semicolons and finding-joining conjunctions split into separate phrases.

Phrases are then canonicalized (numbers → `#`), TF-IDF vectorized, and clustered
by agglomerative cosine linkage. Each cluster becomes a **prototype**: one
reportable statement type, represented by the member with the highest mean METEOR
against the other members.

`build_labels` produces the multi-hot target matrix. It unions findings across
*all* of a case's reports: a finding one clinician recorded is present in the
scan even when the selected reference omits it, so the imaging model should be
trained to see it.

Watch `phrase_coverage` in the `prototypes` stage output. Below ~0.85, findings
exist in the corpus that the decoder can never emit; lower
`prototypes.min_support` or raise `max_prototypes`.

## 3. Encoder

[`models/network.py`](../src/cbct_reasoner/models/network.py)

`slice2d` (default) samples slices along one or more planes, stacks neighbouring
slices as channels, runs an ImageNet-pretrained timm backbone, and pools the
sequence with gated attention MIL. With ~630 volumes and a few hundred bits of
label per case, putting most of the capacity in transferred weights beats
training 3D convolutions from scratch.

`resnet3d` is a compact from-scratch alternative kept for ensemble diversity -
its errors are decorrelated from the 2D path.

Two deliberate choices:

- **The classifier bias is initialized to the corpus log-odds** of each
  prototype, and its weights to zero. The network starts at the prior - already a
  competitive report - and spends capacity on deviations rather than
  rediscovering base rates.
- **No left-right mirroring in augmentation.** About a third of the vocabulary is
  lateralized; a mirror flip inverts the target unless every laterality label is
  swapped with it. Losing one augmentation beats teaching the model to confuse
  sides, which is also what clinicians penalise hardest in the Phase-2 arena.

Loss is asymmetric (Ridnik et al., 2021): prototype prevalence spans two orders
of magnitude, and plain BCE lets the abundant negatives of rare prototypes
dominate the gradient until the model collapses back to the prior.

## 4. Decoding and calibration

[`decode/`](../src/cbct_reasoner/decode/)

Selection is a **per-prototype threshold**, not a top-k cut, because the
break-even point differs per statement: RadFact precision divides by your own
phrase count, so an extra sentence pays only when its entailment probability
exceeds the precision you already have - while recall rewards covering
frequently-reported findings even at moderate confidence.

Thresholds are fitted by coordinate ascent directly on
`0.8 × clinical + 0.2 × captioning`, evaluated on out-of-fold probabilities. One
full-corpus evaluation runs in well under a second because entailment matrices
are precomputed, METEOR uses an O(P+R) matcher proven equal to the grader's
greedy alignment, and reference n-grams are cached.

A contradiction filter drops the weaker of any two selected statements that
assert opposite polarity on the same concept. Rendering orders sentences by
report section (technique → dentition → periapical → periodontal → mandible →
maxilla → sinus → TMJ → other → impression), which raises contiguous n-gram
overlap and reads the way a report is dictated.

`decode/mbr.py` arbitrates between candidate reports by expected metric - used
for ensembling and for choosing between the two allowed submissions.

## 5. Optional narrative renderer

[`models/llm.py`](../src/cbct_reasoner/models/llm.py)

A LoRA fine-tune that rewrites an already-selected finding list as flowing prose.
It never decides what is true. `verify_faithfulness` compares the generation's
ontology concept profile against the input's and rejects anything that adds,
drops, or flips a finding, falling back to the template. `compare_renderers`
scores template versus narrative on identical finding sets - adopt it only if it
wins out-of-fold.

Disabled by default (`llm.enabled: false`).

## 6. Inference

[`pipeline/bundle.py`](../src/cbct_reasoner/pipeline/bundle.py)

One directory holds the prototype bank, thresholds, preprocessing config, fold
checkpoints, and a precomputed fallback report. `predict` degrades in stages -
full ensemble → single fold → prior-only report → hardcoded paragraph - and never
raises, because a missing result is scored as zero characters.

## Extension points

| To change | Edit |
|---|---|
| Dataset layout | `VOLUME_CANDIDATES` / `REPORT_DIR_CANDIDATES` in `data/discovery.py` |
| Finding vocabulary | `CONCEPTS` in `ontology.py` |
| Label granularity | `PrototypeConfig` in `config.py` |
| Backbone | `EncoderConfig.backbone` / `timm_model`, or add to `build_network` |
| Objective weights | `DecodeConfig.clinical_weight` / `captioning_weight` |
| Entailment surrogate | `phrase_entailment_score` in `metrics/radfact.py` |

Anything replacing the encoder must keep the contract
`probabilities(volume) -> np.ndarray[K]`; the decoder, calibration, and container
are unchanged by it.
