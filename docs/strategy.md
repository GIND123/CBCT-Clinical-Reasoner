# Scoring strategy

Everything in this repository follows from one equation and two metric
implementations. This document derives the design so the choices can be argued
with rather than taken on faith.

```text
Final = 0.8 × RadFact-F1 + 0.2 × mean(BLEU-4, METEOR)
```

## What the grader actually runs

Source: [`ODIN2026/ToothFairy4/evaluation/evaluate.py`](https://github.com/AImageLab-zip/ToothFairy/blob/main/ODIN2026/ToothFairy4/evaluation/evaluate.py)
in the organizer repository, plus the `radfact-lite` package it imports.

Inside the platform the evaluator sets `RUNNING_ON_GRAND_CHALLENGE`, which routes
both captioning metrics through its **local** implementations rather than the
HuggingFace `evaluate` library:

| Metric | Implementation on the platform | Aggregation |
|---|---|---|
| BLEU-4 | NLTK `corpus_bleu`, uniform 4-gram weights, `SmoothingFunction().method1` | **Corpus level** over all 50 cases |
| METEOR | The evaluator's own `meteor_lite_score`: exact-token greedy match, no WordNet, no stemming | **Mean of per-case scores** |
| RadFact | `radfact_lite` with `report_type=TOOTHFAIRY`, `is_narrative_text=True`, `filter_negatives=False` | F1 of the *mean* precision and the *mean* recall |

[`src/cbct_reasoner/metrics/official.py`](../src/cbct_reasoner/metrics/official.py)
reimplements the first two in pure Python.
[`tests/test_official_metrics.py`](../tests/test_official_metrics.py) asserts
agreement with NLTK to machine precision over randomized corpora. That parity is
the foundation: without it, local tuning optimizes a different function than the
leaderboard.

RadFact is disabled inside the platform (`disabled_on_grand_challenge`), so the
public board shows BLEU-4 and METEOR only. The 80%-weight clinical score is
computed by the organizers offline. **A leaderboard position built purely on
BLEU/METEOR is therefore not the ranking that decides the challenge.**

## Five properties worth exploiting

### 1. METEOR is recall-weighted about 9:1

`F = 10PR / (R + 9P)`. Adding text that covers more reference tokens raises the
score even when it dilutes precision. Concretely, a report scoring P=0.5, R=1.0
beats one scoring P=1.0, R=0.5 by 0.91 to 0.53.

*Implication:* err toward longer reports for the captioning component.

### 2. The METEOR chunk penalty rewards contiguous reuse

`penalty = 0.5 × (chunks / matches)³`. Scattered single-token hits approach
`chunks = matches` and lose half the score. Reusing whole phrases as clinicians
actually wrote them keeps `chunks` low.

*Implication:* emit real corpus sentences, not paraphrases assembled word by
word. This is the main reason the decoder selects from a bank of observed
sentences instead of generating free text.

### 3. BLEU-4 punishes short reports structurally

A report under four tokens has no 4-grams, so `p₄` falls back to the smoothing
epsilon and the geometric mean collapses. Two identical three-word reports score
0.47, not 1.0 - see `test_short_identical_text_is_penalised_by_missing_4grams`.

*Implication:* there is a hard floor on useful report length, independent of
correctness.

### 4. RadFact scores phrases, and counts negatives

Precision = (my phrases entailed by the reference) / (my phrase count).
Recall = (reference phrases entailed by mine) / (reference phrase count).

Because the grader passes `filter_negatives=False`, normal and negative
statements count in both directions. "No periapical radiolucency is observed"
appears in a large share of references and is true in a large share of cases, so
emitting it is cheap recall *and* cheap precision.

*Implication:* the marginal rule for emitting a candidate phrase is
approximately **emit when P(entailed) exceeds the precision you already have**,
adjusted upward by whatever recall it adds. That threshold differs per statement
- which is exactly the parameter
[`decode/calibrate.py`](../src/cbct_reasoner/decode/calibrate.py) fits.

### 5. A missing result scores zero

The rules state that missing results are treated as a zero-character report.
One unhandled exception on one unusual volume costs 2% of the final score
outright. The inference path therefore degrades - ensemble, then single fold,
then prior-only report, then a hardcoded paragraph - and never raises.

## The resulting architecture

```text
CBCT volume
   │  orient LPS, resample to fixed mm spacing, crop around the jaws
   ▼
2.5D encoder (pretrained 2D backbone over multi-planar slices, attention-pooled)
   │  output bias initialized to the corpus log-odds of each statement
   ▼
P(statement k is reportable for this case), k = 1..K
   │  per-statement thresholds fitted on out-of-fold predictions
   ▼
selected statements → contradiction filter → section-ordered narrative
```

Why a prototype bank rather than a captioning decoder:

* Every emitted sentence is a phrasing a clinician wrote about this dataset, so
  entailment failures come from choosing the wrong finding - never from invented
  language. That protects RadFact precision structurally.
* Each cluster's representative is the member maximizing expected METEOR against
  the other members, so a correct prediction also lands the best-scoring surface
  form.
* The label space is small enough (~192 statements) that per-statement threshold
  calibration on 630 cases is statistically sound.

Why the head starts at the prior: an image-free "most likely report" is already
a strong entry on these metrics. Initializing the output bias to the corpus
log-odds means the network *starts* there and spends capacity only on learning
where a given scan deviates - rather than spending its first several epochs
rediscovering base rates.

## Calibration is the highest-leverage stage

The objective is not differentiable, so it is optimized by coordinate ascent
directly on `0.8 × clinical + 0.2 × captioning`. One evaluation over the full
corpus takes well under a second because:

* entailment between prototypes and references does not depend on thresholds and
  is precomputed once into boolean matrices;
* METEOR uses an O(P+R) matcher proven equal to the grader's O(P·R) greedy
  alignment;
* reference n-gram counts are precomputed per case.

Calibration always runs on **out-of-fold** probabilities. Fitting thresholds on
in-sample predictions yields a decoder that is confidently wrong on the hidden
test set, and on 630 cases that error is larger than any architecture change.

The fitted thresholds have an interpretable shape: common statements end up with
low thresholds, rare ones with high thresholds. If that ordering inverts,
something is wrong upstream.

## Honest accounting of what is and is not measured

`cbct-reasoner evaluate` reports a **surrogate** clinical score computed by
[`metrics/radfact.py`](../src/cbct_reasoner/metrics/radfact.py) - a lexical
entailment model over the ontology, not an LLM. It is built for ranking decoder
variants cheaply, which requires tens of thousands of evaluations.

It is not the challenge metric. Before committing a submission, run the real
thing:

```bash
# organizer's own package; keep patient reports off third-party APIs by
# pointing base_url at a local Ollama or vLLM server
pip install '.[radfact]'
cbct-reasoner evaluate --radfact-lite --radfact-provider ollama --radfact-model llama3.1
```

The surrogate deliberately hard-zeros the errors RadFact punishes - opposite
polarity, swapped laterality, wrong tooth number - so it cannot reward a model
for a class of mistake the real grader would catch. It will still disagree with
an LLM judge on borderline paraphrase, which is why it selects candidates rather
than certifies them.

## What the real release actually looks like

Measured on `toothfairy4_v03` (622 cases, 1000 English reports). Several
assumptions taken from the challenge description turned out to be wrong, and each
correction is worth recording because it changed a design decision.

### Counts

| | |
|---|---|
| Cases | **622** (the dataset page states 625; its subset counts sum to 632) |
| Centres | P 412, A 95, F 63, S 52 |
| English reports | 1000 — 255 cases with 1, 360 with 2, 3 with 3, 4 with 4 |
| Reference length | median **105** tokens, p90 211 |
| Verifiable phrases | median **8**, mean 8.9 per report |

The phrase count is the number that mattered most: a decoder configured to emit
up to 26 sentences would triple the reference length and destroy RadFact
precision for nothing. The cap was refit to 16 and the calibrated optimum sits
well below it.

### Geometry

Field of view is **much smaller and flatter** than a head CT:

| axis | p5 | median | p95 | max |
|---|---|---|---|---|
| z (axial) | 50 mm | **51 mm** | 97 mm | 120 mm |
| y | 82 mm | 103 mm | 123 mm | 160 mm |
| x | 82 mm | 111 mm | 139 mm | 160 mm |

Native spacing is 0.30 mm (p5 0.16, p95 0.30). The initial 77 x 134 x 134 mm
window was therefore ~70% air. Refitting to **56 x 96 x 96 mm at 0.5 mm** cut
padding to 38% and halved the cache. Crops also centre on the *dentition*
(a high intensity percentile, which isolates enamel and restorations) rather than
all bone, whose centroid drifts toward the skull on a large field of view.

### Language

The reports are dictated Italian translated to English. Practical consequences:

* Tooth numbers appear **bare** ("33 abutment of a prosthetic bridge; 34 absent"),
  not as "tooth 33", so FDI extraction cannot require a keyword prefix.
* Findings are chained with semicolons, and one sentence routinely carries five
  tooth-level statements.
* `mm.` ends sentences constantly; guarding it as an abbreviation merged two
  findings into one phrase that RadFact then scores all-or-nothing.
* The commonest single statement is a negative: *"No definite osteolytic or
  osteocondensing lesions."*

Extending the ontology to the vocabulary actually used - osteolytic/
osteocondensing lesions, canal course, acquisition coverage, complete dentition,
periodontitis, plural "canals" - raised concept coverage of real phrases from
**81% to 90%**.

### Which label space to use

The decisive experiment was the **oracle score**: what the pipeline would achieve
with a perfect encoder, emitting exactly the prototypes whose ground-truth label
is 1. It upper-bounds everything downstream, so it is the right way to compare
label spaces before spending GPU hours.

| label space | K | phrase coverage | **oracle final** | clinical | BLEU-4 | METEOR |
|---|---:|---:|---:|---:|---:|---:|
| tooth-masked, 512 | 507 | 84.6% | 0.5406 | 0.583 | 0.296 | 0.447 |
| tooth-aware, 512 | 512 | 82.1% | 0.5645 | 0.615 | 0.293 | 0.436 |
| **tooth-aware, 1024** | 989 | **89.0%** | **0.6008** | 0.649 | 0.341 | 0.473 |

Masking tooth numbers collapses every *"absence of teeth …"* sentence into one
cluster, and its representative then asserts a specific list of eight teeth that
is wrong for almost every case. Keeping numbers distinct costs nothing and gains
6 points of oracle score.

### The fine-tuned encoder did not work, and the measurement that showed it

The 29M-parameter `slice2d` encoder trained cleanly - five folds, loss 97 → 60,
out-of-fold coverage 1.0 - and its validation mAP even drifted upward from 0.057
to 0.072. All of that was noise. Per-prototype AUC against the prior, which
scores exactly 0.500 by construction, tells the real story:

| support band | prototypes | mean AUC |
|---|---:|---:|
| 1-5 cases | 641 | 0.477 |
| 6-20 cases | 297 | 0.484 |
| 21-60 cases | 35 | 0.474 |
| 60+ cases | 12 | 0.490 |
| **prevalence-weighted** | | **0.486** |

Every band sits at or slightly below chance, and for each prevalent statement the
mean predicted probability was the same for positives and negatives (e.g. 0.891
vs 0.909). The model memorised 497 training cases and generalised nothing.

**mAP was the wrong instrument.** Averaged over a thousand statements it is
dominated by prototypes seen two or three times, where average precision is
noise, so it barely moves whatever the model does — and it moved *upward* while
the model was getting no better. Any label space this sparse needs a
selection metric restricted to learnable columns; that is what
`EncoderConfig.min_head_support` is for.

The decisive follow-up was a control, not a bigger model: **a ten-feature
logistic regression over acquisition geometry alone**, cross-validated.

| statement | n | AUC from geometry alone |
|---|---:|---:|
| Mandibular condyles: excluded from the acquisition | 104 | 0.876 |
| Maxilla: partially included in the scan | 112 | 0.797 |
| Mandibular canal with a regular course | 104 | 0.754 |
| Mandibular CT including the mandibular body | 172 | 0.712 |
| Mandibular condyles are not included in the scan | 277 | 0.611 |

So the task *is* learnable from information the network already received on its
auxiliary input — the deep model was the problem, not the problem's difficulty.

That result motivated `models/shallow.py`: one heavily-regularised linear model
per statement over a 122-dimensional global descriptor (acquisition geometry,
coarse per-axis bone occupancy, intensity shape), fitted on the same folds.

| predictor | mean AUC | prevalence-weighted | support ≥ 12 (n=112) |
|---|---:|---:|---:|
| corpus prior | 0.500 | 0.500 | 0.500 |
| fine-tuned encoder (29M params) | 0.479 | 0.486 | 0.486 |
| **linear, 122 features** | 0.519 | **0.593** | **0.669** |

The deployable form is **52 KB** against 580 MB of fold checkpoints.

The general lesson is not "linear models are better." It is that with 622 cases
and ~1000 sparse targets, the recoverable signal is coarse and global — which jaw
is in the field of view, how much bone and where — and a model with the capacity
to memorise the training set will do exactly that. The control experiment cost
thirty seconds and was worth more than the GPU hour it audited.

### Calibrated prior as a reference point

An image-free, corpus-prior report - one shared threshold, no encoder - scores
**BLEU-4 0.1225 / METEOR 0.2568** out-of-fold. The public leaderboard's top entry
at the time of writing sits at 0.1317 / 0.3191.

That the prior lands in the same range is the most useful single observation
here: it means the visible leaderboard is not yet separating methods on imaging
ability, and that a submission has to beat a well-calibrated prior before it
beats anyone else. It is also why `ablation` scores the prior on identical
machinery - if the encoder cannot beat it out-of-fold, the prior-only bundle is
the honest submission.

## Where the remaining headroom is

Ranked by expected value, given that the pipeline above is in place:

1. **Encoder quality on the rare findings.** Common statements are handled by the
   prior; the score separation between entries comes from correctly calling
   impaction, canal proximity, sinus pathology, and bone quantity. Per-concept
   OOF average precision is the metric to watch, not overall mAP.
2. **Reference selection for multi-report cases.** 374 of 625 patients have more
   than one report. `select_reference` picks the METEOR medoid; training on the
   *union* of findings while calibrating against the medoid is the current
   split, and the alternative choices are worth an ablation.
3. **Leave-one-centre-out validation.** The hidden test set is from an
   independent centre. `--strategy center` gives the honest estimate; a
   stratified K-fold number will read higher than the leaderboard.
4. **Phase 2 fluency.** The arena is judged by surgeons comparing two reports
   side by side. The optional narrative renderer
   ([`models/llm.py`](../src/cbct_reasoner/models/llm.py)) rewrites an
   already-selected finding list into prose and is rejected automatically if it
   changes the finding set. Adopt it only if it wins on out-of-fold data.

## What this repository does not claim

It is a benchmark system, not a medical device. The generated text is a draft for
review by a qualified clinician. No component was validated for patient care, and
the ontology in `ontology.py` is a lexical mapping for supervision - not a
diagnostic rule set.
