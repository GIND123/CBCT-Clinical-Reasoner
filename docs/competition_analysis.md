# ToothFairy4 competition analysis

Analysis date: 16 August 2026. Facts are sourced from the official challenge, dataset,
submission, workshop, and organizer-repository pages linked in the root README. The
leaderboard section analyzes the user-supplied snapshot; it does not reproduce participant
or team identities.

## 1. What is actually being optimized

ToothFairy4 is not “write a plausible dental report.” It is a coupled perception,
reasoning, and language task under domain shift:

```text
3D evidence -> localized anatomy/findings -> clinical relations -> concise report
```

A system can fail at any boundary. It may miss a tooth, assign the wrong FDI number,
detect a lesion but reverse laterality, estimate anatomy correctly but omit it from the
report, or produce fluent unsupported text. A single end-to-end score hides those failure
modes, so internal evaluation must decompose them.

The hidden set contains 50 cases from an independent center. That makes scanner style,
voxel spacing, reconstruction kernel, field of view, population, and reporting style
plausible nuisance variables. Center recognition is not clinical reasoning; a competitive
model must learn anatomy and findings that survive those changes.

## 2. Dataset implications

The official page states 625 cases and 1,001 reports in each of Italian and English. It
reports 250 patients with one report, 371 with two, and 3 with three. Those patient counts
sum to 624, while the published subset counts P=417, F=63, S=52, and A=100 sum to 632.
These inconsistencies should be resolved against the downloaded release and organizer
metadata before a paper reports sample counts.

P, F, and S derive from ToothFairy3. A may have non-isotropic spacing, whereas the page
states 0.3 mm isotropic spacing for the other sets. Consequences:

- All measurements must use physical coordinates, not voxel counts.
- Reusing ToothFairy3 labels requires an explicit orientation audit.
- Splits must group by patient/case, because multiple reports are alternative references
  for the same image, not independent samples.
- Every reference should contribute to training. For validation, score against all
  references if the official protocol permits it; otherwise mirror its chosen-reference
  behavior exactly.
- English translations may carry systematic phrasing. Exploit the report ontology and
  style consistently, but avoid learning translation artifacts as if they were anatomy.

The new A subset is the most obvious public proxy for domain shift. Report both train-on-
PFS/test-on-A and reciprocal center holdouts. Do not tune exclusively on A and then call
it an untouched external test.

## 3. Metric interpretation

The organizer repository currently implements BLEU-4 and METEOR and an optional RadFact
path with logical precision, recall, and F1. Challenge materials also describe blinded
expert review for top methods.

| Signal | Rewards | Main blind spot |
|---|---|---|
| BLEU-4 | Exact local phrase overlap | Paraphrases and clinical entailment |
| METEOR | Token alignment with some linguistic tolerance | Anatomical truth and severity |
| RadFact | Entailment-oriented factual overlap | Parser/model sensitivity and cost |
| Expert review | Usefulness, errors, and safety | Limited scale and reviewer variation |
| Mean position | Robust rank aggregation across metrics | Hides absolute gaps and uncertainty |

Optimization should be Pareto-aware. Template memorization can raise lexical scores but
miss rare critical findings. Verbose generation can increase recall while damaging factual
precision. The safest strategy predicts structured facts first, calibrates their inclusion,
then realizes concise prose in the reference style.

Recommended internal dashboard:

- Official-version BLEU-4 and METEOR.
- RadFact logical precision, recall, F1, and failure count.
- Entity/relation F1 for tooth number, finding, side, severity, and anatomical relation.
- Critical-error rate: wrong tooth, wrong side, false lesion, missed canal/sinus risk,
  fabricated measurement, and negation inversion.
- Report length, empty-output rate, runtime, peak RAM/VRAM, and failure rate by center.
- Bootstrap confidence intervals by case, stratified by center.

## 4. Leaderboard reading

The snapshot has 10 entries, mean positions from 1.5 to 10.0, and two tied pairs. The
median mean position is 5.25 and the arithmetic mean is 5.55; neither is a performance
score. Six labels are explicitly baseline-like, including two retrieval labels. This
suggests an early or low-information leaderboard in which robust priors and report style
matter substantially.

The third-place retrieval label is strategically important: every advanced model should
be compared against image-agnostic frequent-report, center-prior, global-feature retrieval,
and learned-image retrieval baselines. If a complex generator cannot beat these on held-out
centers and factual metrics, its apparent language quality is not evidence of image use.

No architecture should be inferred from an algorithm title. “Stage Seg GPU” may indicate
segmentation, but the public label does not establish which anatomy is segmented or how text
is produced. “SINUS” likewise does not prove a sinus-specific model. Method claims require
released descriptions or code.

## 5. Recommended model stack

### Stage A — canonicalization and quality control

- Read orientation, origin, direction, spacing, and field of view.
- Convert to a tested canonical physical orientation.
- Resample only where needed; retain a map back to native coordinates.
- Normalize CBCT intensity robustly because CBCT values are not reliable universal HU.
- Flag truncation, metal artifacts, motion, and incomplete anatomy as explicit quality
  features rather than silently treating them as normal anatomy.

### Stage B — anatomy and finding extraction

Start with multi-class 3D segmentation or detection for teeth, jaw bones, mandibular
canals, maxillary sinuses, and other ontology targets. Use multi-scale crops so a coarse
full-volume encoder preserves context while tooth-level and risk-structure encoders retain
detail. Predict tooth labels in a topology-aware head; unconstrained per-instance FDI
classification is prone to numbering shifts when teeth are absent.

### Stage C — structured clinical state

Represent every assertion as data before prose, for example:

```json
{
  "subject": "tooth_48",
  "finding": "impacted",
  "relation": "close_to",
  "object": "right_mandibular_canal",
  "certainty": 0.91,
  "evidence": {"crop": "...", "measurement_mm": 1.8}
}
```

Use a constrained ontology with explicit `present`, `absent`, `uncertain`, and
`not_assessable` states. Predict relations, not only entities: surgical utility often
depends on distance and topology.

### Stage D — report realization

Train a compact decoder or deterministic renderer from structured facts to the released
English style. Constrain tooth identifiers, laterality, numbers, and negation to the fact
graph. Retrieval-augmented phrasing can improve style, but retrieved facts must never enter
the output unless supported by the current case.

### Stage E — ensemble and calibration

Average segmentation/detection probabilities across folds and test-time views. Merge
structured facts using calibrated confidence thresholds by finding type. High-risk false
positives and high-risk omissions may require different operating points. Render prose
once from the merged state; ensembling free text is hard to audit.

## 6. Validation protocol

1. Freeze a case manifest with center, spacing, dimensions, report count, and checksums.
2. Hold out each center/subset in turn. Never separate reports from the same case.
3. Fit preprocessing statistics, retrieval indices, tokenizers, and templates on training
   folds only.
4. Select thresholds on a nested validation portion, not the reported holdout.
5. Score every case and retain structured predictions for error analysis.
6. Bootstrap case-level confidence intervals and report center-specific metrics.
7. Ask clinicians to review a balanced error set, including metric disagreements and rare
   findings.
8. Run the final container offline on unseen synthetic and organizer-format samples.

Suggested ablations:

| Ablation | Question |
|---|---|
| No image / most frequent report | Does the model use the scan? |
| Global retrieval baseline | Does learned anatomy beat scan-level similarity? |
| No segmentation supervision | How much do prior ToothFairy labels help? |
| No ontology bottleneck | Does direct text decoding hallucinate more? |
| Random split vs center holdout | How large is domain-shift optimism? |
| Single vs multiple references | How much does report variation affect scores? |
| No constrained decoding | Do tooth/side/number errors rise? |
| No calibration | What is the factual precision-recall tradeoff? |

## 7. Priority risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Cross-center intensity/spacing shift | High | High | Physical normalization and center holdouts |
| Fluent unsupported findings | High | High | Ontology bottleneck and evidence gating |
| Wrong tooth number or side | Medium | High | Topology constraints and structured decoder |
| Missed rare critical anatomy | Medium | High | Targeted sampling and recall calibration |
| Multiple-report leakage | Medium | High | Case-grouped manifests and split assertions |
| Metric overfitting | High | Medium | RadFact, entity checks, and clinician review |
| Container/runtime failure | Medium | High | Offline deterministic smoke test and limits |
| External-model licensing issue | Medium | High | Model/data provenance ledger before training |
| Published count inconsistency | High | Medium | Release checksum and observed manifest counts |

## 8. Two-submission strategy

Because the policy allows two final submissions and uses the better one, submit models
that differ meaningfully:

- Submission A: precision-oriented, ontology-constrained, conservative inclusion of
  uncertain findings.
- Submission B: recall-oriented thresholds or a genuinely different anatomy ensemble.

Two seeds of the same decoder provide little risk diversification. Freeze both images,
their model hashes, expected runtime, and local metrics before upload. Do not use the test
leaderboard as a high-frequency hyperparameter loop.

## 9. Definition of a competition-ready system

A candidate is ready only when it produces the exact output schema for every case, runs
without network access or human interaction, declares all external resources, passes
held-out-center evaluation, has no known systematic side/tooth swaps, fits platform time
and memory limits, and can be reconstructed from a commit plus checksums. A high lexical
score alone does not satisfy this bar.
