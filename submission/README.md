# Submissions

Two uploadable algorithm images, both built from `Dockerfile.slim` (no torch, no
CUDA — the shipped decoders need only numpy and SimpleITK) and both verified by
running with `--network none`, exactly as Grand Challenge does.

| file | predictor | out-of-fold final | BLEU-4 | METEOR | size |
|---|---|---:|---:|---:|---:|
| `cbct-clinical-reasoner-shallow.tar.gz` | linear, 122 features | 0.3544 | **0.1493** | 0.3064 | 116 MB |
| `cbct-clinical-reasoner-prior.tar.gz` | corpus prior | **0.3575** | 0.1170 | 0.3082 | 116 MB |

BLEU-4 and METEOR are exact reimplementations of the grader's own local
implementations. The `final` column uses the repository's **offline RadFact
surrogate** for its clinical component and is therefore indicative, not official.

## Which to submit

Teams get two final submissions and the better is ranked. Submit both.

**Primary: `shallow`.** It is behind the prior by 0.003 on a surrogate metric and
ahead of it by 0.032 on BLEU-4, which is measured exactly. More importantly,
Phase 2 is an expert arena in which maxillofacial surgeons compare two reports
for the *same* case. A prior-only model emits the identical report for every
patient; once a clinician sees that, it loses essentially every matchup. The
linear model conditions on the scan — out-of-fold AUC 0.669 on the statements
with enough support to be learnable, against 0.500 for the prior.

**Safe: `prior`.** No model to go wrong, marginally better on the surrogate
clinical score, and it cannot fail on an unusual volume.

## Interface

| | |
|---|---|
| Input socket | `cbct-image` → `/input/images/cbct/<case>.mha` |
| Output socket | `diagnostic-imaging-report` → `/output/diagnostic-imaging-report.json` |
| Payload | exactly `{"report": "<non-empty string>"}` |
| Model resources | `/opt/ml/model` |

## Rebuilding

```bash
# after a Modal run
bash scripts/finalize_submission.sh

# or by hand, for a bundle without checkpoints
cp -r artifacts/bundle/. submission/model/
docker build --platform linux/amd64 -t cbct-clinical-reasoner:slim -f submission/Dockerfile.slim .
docker run --rm --network none -v "$PWD/test/input":/input:ro -v "$PWD/test/output":/output cbct-clinical-reasoner:slim
docker save cbct-clinical-reasoner:slim | gzip -c > submission/cbct-clinical-reasoner.tar.gz
```

Use `submission/Dockerfile` (the organizer's CUDA base, ~12 GB) only when the
bundle contains `checkpoints/*.pt`.

## Before uploading

- [ ] Run `cbct-reasoner evaluate --radfact-lite` — the clinical figure above is
      a surrogate and carries 80% of the ranking.
- [ ] Confirm the tarball loads on a second machine (`docker load < …`).
- [ ] Record the image digest and bundle checksum alongside the source commit.

## Not committed

The `.tar.gz` files and `model/` are git-ignored: they embed clinical sentences
taken verbatim from the access-controlled ToothFairy4 release.
