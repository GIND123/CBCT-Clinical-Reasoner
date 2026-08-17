# Data and split guide

## Released structure

Each case is `{CASE_ID}/cbct/volume.nii.gz` with sibling `reports_it/` (original
Italian) and `reports_en/` (English translations). This pipeline trains on the
English reports; the loader keeps the language choice isolated so a bilingual
objective can be added without touching the inference contract.

The loader also accepts an extra wrapper directory, `.mha`/`.nrrd` volumes, and a
flat `volumes/` + `reports_en/` arrangement. It does **not** accept a case with a
volume but no report, or the reverse — those raise, because silently dropping
cases is how training sets shrink unnoticed.

Run `cbct-reasoner inspect` immediately after extracting the archive. It never
raises and reports exactly what it found:

```text
detected layout   : per-case directories
complete cases    : 625
cases by center   : {'A': 100, 'F': 63, 'P': 417, 'S': 52}
reports per case  : {1: 250, 2: 371, 3: 3}
```

## Known count discrepancy

The dataset page states 625 patients but its subset counts sum to
417 + 63 + 52 + 100 = 632. This repository does not guess which is correct: it
counts the actual case directories, records the observed number in
`work/manifest.jsonl`, and reports both when discussing results.

## Multiple reports per case

374 of 625 patients have more than one report, but the hidden test set supplies
exactly one reference per case. That asymmetry is handled deliberately:

- **Reference for scoring** — `select_reference` picks the METEOR medoid, the
  report a grader would score highest on average against the alternatives.
- **Targets for training** — `build_labels` unions the findings across *every*
  report for the case. A finding one clinician recorded is present in the scan
  even when the selected reference omits it, so the imaging model should be
  trained to see it.

Both choices are worth an ablation once real data is available; they are the two
largest text-side decisions in the pipeline.

## Split rules

Case ID is the grouping unit — all of a patient's reports stay on one side of
every split. The pipeline enforces this by splitting on case, never on report.

| Strategy | Command | Use for |
|---|---|---|
| Stratified group K-fold | `cbct-reasoner splits` | model selection; centre proportions preserved across folds |
| Leave-one-centre-out | `cbct-reasoner splits --strategy center` | the honest external-validation estimate |

The hidden test set comes from an independent centre, so a stratified K-fold
number measures in-domain interpolation and will read **higher** than the
leaderboard. Quote both.

If an official split is ever released, preserve it unchanged and nest a
training-only validation split inside it.

## Before training, profile by centre

- case and report counts;
- voxel dimensions, spacing, physical field of view, orientation;
- intensity quantiles and artifact/quality flags;
- report length, tooth-number frequency, and finding frequency.

The `prepare` stage writes the geometry half of this into the cache sidecars, and
`prototypes` reports finding frequencies. Large per-centre differences in field of
view or intensity range are the first thing to check when leave-one-centre-out
scores drop.

## Privacy and storage

- Keep the release outside version control (`data/raw/` is git-ignored).
- Encrypt at rest where institutional policy requires it.
- Never log voxels or report text in hosted CI.
- `work/` holds derived clinical data — voxel caches and the report corpus — and
  is git-ignored for the same reason as the raw release.
- `artifacts/prototypes.json` contains sentences copied verbatim from clinical
  reports. `cbct-reasoner hub push` therefore creates private repositories by
  default; check the ToothFairy4 data-use agreement before `--public`.
