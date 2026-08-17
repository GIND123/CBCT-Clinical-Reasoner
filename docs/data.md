# Data and split guide

## Released structure

The dataset portal describes one `cbct/volume.nii.gz` per case, original Italian reports
under `reports_it/`, and English reports under `reports_en/`. This code trains only on
English reports but leaves the loader isolated so bilingual objectives can be added without
changing the inference contract.

Run `cbct-reasoner index` first. It rejects cases that have a volume without English text,
or text without a volume, and writes paths/counts without copying report contents.

## Privacy and storage

- Store the downloaded release outside Git.
- Encrypt clinical data at rest where institutional policy requires it.
- Restrict manifests if case identifiers are considered sensitive locally.
- Do not publish trained retrieval artifacts without checking dataset terms: this baseline
  embeds complete selected report strings in its model file.
- Never log volume voxels or generated patient reports in hosted CI.

## Split rules

Use case ID as the grouping unit. All reports belonging to a CBCT stay with that CBCT.
Prefer leave-one-subset/center-out evaluation and record the exact case IDs. If an official
split is released, preserve it unchanged and create a nested training-only validation split.

Before training, summarize by center:

- number of cases and reports;
- voxel dimensions, spacing, physical field of view, orientation/direction;
- intensity quantiles and artifact/quality flags;
- report length, language, tooth-number frequency, and finding frequency.

The official dataset page currently has inconsistent total/subset counts. Treat the
downloaded release plus its checksum as authoritative for experiments and report both the
claimed and observed counts when discussing the discrepancy.
