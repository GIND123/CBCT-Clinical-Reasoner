# Paste the ToothFairy4 release here

This directory is git-ignored. Nothing patient-derived is ever committed.

## Expected layout

```text
data/raw/
├── A001/
│   ├── cbct/volume.nii.gz
│   ├── reports_en/report_1.txt
│   ├── reports_en/report_2.txt
│   └── reports_it/...          # present but unused
├── F001/
├── P001/
└── S001/
```

The loader also accepts an extra wrapper directory (`data/raw/toothfairy4/A001/...`),
`.mha`/`.nrrd` volumes instead of `.nii.gz`, and a flat
`volumes/` + `reports_en/` arrangement. It does **not** accept a case with a
volume but no English report, or the reverse - those are errors, not warnings,
because silently dropping cases is how training sets shrink unnoticed.

## After pasting

```bash
cbct-reasoner inspect          # what was found, and what is wrong
cbct-reasoner run-all          # the whole pipeline
```

If the layout differs from all of the above, `cbct-reasoner inspect` prints what
it detected; adjust `VOLUME_CANDIDATES` / `REPORT_DIR_CANDIDATES` in
[`src/cbct_reasoner/data/discovery.py`](../../src/cbct_reasoner/data/discovery.py).

## Data handling

The release is access-controlled and contains clinical reports. Keep it out of
version control, out of public Hugging Face repositories, and out of hosted CI
logs. `cbct-reasoner hub push` defaults to private repositories for this reason.
