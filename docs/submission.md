# Grand Challenge submission checklist

## Interface

| | |
|---|---|
| Input socket | `cbct-image` → `/input/images/cbct/<case>.mha` |
| Output socket | `diagnostic-imaging-report` → `/output/diagnostic-imaging-report.json` |
| Payload | exactly `{"report": "<non-empty string>"}` — no extra fields |
| Model resources | `/opt/ml/model` |
| Runtime | fully automatic, one case per invocation, **no internet** |

Implemented in [`submission/inference.py`](../submission/inference.py), mirroring
the organizer's template handler dispatch.

## Build

After a Modal run, one command does everything — pull the bundle, collect
results, build, no-network smoke test, save the tarball, push to the Hub:

```bash
bash scripts/finalize_submission.sh
```

Or step by step:

```bash
# 1. get a bundle into submission/model/
cbct-reasoner package && cp -r artifacts/bundle/. submission/model/
#    or, from Modal:
modal run modal_app/app.py --stage download-bundle --output submission/model.tar.gz
tar -xzf submission/model.tar.gz -C submission

# 2. build, test, save
python submission/make_test_input.py --case data/raw/P001
bash submission/do_build.sh
bash submission/do_test_run.sh
bash submission/do_save.sh
```

`do_test_run.sh` runs with `--network none`, as the platform does, and asserts
the output contract.

### Post-mortem: why the first submission scored 16th

It emitted the hardcoded 52-token fallback in `inference.py` for **all 50 test
cases**. The evidence is arithmetic rather than circumstantial — that paragraph
scores BLEU `0.0111 ± 0.0139` / METEOR `0.0984 ± 0.0350` against the training
references, and the leaderboard reported `0.0161 ± 0.0129` / `0.1088 ± 0.0284`.
Mean and spread both match; nothing else in the repository scores that low.

The container could not load its bundle. **Grand Challenge mounts its own model
volume at `/opt/ml/model`, which shadows whatever the image baked into that
path**, so `prototypes.json` did not exist at run time. It never reproduced
locally because nothing local shadows it — the local `docker run` test passed
every time.

The model itself was fine: the same bundle scores BLEU 0.1493 / METEOR 0.3064
out-of-fold, which would have placed near the top. The failure cost roughly
fourteen positions and none of it was modelling.

Three changes prevent a recurrence:

1. The bundle is baked to **`/opt/app/model`** as well, and `inference.py`
   searches several paths for one that actually contains `prototypes.json`
   instead of assuming a path exists.
2. The last-resort fallback is the **corpus-optimized constant report**, embedded
   in the source rather than read from disk, so a total failure still scores
   competitively instead of discarding the run.
3. The container test now mounts an **empty directory over `/opt/ml/model`** to
   reproduce the platform's shadowing before shipping.

The general lesson: a local container test that cannot reproduce the platform's
mounts is not a test of the platform. Simulate the mount.

### Verified

The full path has been exercised on a real case with a prior-only bundle: the
image builds from `submission/Dockerfile` on the organizer's
`pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime` base, runs with `--network none`,
and writes a valid `{"report": "<683 characters>"}`. The resulting image is ~12 GB
uncompressed, which is normal for that CUDA base.

### Runtime pinning

`submission/requirements.txt` and the Modal training image pin the **same**
`torch`, `timm`, `numpy`, `SimpleITK` and `scikit-learn` versions. This is not
cosmetic: a checkpoint trained against a different torch or timm build can fail
to load inside the algorithm container, and that failure surfaces only at
submission time. If you change one, change the other.

## Before uploading

- [ ] `submission/model/bundle.json` records the config and evaluation used.
- [ ] Checkpoint and image checksums recorded alongside the source commit.
- [ ] Every external dataset, pretrained model, and annotation source declared —
      the challenge forbids undisclosed private data.
- [ ] No secrets, `.env`, local paths, patient data, or debug dumps in the image.
- [ ] Image builds and runs on a second machine from the saved archive.
- [ ] Runs within the platform's memory and time limits.

## Failure modes to test explicitly

A missing result is scored as a zero-character report, so robustness is worth
real points. The container is designed to degrade rather than fail; verify it:

| Input | Expected |
|---|---|
| No volume in `/input/images/cbct` | emergency report, exit 0 |
| Corrupt or truncated file | fallback report, exit 0 |
| Multiple volumes | first is used, warning logged |
| Very small / highly anisotropic volume | valid report |
| Metal-artifact-heavy scan | valid report |
| No GPU available | valid report on CPU |
| Repeated runs | identical output |

`tests/test_pipeline_end_to_end.py` covers the fallback path and the output
contract; the rest are worth checking by hand against real cases.

## Choosing the two submissions

Teams get two final submissions and the better one is ranked. Treat them as a
hedge, not two attempts at one idea:

1. **Safe** — a `--prior-only` bundle. No GPU, no checkpoint, nothing that can
   fail on an unusual volume, and a calibrated corpus prior is genuinely
   competitive on these metrics.
2. **Best** — the full ensemble, selected on out-of-fold score computed with the
   *real* RadFact (`--radfact-lite`), not the surrogate.

If the two are within noise of each other on out-of-fold data, prefer the safe
one: variance on 50 hidden cases is large.

## Evidence packet

Archive for reproducibility and the post-challenge publication requirement:
source commit, Docker digest and archive checksum, bundle checksum, split
manifest (`work/folds.json`), training configuration, per-centre development
metrics, hardware and runtime measurements, and the data/model declaration.

The challenge requires public reproducible code for top-three methods and public
algorithms and parameters for publication eligibility.
