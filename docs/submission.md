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
