# Packaged bundle

`submission/Dockerfile` copies this directory to `/opt/ml/model`. Populate it
before building:

```bash
# after a local run
cbct-reasoner package
cp -r artifacts/bundle/. submission/model/

# after training on Modal
modal run modal_app/app.py --stage download-bundle --output submission/model.tar.gz
tar -xzf submission/model.tar.gz -C submission          # extracts to submission/model
```

Expected contents:

```text
prototypes.json        sentence-prototype label space
decoder.json           calibrated per-prototype thresholds
config.json            preprocessing + training configuration
fallback_report.txt    prior-only report used if inference fails
bundle.json            provenance manifest
checkpoints/fold*.pt   encoder weights (omit for a prior-only submission)
```

Everything except `checkpoints/` is small. A prior-only bundle is a valid,
GPU-free submission and is worth keeping as the safe entry of the two the
challenge allows.

Contents are git-ignored: `prototypes.json` holds sentences copied verbatim from
ToothFairy4 clinical reports.
