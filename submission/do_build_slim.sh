#!/usr/bin/env bash
# Build the slim Grand Challenge algorithm image.
#
# The report is served from report_model.py - Python source, not a data file -
# so the container needs numpy and SimpleITK and nothing else. Building on the
# organizer's pytorch base would ship ~12 GB to serve a report that is a few
# hundred kilobytes of arithmetic.
#
# submission/model is optional here. It only matters when CBCT_USE_MODEL=1 asks
# for full image-conditioned inference; without it the geometry-gated report
# still runs, which is the point of putting the model in an import.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-cbct-clinical-reasoner-slim:latest}"

cd "$REPO_ROOT"

if [ ! -f submission/report_model.py ]; then
  echo "error: submission/report_model.py is missing." >&2
  echo "       Generate it with: python scripts/build_submission.py" >&2
  exit 1
fi

# A stale report_model.py is the quiet failure mode: the image builds, the
# container runs, and it serves last week's report. Refuse rather than ship it.
python - <<'PY'
import json
import pathlib
import sys

generated = pathlib.Path("submission/report_model.py")
artifact = pathlib.Path("artifacts/adaptive_report.json")
if not artifact.is_file():
    print("note: no artifacts/adaptive_report.json to compare against")
    sys.exit(0)

source = generated.read_text(encoding="utf-8")
fitted = json.loads(artifact.read_text(encoding="utf-8"))
for threshold in fitted["thresholds"]:
    if str(threshold) not in source:
        print(f"error: {generated} does not contain fitted threshold {threshold};")
        print("       regenerate it with python scripts/build_submission.py")
        sys.exit(1)
print(f"report_model.py matches {artifact} ({len(fitted['conditional'])} gates)")
PY

mkdir -p submission/model

echo "Building $IMAGE_TAG"
docker build \
  --platform linux/amd64 \
  --tag "$IMAGE_TAG" \
  --file submission/Dockerfile.slim \
  .

echo
docker image inspect "$IMAGE_TAG" --format 'built {{.RepoTags}} — {{div .Size 1048576}} MB'
