#!/usr/bin/env bash
# Turn a finished Modal run into an uploadable Grand Challenge algorithm.
#
#   1. pull the packaged bundle off the Modal volume
#   2. collect the run's metrics into results.json / RESULTS.md
#   3. build the algorithm image from submission/Dockerfile
#   4. run it exactly as the platform does (no network) and assert the contract
#   5. save the compressed image tarball
#   6. push bundle, checkpoints, figures and metrics to the Hugging Face repo
#
# Usage:  bash scripts/finalize_submission.sh [IMAGE_TAG]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

IMAGE_TAG="${1:-cbct-clinical-reasoner:latest}"
WORKDIR="${WORKDIR:-$REPO_ROOT/artifacts/submission_work}"

# Git Bash rewrites Modal's remote paths into Windows paths without this, and the
# Modal CLI prints U+2713 on success which cannot be encoded to a redirected
# stream under the Windows cp1252 default.
export MSYS_NO_PATHCONV=1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

mkdir -p "$WORKDIR"

echo "== 1/6 downloading the packaged bundle from Modal =="
python -m modal run modal_app/app.py --stage download-bundle \
  --output "$WORKDIR/model.tar.gz" || exit 1
rm -rf submission/model
mkdir -p submission/model
tar -xzf "$WORKDIR/model.tar.gz" -C submission || exit 1
test -f submission/model/prototypes.json || { echo "bundle missing prototypes.json" >&2; exit 1; }
echo "bundle contents:"; ls -la submission/model

echo "== 1b/6 pulling metrics and out-of-fold predictions =="
# oof.npz lets you re-decode and score locally later - in particular
# `cbct-reasoner evaluate --radfact-lite` for the real clinical metric - without
# re-running inference on Modal.
mkdir -p artifacts work
for f in calibration.json evaluation.json train_history.json ablation.json oof_reports.json; do
  python -m modal volume get --force cbct-toothfairy4 "/artifacts/$f" "artifacts/$f" 2>/dev/null     || echo "  (no $f)"
done
python -m modal volume get --force cbct-toothfairy4 /work/oof.npz work/oof.npz 2>/dev/null   || echo "  (no oof.npz)"
python -m modal volume get --force cbct-toothfairy4 /artifacts/plots artifacts/plots 2>/dev/null   || echo "  (no plots)"

echo "== 2/6 collecting results =="
python scripts/collect_results.py || echo "warning: results collection incomplete"

echo "== 3/6 building the algorithm image =="
docker build --platform linux/amd64 -t "$IMAGE_TAG" -f submission/Dockerfile . || exit 1

echo "== 4/6 container smoke test (no network, as the platform runs it) =="
CASE_DIR="$(ls -d data/raw/*/ 2>/dev/null | head -1)"
if [ -n "$CASE_DIR" ]; then
  python submission/make_test_input.py --case "${CASE_DIR%/}" --output "$WORKDIR/input"
else
  python submission/make_test_input.py --output "$WORKDIR/input"
fi
rm -rf "$WORKDIR/output"; mkdir -p "$WORKDIR/output"
docker run --rm --platform linux/amd64 --network none \
  -v "$WORKDIR/input":/input:ro -v "$WORKDIR/output":/output "$IMAGE_TAG" || exit 1
cbct-reasoner validate-output "$WORKDIR/output/diagnostic-imaging-report.json" || exit 1

echo "== 5/6 saving the image tarball =="
OUTPUT="$REPO_ROOT/submission/cbct-clinical-reasoner.tar.gz"
docker save "$IMAGE_TAG" | gzip -c > "$OUTPUT" || exit 1
ls -lh "$OUTPUT"

echo "== 6/6 pushing artifacts to Hugging Face (private) =="
cbct-reasoner hub push --no-data || echo "warning: Hub push failed"

echo
echo "SUBMISSION READY: $OUTPUT"
echo "Upload it on Grand Challenge with input socket 'cbct-image' and output"
echo "socket 'diagnostic-imaging-report'."
