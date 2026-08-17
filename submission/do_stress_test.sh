#!/usr/bin/env bash
# Run the algorithm image through every way it has failed, or could fail.
#
# A previous submission scored 0.0161 because the container could not load its
# bundle and emitted a hardcoded fallback for all 50 cases. Nothing reproduced
# locally, because the cause - Grand Challenge mounting its own volume over
# /opt/ml/model - does not happen locally unless you make it happen. So this
# script makes it happen, along with the other environment differences that
# could turn a working image into a zero.
#
# Every scenario must write a report. Scenarios that should produce a *specific*
# report check that too, because "wrote something" is exactly the check that let
# the earlier failure through.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-cbct-clinical-reasoner-slim:latest}"
WORK="$REPO_ROOT/submission/test"
REPORT="diagnostic-imaging-report.json"

pass=0
fail=0

run_case() {
  local name="$1"; shift
  local input_dir="$1"; shift
  local out="$WORK/out_${name}"

  rm -rf "$out"; mkdir -p "$out"
  local log="$WORK/log_${name}.txt"

  # --network none and a non-root user match the platform. --read-only makes the
  # root filesystem immutable, which is stricter than the platform and therefore
  # a safe thing to survive.
  docker run --rm \
    --platform linux/amd64 \
    --network none \
    --read-only \
    --tmpfs /tmp \
    --memory 8g \
    --volume "$input_dir":/input:ro \
    --volume "$out":/output \
    "$@" \
    "$IMAGE_TAG" > "$log" 2>&1
  local status=$?

  if [ ! -s "$out/$REPORT" ]; then
    echo "  FAIL  $name: no report written (exit $status)"
    tail -15 "$log" | sed 's/^/        /'
    fail=$((fail + 1))
    return 1
  fi

  local chars
  chars=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['report']))" "$out/$REPORT")
  if [ "$chars" -lt 200 ]; then
    echo "  FAIL  $name: report only $chars characters"
    fail=$((fail + 1))
    return 1
  fi
  echo "  ok    $name: $chars characters (exit $status)"
  pass=$((pass + 1))
  return 0
}

report_of() {
  python -c "import json,sys; print(json.load(open(sys.argv[1]))['report'])" "$WORK/out_$1/$REPORT"
}

echo "== building test inputs =="
python "$REPO_ROOT/submission/make_test_input.py" --output "$WORK/input_a" --shape 160 240 240 --spacing 0.3 >/dev/null
python "$REPO_ROOT/submission/make_test_input.py" --output "$WORK/input_b" --shape 400 500 500 --spacing 0.15 >/dev/null
mkdir -p "$WORK/input_empty"
mkdir -p "$WORK/input_corrupt/images/cbct" && printf 'not an image' > "$WORK/input_corrupt/images/cbct/case.mha"
mkdir -p "$WORK/empty_mount"

echo
echo "== scenarios =="
run_case normal_small  "$WORK/input_a"
run_case normal_large  "$WORK/input_b"

# The 16th-place failure, reproduced deliberately: the platform's own volume is
# mounted over /opt/ml/model, hiding whatever the image baked in there.
run_case shadowed_model "$WORK/input_a" --volume "$WORK/empty_mount":/opt/ml/model:ro

# Both model paths gone. report_model.py is a module beside inference.py, so the
# report should still be the full one rather than a degraded fallback.
run_case both_models_shadowed "$WORK/input_a" \
  --volume "$WORK/empty_mount":/opt/ml/model:ro \
  --volume "$WORK/empty_mount":/opt/app/model:ro

run_case no_input       "$WORK/input_empty"
run_case corrupt_volume "$WORK/input_corrupt"

echo
echo "== behaviour checks =="

if [ -s "$WORK/out_normal_small/$REPORT" ] && [ -s "$WORK/out_normal_large/$REPORT" ]; then
  if [ "$(report_of normal_small)" != "$(report_of normal_large)" ]; then
    echo "  ok    geometry gating: two acquisitions produce different reports"
    pass=$((pass + 1))
  else
    echo "  WARN  geometry gating: identical reports for very different volumes"
    echo "        (expected when no gate separates them; check adaptive_report.json)"
  fi
fi

if [ -s "$WORK/out_shadowed_model/$REPORT" ] && [ -s "$WORK/out_normal_small/$REPORT" ]; then
  if [ "$(report_of shadowed_model)" = "$(report_of normal_small)" ]; then
    echo "  ok    shadowed /opt/ml/model changes nothing"
    pass=$((pass + 1))
  else
    echo "  FAIL  shadowed /opt/ml/model changed the report - the 16th-place bug"
    fail=$((fail + 1))
  fi
fi

if [ -s "$WORK/out_both_models_shadowed/$REPORT" ] && [ -s "$WORK/out_normal_small/$REPORT" ]; then
  if [ "$(report_of both_models_shadowed)" = "$(report_of normal_small)" ]; then
    echo "  ok    report survives with no model directory at all"
    pass=$((pass + 1))
  else
    echo "  FAIL  losing both model paths degraded the report"
    fail=$((fail + 1))
  fi
fi

echo
echo "== $pass passed, $fail failed =="
[ "$fail" -eq 0 ]
