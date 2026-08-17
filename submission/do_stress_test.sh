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

# Under Git Bash, MSYS rewrites anything that looks like an absolute Unix path
# before the process sees it - so "/output" arrives as a Windows directory and
# docker reports 'invalid mount path: C'. Disable that, and convert the host
# side explicitly, since the daemon wants a native path there.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

hostpath() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi
}

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
    --volume "$(hostpath "$input_dir")":/input:ro \
    --volume "$(hostpath "$out")":/output \
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
  chars=$(read_len "$out/$REPORT")
  if [ "$chars" -lt 200 ]; then
    echo "  FAIL  $name: report only $chars characters"
    fail=$((fail + 1))
    return 1
  fi
  echo "  ok    $name: $chars characters (exit $status)"
  pass=$((pass + 1))
  return 0
}

# Windows Python cannot open the /e/... paths Git Bash produces, so every path
# handed to it is converted first. Without this the readers returned empty
# strings, two empty strings compared equal, and the behaviour checks reported
# "ok" for runs that had written nothing - the same false pass that hid the
# original failure.
read_report() {
  python -c "import json,sys; print(json.load(open(sys.argv[1]))['report'])" "$(hostpath "$1")"
}

read_len() {
  python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['report']))" "$(hostpath "$1")"
}

report_of() {
  local text
  text=$(read_report "$WORK/out_$1/$REPORT") || return 1
  [ -n "$text" ] || return 1
  printf '%s' "$text"
}

compare() {
  local name="$1" a="$2" b="$3" expect="$4" message="$5"
  local left right
  if ! left=$(report_of "$a") || ! right=$(report_of "$b"); then
    echo "  FAIL  $name: could not read one of the reports"
    fail=$((fail + 1))
    return 1
  fi
  if { [ "$expect" = "same" ] && [ "$left" = "$right" ]; } ||
     { [ "$expect" = "differ" ] && [ "$left" != "$right" ]; }; then
    echo "  ok    $message"
    pass=$((pass + 1))
    return 0
  fi
  echo "  FAIL  $name: $message did not hold"
  fail=$((fail + 1))
  return 1
}

echo "== building test inputs =="
MAKE_INPUT="$(hostpath "$REPO_ROOT/submission/make_test_input.py")"
python "$MAKE_INPUT" --output "$(hostpath "$WORK/input_a")" --shape 160 240 240 --spacing 0.3 >/dev/null
python "$MAKE_INPUT" --output "$(hostpath "$WORK/input_b")" --shape 400 500 500 --spacing 0.15 >/dev/null
mkdir -p "$WORK/input_empty"
mkdir -p "$WORK/input_corrupt/images/cbct" && printf 'not an image' > "$WORK/input_corrupt/images/cbct/case.mha"
mkdir -p "$WORK/empty_mount"

echo
echo "== scenarios =="
run_case normal_small  "$WORK/input_a"
run_case normal_large  "$WORK/input_b"

# The 16th-place failure, reproduced deliberately: the platform's own volume is
# mounted over /opt/ml/model, hiding whatever the image baked in there.
run_case shadowed_model "$WORK/input_a" --volume "$(hostpath "$WORK/empty_mount")":/opt/ml/model:ro

# Both model paths gone. report_model.py is a module beside inference.py, so the
# report should still be the full one rather than a degraded fallback.
run_case both_models_shadowed "$WORK/input_a" \
  --volume "$(hostpath "$WORK/empty_mount")":/opt/ml/model:ro \
  --volume "$(hostpath "$WORK/empty_mount")":/opt/app/model:ro

run_case no_input       "$WORK/input_empty"
run_case corrupt_volume "$WORK/input_corrupt"

echo
echo "== behaviour checks =="

compare gating normal_small normal_large differ   "geometry gating: two acquisitions produce different reports"

compare shadow shadowed_model normal_small same   "shadowed /opt/ml/model changes nothing"

compare shadow_both both_models_shadowed normal_small same   "report survives with no model directory at all"

echo
echo "== $pass passed, $fail failed =="
[ "$fail" -eq 0 ]
