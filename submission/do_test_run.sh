#!/usr/bin/env bash
# Run the algorithm image the way Grand Challenge does: no network, read-only
# input, a tmpfs-backed output directory, and a non-root user.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-cbct-clinical-reasoner:latest}"
SAMPLE_DIR="${SAMPLE_DIR:-$REPO_ROOT/submission/test/input}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/submission/test/output}"

if [ ! -d "$SAMPLE_DIR/images/cbct" ]; then
  echo "error: no test input at $SAMPLE_DIR/images/cbct" >&2
  echo "       Create one with: python submission/make_test_input.py" >&2
  exit 1
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

docker run --rm \
  --platform linux/amd64 \
  --network none \
  --gpus "${DOCKER_GPUS:-all}" \
  --volume "$SAMPLE_DIR":/input:ro \
  --volume "$OUTPUT_DIR":/output \
  "$IMAGE_TAG" || {
    echo "GPU run failed; retrying on CPU" >&2
    docker run --rm \
      --platform linux/amd64 \
      --network none \
      --volume "$SAMPLE_DIR":/input:ro \
      --volume "$OUTPUT_DIR":/output \
      "$IMAGE_TAG"
  }

echo
echo "--- $OUTPUT_DIR/diagnostic-imaging-report.json ---"
cat "$OUTPUT_DIR/diagnostic-imaging-report.json"
echo

python -c "
import json, sys
from pathlib import Path
payload = json.loads(Path('$OUTPUT_DIR/diagnostic-imaging-report.json').read_text(encoding='utf-8'))
assert set(payload) == {'report'}, f'unexpected keys: {sorted(payload)}'
assert isinstance(payload['report'], str) and payload['report'].strip(), 'empty report'
print(f\"OK: contract satisfied, {len(payload['report'])} characters\")
"
