#!/usr/bin/env bash
# Export the algorithm image as the compressed tarball Grand Challenge uploads.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-cbct-clinical-reasoner:latest}"
OUTPUT="${OUTPUT:-$REPO_ROOT/submission/cbct-clinical-reasoner.tar.gz}"

echo "Saving $IMAGE_TAG -> $OUTPUT"
docker save "$IMAGE_TAG" | gzip -c > "$OUTPUT"
ls -lh "$OUTPUT"
echo "Upload this file as the algorithm container image on Grand Challenge."
