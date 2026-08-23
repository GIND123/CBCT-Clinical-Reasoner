#!/usr/bin/env bash
# Build the Grand Challenge algorithm image.
#
# Run from the repository root. `submission/model` must contain a packaged
# bundle first:
#   cbct-reasoner package && cp -r artifacts/bundle/. submission/model/
# or, when training ran on Modal:
#   modal run modal_app/app.py --stage download-bundle --output submission/model.tar.gz
#   mkdir -p submission/model && tar -xzf submission/model.tar.gz -C submission --strip-components=0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-cbct-clinical-reasoner:latest}"

cd "$REPO_ROOT"

if [ ! -f submission/model/prototypes.json ]; then
  echo "error: submission/model/prototypes.json is missing." >&2
  echo "       Package a bundle before building; see the header of this script." >&2
  exit 1
fi

echo "Building $IMAGE_TAG"
docker build \
  --platform linux/amd64 \
  --tag "$IMAGE_TAG" \
  --file submission/Dockerfile \
  .

echo "Built $IMAGE_TAG"
