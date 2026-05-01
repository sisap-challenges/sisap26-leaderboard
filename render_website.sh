#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBSITE_DIR="$SCRIPT_DIR/website"
IMAGE="sisap26-leaderboard-quarto"

echo "Building Docker image..."
docker build -t "$IMAGE" "$WEBSITE_DIR"

echo "Rendering website..."
docker run --rm \
  -v "$WEBSITE_DIR":/website \
  "$IMAGE"

echo "Done. Output is in $WEBSITE_DIR/_site/"
