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

if [[ "${1:-}" == "--serve" || "${1:-}" == "-s" ]]; then
  PORT="${2:-8000}"
  echo "Serving on http://localhost:$PORT ..."
  cd "$WEBSITE_DIR/_site" && python3 -m http.server "$PORT"
fi
