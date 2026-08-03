#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${VORTEX_EXP049_REPRO_OUTPUT_DIR:-$ROOT/results/exp_049_reproduction}"

rm -rf "$OUTPUT_DIR"
VORTEX_EXP049_OUTPUT_DIR="$OUTPUT_DIR" \
  bash "$ROOT/experiments/exp_049/run_current_env.sh"

cd "$OUTPUT_DIR"
sha256sum -c checksums.sha256
