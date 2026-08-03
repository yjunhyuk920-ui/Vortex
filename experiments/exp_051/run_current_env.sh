#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${VORTEX_EXP051_OUTPUT_DIR:-$ROOT/results/exp_051_candidate}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python experiments/exp_051/run_experiment.py \
  --config experiments/exp_051/config.json \
  --output-dir "$OUTPUT_DIR"
