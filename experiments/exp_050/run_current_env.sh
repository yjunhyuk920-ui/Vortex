#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${VORTEX_EXP050_OUTPUT_DIR:-$ROOT/results/exp_050_candidate}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python experiments/exp_050/run_experiment.py \
  --config experiments/exp_050/config.json \
  --output-dir "$OUTPUT_DIR"
