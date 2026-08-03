#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${VORTEX_EXP049_OUTPUT_DIR:-$ROOT/results/exp_049_candidate}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python experiments/exp_049/run_experiment.py \
  --config experiments/exp_049/config.json \
  --output-dir "$OUTPUT_DIR"
