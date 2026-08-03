#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${1:-${ROOT}/results/exp_054_candidate}"
cd "${ROOT}"
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python experiments/exp_054/run_experiment.py \
  --config experiments/exp_054/config.json \
  --output-dir "${OUTPUT_DIR}"
