#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${1:-${ROOT}/results/exp_052_candidate}"

cd "${ROOT}"
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python experiments/exp_052/run_experiment.py \
  --config experiments/exp_052/config.json \
  --output-dir "${OUTPUT_DIR}"
