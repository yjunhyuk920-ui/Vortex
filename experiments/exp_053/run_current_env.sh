#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${1:-${ROOT}/results/exp_053_candidate}"

cd "${ROOT}"
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python experiments/exp_053/run_experiment.py \
  --config experiments/exp_053/config.json \
  --output-dir "${OUTPUT_DIR}"
