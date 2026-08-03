#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${EXP056_OUTPUT_DIR:-${ROOT}/results/exp_056_candidate}"
cd "${ROOT}"
PYTHONPATH="${ROOT}" python experiments/exp_056/run_experiment.py \
  --config experiments/exp_056/config.json \
  --output-dir "${OUTPUT_DIR}"
