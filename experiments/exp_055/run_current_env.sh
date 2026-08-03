#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${EXP055_OUTPUT_DIR:-${ROOT}/results/exp_055_candidate}"
cd "${ROOT}"
PYTHONPATH="${ROOT}" python experiments/exp_055/run_experiment.py \
  --config experiments/exp_055/config.json \
  --output-dir "${OUTPUT_DIR}"
