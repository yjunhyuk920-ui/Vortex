#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${EXP055_OUTPUT_DIR:-${ROOT}/results/exp_055_reproduction}"
cd "${ROOT}"
python -m pytest -q tests/exp_055/test_column_signature.py
PYTHONPATH="${ROOT}" python experiments/exp_055/run_experiment.py \
  --config experiments/exp_055/config.json \
  --output-dir "${OUTPUT_DIR}"
python scripts/run_validation.py
