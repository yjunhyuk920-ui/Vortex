#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${EXP056_OUTPUT_DIR:-${ROOT}/results/exp_056_reproduction}"
cd "${ROOT}"
python -m pytest -q tests/exp_056/test_prototype_residual.py
PYTHONPATH="${ROOT}" python experiments/exp_056/run_experiment.py \
  --config experiments/exp_056/config.json \
  --output-dir "${OUTPUT_DIR}"
python scripts/run_validation.py
