#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP064_OUTPUT_DIR:-${ROOT}/results/exp_064_candidate}"
CACHE="${EXP064_CACHE_DIR:-${ROOT}/.cache/exp_064_huggingface}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q "${ROOT}/tests/exp_064/test_output_row_structure.py"
python "${ROOT}/experiments/exp_064/run_experiment.py" \
  --config "${ROOT}/experiments/exp_064/config.json" \
  --output-dir "${OUTPUT}" \
  --cache-dir "${CACHE}"
python "${ROOT}/scripts/run_validation.py"
