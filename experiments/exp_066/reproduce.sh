#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP066_OUTPUT_DIR:-${ROOT}/results/exp_066_candidate}"
CACHE="${EXP066_CACHE_DIR:-${ROOT}/.cache/exp_066_huggingface}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q "${ROOT}/tests/exp_066"
python "${ROOT}/experiments/exp_066/run_threshold_experiment.py" \
  --config "${ROOT}/experiments/exp_066/config.json" \
  --output-dir "${OUTPUT}" \
  --cache-dir "${CACHE}"
python "${ROOT}/scripts/run_validation.py"
