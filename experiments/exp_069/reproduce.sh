#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP069_OUTPUT_DIR:-${ROOT}/results/exp_069_candidate}"
CACHE="${EXP069_CACHE_DIR:-${ROOT}/.cache/exp_069_huggingface}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q "${ROOT}/tests/exp_069"
python "${ROOT}/experiments/exp_069/run_experiment.py" \
  --config "${ROOT}/experiments/exp_069/config.json" \
  --output-dir "${OUTPUT}" \
  --cache-dir "${CACHE}"
python "${ROOT}/scripts/run_validation.py"
