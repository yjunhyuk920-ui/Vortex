#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP070_OUTPUT_DIR:-${ROOT}/results/exp_070_candidate}"
CACHE="${EXP070_CACHE_DIR:-${ROOT}/.cache/exp_070_huggingface}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q "${ROOT}/tests/exp_070"
python "${ROOT}/experiments/exp_070/run_experiment.py" \
  --config "${ROOT}/experiments/exp_070/config.json" \
  --output-dir "${OUTPUT}" \
  --cache-dir "${CACHE}"
python "${ROOT}/scripts/run_validation.py"
