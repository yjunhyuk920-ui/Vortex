#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP068_OUTPUT_DIR:-${ROOT}/results/exp_068_candidate}"
CACHE="${EXP068_CACHE_DIR:-${ROOT}/.cache/exp_068_huggingface}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q "${ROOT}/tests/exp_068"
python "${ROOT}/experiments/exp_068/run_experiment.py" \
  --config "${ROOT}/experiments/exp_068/config.json" \
  --output-dir "${OUTPUT}" \
  --cache-dir "${CACHE}"
python "${ROOT}/scripts/run_validation.py"
