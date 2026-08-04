#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP065_OUTPUT_DIR:-${ROOT}/results/exp_065_candidate}"
CACHE="${EXP065_CACHE_DIR:-${ROOT}/.cache/exp_065_huggingface}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q "${ROOT}/tests/exp_065/test_kronecker_rank.py"
python "${ROOT}/experiments/exp_065/run_experiment.py" \
  --config "${ROOT}/experiments/exp_065/config.json" \
  --output-dir "${OUTPUT}" \
  --cache-dir "${CACHE}"
python "${ROOT}/scripts/run_validation.py"
