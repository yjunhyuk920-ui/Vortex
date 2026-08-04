#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP067_OUTPUT_DIR:-${ROOT}/results/exp_067_candidate}"
CACHE="${EXP067_CACHE_DIR:-${ROOT}/.cache/exp_067_huggingface}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q "${ROOT}/tests/exp_067"
python "${ROOT}/experiments/exp_067/run_experiment.py" \
  --config "${ROOT}/experiments/exp_067/config.json" \
  --output-dir "${OUTPUT}" \
  --cache-dir "${CACHE}"
