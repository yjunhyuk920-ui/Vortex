#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP066_OUTPUT_DIR:-${ROOT}/results/exp_066_candidate}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q "${ROOT}/tests/exp_066"
python "${ROOT}/experiments/exp_066/run_from_exp065_v2.py" \
  --config "${ROOT}/experiments/exp_066/config.json" \
  --output-dir "${OUTPUT}"
python "${ROOT}/scripts/run_validation.py"
