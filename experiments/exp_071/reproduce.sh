#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP071_OUTPUT_DIR:-${ROOT}/results/exp_071_candidate}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q "${ROOT}/tests/exp_071"
python "${ROOT}/experiments/exp_071/run_experiment.py" \
  --config "${ROOT}/experiments/exp_071/config.json" \
  --output-dir "${OUTPUT}"
python "${ROOT}/scripts/run_validation.py"
