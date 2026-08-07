#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP072A_OUTPUT_DIR:-${ROOT}/results/exp_072a_candidate}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q "${ROOT}/tests/exp_072a"
python "${ROOT}/experiments/exp_072a/run_experiment.py" \
  --config "${ROOT}/experiments/exp_072a/config.json" \
  --output-dir "${OUTPUT}"
