#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP075_OUTPUT_DIR:-${ROOT}/results/exp_075_candidate}"
SOURCE_ARGS=()
if [[ -n "${EXP075_SOURCE_DIR:-}" ]]; then
  SOURCE_ARGS=(--source-dir "${EXP075_SOURCE_DIR}")
fi
if command -v cygpath >/dev/null 2>&1 && [[ "$(python -c 'import os; print(os.pathsep)')" == ";" ]]; then
  ROOT_PY="$(cygpath -w "${ROOT}")"
  export PYTHONPATH="${ROOT_PY}${PYTHONPATH:+;${PYTHONPATH}}"
else
  export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
python -m pytest -q "${ROOT}/tests/exp_075"
python "${ROOT}/experiments/exp_075/run_experiment.py" \
  --config "${ROOT}/experiments/exp_075/config.json" \
  --output-dir "${OUTPUT}" \
  "${SOURCE_ARGS[@]}"
