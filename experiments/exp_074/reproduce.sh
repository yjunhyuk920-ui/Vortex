#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP074_OUTPUT_DIR:-${ROOT}/results/exp_074_reproduction}"
if command -v cygpath >/dev/null 2>&1 && [[ "$(python -c 'import os; print(os.pathsep)')" == ";" ]]; then
  ROOT_PY="$(cygpath -w "${ROOT}")"
  export PYTHONPATH="${ROOT_PY}${PYTHONPATH:+;${PYTHONPATH}}"
else
  export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
python -m pytest -q "${ROOT}/tests/exp_074"
python "${ROOT}/experiments/exp_074/run_experiment.py" \
  --config "${ROOT}/experiments/exp_074/config.json" \
  --output-dir "${OUTPUT}"
