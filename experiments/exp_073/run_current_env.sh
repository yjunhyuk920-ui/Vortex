#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP073_OUTPUT_DIR:-${ROOT}/results/exp_073_candidate}"
if command -v cygpath >/dev/null 2>&1 && [[ "$(python -c 'import os; print(os.pathsep)')" == ";" ]]; then
  ROOT_PY="$(cygpath -w "${ROOT}")"
  export PYTHONPATH="${ROOT_PY}${PYTHONPATH:+;${PYTHONPATH}}"
else
  export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
python -m pytest -q "${ROOT}/tests/exp_073"
python "${ROOT}/experiments/exp_073/run_stage1.py" \
  --fixture "${ROOT}/tests/exp_073/fixtures/valid_wire.txt" \
  --config "${ROOT}/experiments/exp_073/config.json" \
  --output-dir "${OUTPUT}"
