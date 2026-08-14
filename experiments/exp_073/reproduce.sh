#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${EXP073_HOST:-}" ]]; then
  echo "EXP073_HOST must contain a runtime-only SSH alias; it is never serialized." >&2
  exit 2
fi
OUTPUT="${EXP073_OUTPUT_DIR:-${ROOT}/results/exp_073_reproduction}"
if command -v cygpath >/dev/null 2>&1 && [[ "$(python -c 'import os; print(os.pathsep)')" == ";" ]]; then
  ROOT_PY="$(cygpath -w "${ROOT}")"
  export PYTHONPATH="${ROOT_PY}${PYTHONPATH:+;${PYTHONPATH}}"
else
  export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
python -m pytest -q "${ROOT}/tests/exp_073"
python "${ROOT}/experiments/exp_073/run_stage1.py" \
  --host "${EXP073_HOST}" \
  --config "${ROOT}/experiments/exp_073/config.json" \
  --output-dir "${OUTPUT}"
