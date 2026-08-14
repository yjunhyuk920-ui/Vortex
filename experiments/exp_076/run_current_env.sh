#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${EXP076_PYTHON:-python}"
MODEL_DIR="${EXP076_MODEL_DIR:-}"
OUTPUT="${EXP076_OUTPUT_DIR:-${ROOT}/results/exp_076_candidate}"
if [[ -z "${MODEL_DIR}" ]]; then
  echo "EXP076_MODEL_DIR must point to the verified pinned checkpoint" >&2
  exit 2
fi
if command -v cygpath >/dev/null 2>&1 && [[ "$("${PYTHON_BIN}" -c 'import os; print(os.pathsep)')" == ";" ]]; then
  ROOT_PY="$(cygpath -w "${ROOT}")"
  export PYTHONPATH="${ROOT_PY}${PYTHONPATH:+;${PYTHONPATH}}"
else
  export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
"${PYTHON_BIN}" -m pytest -q "${ROOT}/tests/exp_076"
"${PYTHON_BIN}" "${ROOT}/experiments/exp_076/run_experiment.py" \
  --config "${ROOT}/experiments/exp_076/config.json" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${OUTPUT}"
