#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${EXP061_OUTPUT_DIR:-${ROOT}/results/exp_061_reproduction}"
CACHE_DIR="${EXP061_CACHE_DIR:-${ROOT}/.cache/exp_061_huggingface}"
cd "${ROOT}"
python -m pytest -q tests/exp_061/test_activation_sparsity.py
PYTHONPATH="${ROOT}" python experiments/exp_061/run_experiment.py \
  --config experiments/exp_061/config.json \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}"
python scripts/run_validation.py
