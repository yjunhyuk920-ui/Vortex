#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${EXP062_OUTPUT_DIR:-${ROOT}/results/exp_062_reproduction}"
CACHE_DIR="${EXP062_CACHE_DIR:-${ROOT}/.cache/exp_062_huggingface}"
cd "${ROOT}"
python -m pytest -q tests/exp_062/test_attention_probability_sparsity.py
PYTHONPATH="${ROOT}" python experiments/exp_062/run_experiment.py \
  --config experiments/exp_062/config.json \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}"
python scripts/run_validation.py
