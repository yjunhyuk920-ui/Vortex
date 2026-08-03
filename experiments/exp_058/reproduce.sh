#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${EXP058_OUTPUT_DIR:-${ROOT}/results/exp_058_reproduction}"
CACHE_DIR="${EXP058_CACHE_DIR:-${ROOT}/.cache/exp_058_huggingface}"
cd "${ROOT}"
python -m pytest -q tests/exp_058/test_modular_rank.py
PYTHONPATH="${ROOT}" python experiments/exp_058/run_experiment.py \
  --config experiments/exp_058/config.json \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}"
python scripts/run_validation.py
