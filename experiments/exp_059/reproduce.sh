#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${EXP059_OUTPUT_DIR:-${ROOT}/results/exp_059_reproduction}"
CACHE_DIR="${EXP059_CACHE_DIR:-${ROOT}/.cache/exp_059_huggingface}"
cd "${ROOT}"
python -m pytest -q tests/exp_059/test_displacement_rank.py
PYTHONPATH="${ROOT}" python experiments/exp_059/run_experiment.py \
  --config experiments/exp_059/config.json \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}"
python scripts/run_validation.py
