#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${EXP057_OUTPUT_DIR:-${ROOT}/results/exp_057_reproduction}"
CACHE_DIR="${EXP057_CACHE_DIR:-${ROOT}/.cache/exp_057_huggingface}"
cd "${ROOT}"
python -m pytest -q tests/exp_057/test_weight_structure.py
PYTHONPATH="${ROOT}" python experiments/exp_057/run_experiment.py \
  --config experiments/exp_057/config.json \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}"
python scripts/run_validation.py
