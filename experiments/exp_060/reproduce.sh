#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${EXP060_OUTPUT_DIR:-${ROOT}/results/exp_060_reproduction}"
CACHE_DIR="${EXP060_CACHE_DIR:-${ROOT}/.cache/exp_060_huggingface}"
cd "${ROOT}"
python -m pytest -q tests/exp_060/test_sparse_streaming.py
PYTHONPATH="${ROOT}" python experiments/exp_060/run_experiment.py \
  --config experiments/exp_060/config.json \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}"
python scripts/run_validation.py
