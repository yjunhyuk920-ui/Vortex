#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${EXP063_OUTPUT_DIR:-${ROOT}/results/exp_063_reproduction}"
CACHE_DIR="${EXP063_CACHE_DIR:-${ROOT}/.cache/exp_063_huggingface}"
cd "${ROOT}"
python -m pytest -q tests/exp_063/test_kv_equivalence.py
PYTHONPATH="${ROOT}" python experiments/exp_063/run_experiment.py \
  --config experiments/exp_063/config.json \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}"
python scripts/run_validation.py
