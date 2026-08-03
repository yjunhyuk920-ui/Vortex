#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${EXP053_OUTPUT_DIR:-${ROOT}/results/exp_053_reproduction}"

if [[ "${OUTPUT_DIR}" == "${ROOT}/results/exp_053" ]]; then
  echo "refusing to overwrite frozen EXP-053 evidence" >&2
  exit 2
fi

cd "${ROOT}"
python -m pytest -q tests/exp_053
python scripts/run_validation.py
bash experiments/exp_053/run_current_env.sh "${OUTPUT_DIR}"
(
  cd "${OUTPUT_DIR}"
  sha256sum -c checksums.sha256
)
