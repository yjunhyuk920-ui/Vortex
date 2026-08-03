#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${EXP052_OUTPUT_DIR:-${ROOT}/results/exp_052_reproduction}"

if [[ "${OUTPUT_DIR}" == "${ROOT}/results/exp_052" ]]; then
  echo "refusing to overwrite frozen EXP-052 evidence" >&2
  exit 2
fi

cd "${ROOT}"
python -m pytest -q tests/exp_052
python scripts/run_validation.py
bash experiments/exp_052/run_current_env.sh "${OUTPUT_DIR}"
(
  cd "${OUTPUT_DIR}"
  sha256sum -c checksums.sha256
)
