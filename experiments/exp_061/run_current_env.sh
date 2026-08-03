#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export EXP061_OUTPUT_DIR="${EXP061_OUTPUT_DIR:-${ROOT}/results/exp_061_candidate}"
export EXP061_CACHE_DIR="${EXP061_CACHE_DIR:-${ROOT}/.cache/exp_061_huggingface}"
bash experiments/exp_061/reproduce.sh
