#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export EXP062_OUTPUT_DIR="${EXP062_OUTPUT_DIR:-${ROOT}/results/exp_062_candidate}"
export EXP062_CACHE_DIR="${EXP062_CACHE_DIR:-${ROOT}/.cache/exp_062_huggingface}"
bash experiments/exp_062/reproduce.sh
