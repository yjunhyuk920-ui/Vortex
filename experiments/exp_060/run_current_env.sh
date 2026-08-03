#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export EXP060_OUTPUT_DIR="${EXP060_OUTPUT_DIR:-${ROOT}/results/exp_060_candidate}"
export EXP060_CACHE_DIR="${EXP060_CACHE_DIR:-${ROOT}/.cache/exp_060_huggingface}"
bash experiments/exp_060/reproduce.sh
