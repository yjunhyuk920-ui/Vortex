#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export EXP059_OUTPUT_DIR="${EXP059_OUTPUT_DIR:-${ROOT}/results/exp_059_candidate}"
export EXP059_CACHE_DIR="${EXP059_CACHE_DIR:-${ROOT}/.cache/exp_059_huggingface}"
bash experiments/exp_059/reproduce.sh
