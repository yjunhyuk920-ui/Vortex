#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export EXP063_OUTPUT_DIR="${EXP063_OUTPUT_DIR:-${ROOT}/results/exp_063_candidate}"
export EXP063_CACHE_DIR="${EXP063_CACHE_DIR:-${ROOT}/.cache/exp_063_huggingface}"
bash experiments/exp_063/reproduce.sh
