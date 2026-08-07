#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EXP075_OUTPUT_DIR="${EXP075_OUTPUT_DIR:-${ROOT}/results/exp_075_reproduction}"
bash "${ROOT}/experiments/exp_075/run_current_env.sh"
