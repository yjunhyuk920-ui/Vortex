#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EXP076_OUTPUT_DIR="${EXP076_OUTPUT_DIR:-${ROOT}/results/exp_076_reproduction}"
exec bash "${ROOT}/experiments/exp_076/run_current_env.sh"
