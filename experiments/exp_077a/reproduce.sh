#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EXP077A_OUTPUT_DIR="${EXP077A_OUTPUT_DIR:-${ROOT}/results/exp_077a_reproduction}"
exec bash "${ROOT}/experiments/exp_077a/run_current_env.sh"
