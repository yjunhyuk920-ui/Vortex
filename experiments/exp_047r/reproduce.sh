#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EXP047R_OUTPUT="${EXP047R_OUTPUT:-$ROOT/results/exp_047r_reproduction}"

bash experiments/exp_047r/run_current_env.sh

cd "$EXP047R_OUTPUT"
sha256sum -c checksums.sha256
