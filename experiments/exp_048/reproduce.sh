#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EXP048_OUTPUT="${EXP048_OUTPUT:-$ROOT/results/exp_048_reproduction}"

bash experiments/exp_048/run_current_env.sh
cd "$EXP048_OUTPUT"
sha256sum -c checksums.sha256
