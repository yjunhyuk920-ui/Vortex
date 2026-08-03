#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python experiments/exp_047/future_gpu_preflight.py

if [[ ! -f experiments/exp_047/run_real_model.py ]]; then
  cat >&2 <<'EOF'
EXP-047 Phase-D real-operation runner is not implemented in this Phase-A/B branch.
Preflight evidence was recorded, but Phase D remains NOT TESTED.
Do not interpret this command as a model execution or performance measurement.
EOF
  exit 3
fi

python experiments/exp_047/run_real_model.py "$@"
