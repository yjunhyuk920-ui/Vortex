#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q tests/exp_047
python experiments/exp_047/run_experiment.py \
  --config experiments/exp_047/config.json
