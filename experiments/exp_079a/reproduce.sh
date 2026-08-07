#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python -m pytest -q tests/exp_079a
python experiments/exp_079a/run_experiment.py \
  --model-dir "${1:-.deps/exp076-model}" \
  --output-dir "${2:-results/exp_079a_reproduction}"
