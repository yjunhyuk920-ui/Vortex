#!/usr/bin/env bash
set -euo pipefail

python_bin="${EXP080A_PYTHON:-python}"
output_dir="${EXP080A_OUTPUT_DIR:-results/exp_080a}"

"${python_bin}" experiments/exp_080a/run_experiment.py \
  --output-dir "${output_dir}"
