#!/usr/bin/env bash
set -euo pipefail

python_bin="${EXP081A_PYTHON:-python}"
output_dir="${EXP081A_OUTPUT_DIR:-results/exp_081a}"

"${python_bin}" experiments/exp_081a/run_experiment.py --output-dir "${output_dir}"

