#!/usr/bin/env bash
set -euo pipefail

python_bin="${EXP082A_PYTHON:-python}"
output_dir="${EXP082A_OUTPUT_DIR:-results/exp_082a}"

"${python_bin}" experiments/exp_082a/run_experiment.py --output-dir "${output_dir}"
