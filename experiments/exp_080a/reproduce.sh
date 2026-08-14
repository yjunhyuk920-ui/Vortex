#!/usr/bin/env bash
set -euo pipefail

python_bin="${EXP080A_PYTHON:-python}"
output_dir="${EXP080A_OUTPUT_DIR:-results/exp_080a_reproduction}"

if [[ -e "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to overwrite nonempty output: ${output_dir}" >&2
  exit 1
fi

"${python_bin}" experiments/exp_080a/run_experiment.py \
  --output-dir "${output_dir}"
