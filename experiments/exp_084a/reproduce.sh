#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="."
.deps/exp076-venv/Scripts/python.exe experiments/exp_084a/verify_results.py \
  --model-dir .deps/exp076-model --output-dir results/exp_084a
