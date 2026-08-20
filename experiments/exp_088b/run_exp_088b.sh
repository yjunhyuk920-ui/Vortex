#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_ID="${GITHUB_SHA:-local}"
cd "$ROOT"

python experiments/exp_088b/run_experiment.py \
  --config experiments/exp_088b/config.json \
  --output-dir "results/exp_088b/${SOURCE_ID}"
