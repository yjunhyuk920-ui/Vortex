#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${EXP071_OUTPUT_DIR:-$ROOT/results/exp_071_candidate}"
cd "$ROOT"
python -m pytest -q tests/exp_071
python experiments/exp_071/run_experiment.py --output-dir "$OUTPUT"
python validate_repo.py
