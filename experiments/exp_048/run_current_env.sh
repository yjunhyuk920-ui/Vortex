#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
OUTPUT="${EXP048_OUTPUT:-$ROOT/results/exp_048_candidate}"
rm -rf "$OUTPUT"

python -m pytest -q tests/exp_048
python experiments/exp_048/run_experiment.py \
  --config experiments/exp_048/config.json \
  --output "$OUTPUT"

python - <<'PY' "$OUTPUT/summary.json"
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
assert summary["experiment"] == "EXP-048"
assert summary["evidence_level"] == "E1"
assert summary["complete_real_operation_replacement"] is False
assert summary["phase_d_status"] == "NOT TESTED"
assert summary["MEASURED"]["exact_output_mismatches_b1"] == 0
assert summary["MEASURED"]["exact_output_mismatches_b2"] == 0
assert summary["MEASURED"]["exact_output_mismatches_b3"] == 0
assert summary["MEASURED"]["deployable_future_information_uses"] == 0
assert summary["DERIVED"]["b1_is_non_deployable_future_oracle"] is True
assert summary["gate"]["decision"] in {
    "CONTINUE_PARTIAL_LAYER_SELF_DRAFT_TO_COMPLETE_GENERATION_PHASE_C",
    "REJECT_PARTIAL_LAYER_SELF_DRAFT_CORE_RETAIN_EXACT_BLOCK_VERIFIER",
}
print(json.dumps(summary["gate"], indent=2, sort_keys=True))
PY
