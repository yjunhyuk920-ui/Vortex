#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
OUTPUT="${EXP047R_OUTPUT:-$ROOT/results/exp_047r_candidate}"
rm -rf "$OUTPUT"

python -m pytest -q tests/exp_047r
python experiments/exp_047r/run_oracle_audit.py \
  --config experiments/exp_047r/config.json \
  --output "$OUTPUT"

python - <<'PY' "$OUTPUT/summary.json"
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
assert summary["experiment"] == "EXP-047R"
assert summary["evidence_level"] == "E1"
assert summary["phase_c_operation_replacement"] is False
assert summary["phase_d_status"] == "NOT TESTED"
assert summary["future_information_used"] is False
assert summary["MEASURED"]["wrong_accepts"] == 0
assert summary["MEASURED"]["bound_violations"] == 0
assert summary["gate"]["decision"] in {
    "CONTINUE_TO_INDEPENDENT_C3_AND_REAL_OPERATION_REPLACEMENT",
    "REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY",
}
print(json.dumps(summary["gate"], indent=2, sort_keys=True))
PY
