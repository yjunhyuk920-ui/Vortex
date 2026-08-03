#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

rm -rf results/exp_047
bash experiments/exp_047/run_current_env.sh

python - <<'PY'
from pathlib import Path
required = [
    "results/exp_047/raw/cases.jsonl",
    "results/exp_047/processed/scaling.json",
    "results/exp_047/summary.json",
    "results/exp_047/logs/run.log",
    "results/exp_047/artifacts/certificate_contract.txt",
    "results/exp_047/checksums.sha256",
]
missing = [path for path in required if not Path(path).is_file()]
if missing:
    raise SystemExit(f"missing reproduction outputs: {missing}")
print("EXP-047 reproduction outputs verified")
PY
