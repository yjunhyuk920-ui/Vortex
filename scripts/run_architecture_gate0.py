from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.feasibility import default_gate0_report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validation_path = root / "validation_results.json"
    observed = 1.2751790996462853
    if validation_path.exists():
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        observed = float(
            payload.get("jacobi", {}).get("mean_committed_block", observed)
        )

    report = default_gate0_report(observed)
    output = root / "architecture_gate0_budget.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {
        "status": report["status"],
        "memory_total_gib": report["memory"]["total_gib"],
        "projected_traffic_gib_per_token": report["traffic"][
            "projected_gib_per_token"
        ],
        "traffic_limit_gib_per_token": report["traffic"][
            "limit_gib_per_token"
        ],
        "required_repair_efficiency": report["mechanism"][
            "required_tokens_per_full_repair_equivalent"
        ],
        "observed_repair_efficiency": report["mechanism"]["observed"],
        "shortfall_factor": report["mechanism"]["shortfall_factor"],
    }
    print(output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
