from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.feasibility import default_gate0_report
from vortex_runtime.gate0_observations import apply_real_model_observation


def generate_report(root: Path) -> dict[str, object]:
    validation_path = root / "validation_results.json"
    observed = 1.2751790996462853
    if validation_path.exists():
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        observed = float(
            payload.get("jacobi", {}).get("mean_committed_block", observed)
        )

    report = default_gate0_report(observed)
    return apply_real_model_observation(
        report,
        exact_span_path=(
            root / "results/tinyllama_1_1b_exact_span_warm_decode.json"
        ),
        oracle_path=(
            root / "results/tinyllama_1_1b_rank32_repair_oracles.json"
        ),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = generate_report(root)
    output = root / "architecture_gate0_budget.json"
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

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
        "observed_component_decision": report.get(
            "observed_component_decision"
        ),
        "rejected_mechanisms": report.get("rejected_mechanisms", []),
    }
    print(output)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
