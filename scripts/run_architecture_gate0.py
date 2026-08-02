from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.feasibility import default_gate0_report
from vortex_runtime.gate0_observations import apply_real_model_observation
from vortex_runtime.gate0_selector_results import apply_selector_falsifications


def generate_report(root: Path) -> dict[str, object]:
    validation_path = root / "validation_results.json"
    observed = 1.2751790996462853
    if validation_path.exists():
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        observed = float(
            payload.get("jacobi", {}).get("mean_committed_block", observed)
        )

    report = default_gate0_report(observed)
    report = apply_real_model_observation(
        report,
        exact_span_path=(
            root / "results/tinyllama_1_1b_exact_span_warm_decode.json"
        ),
        oracle_path=(
            root / "results/tinyllama_1_1b_rank32_repair_oracles.json"
        ),
        residual_path=(
            root / "results/tinyllama_1_1b_residual_tile_oracle.json"
        ),
        adjoint_path=(
            root / "results/tinyllama_1_1b_adjoint_tile_oracle.json"
        ),
        combined_path=(
            root
            / "results/tinyllama_1_1b_block_shared_combined_gate.json"
        ),
        residual_selector_path=(
            root
            / "results/tinyllama_1_1b_block_shared_residual_selector.json"
        ),
    )
    return apply_selector_falsifications(
        report,
        proposal_adjoint_path=(
            root
            / "results/tinyllama_1_1b_block_shared_proposal_adjoint_oracle.json"
        ),
        margin_bound_path=(
            root
            / "results/tinyllama_1_1b_block_shared_margin_bound_selector.json"
        ),
        prefill_compiled_path=(
            root
            / "results/tinyllama_1_1b_prefill_compiled_adjoint_oracle.json"
        ),
        candidate_coverage_path=(
            root / "results/tinyllama_1_1b_hot_candidate_coverage.json"
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
        "original_design_projected_traffic_gib_per_token": report["traffic"][
            "projected_gib_per_token"
        ],
        "traffic_limit_gib_per_token": report["traffic"][
            "limit_gib_per_token"
        ],
        "original_design_projected_compute_gflop_per_token": report["compute"][
            "projected_gflop_per_token"
        ],
        "compute_limit_gflop_per_token": report["compute"][
            "limit_gflop_per_token"
        ],
        "original_design_repair_fraction": report["compute"][
            "candidate_repair_fraction"
        ],
        "maximum_compute_repair_fraction": report["compute"][
            "maximum_repair_fraction"
        ],
        "required_repair_efficiency": report["mechanism"][
            "required_tokens_per_full_repair_equivalent"
        ],
        "observed_repair_efficiency": report["mechanism"]["observed"],
        "observed_repair_fraction": report["mechanism"][
            "observed_repair_fraction"
        ],
        "observed_incremental_committed_tokens": report["mechanism"].get(
            "observed_incremental_committed_tokens"
        ),
        "logical_oracle": report.get("logical_oracle"),
        "selector_falsification": report.get("selector_falsification"),
        "family_decision": report.get("family_decision"),
        "hot_representation_coverage": report.get("hot_representation_coverage"),
        "revised_oracle_envelope": report.get("revised_oracle_envelope"),
        "observed_component_decision": report.get(
            "observed_component_decision"
        ),
        "rejected_mechanisms": report.get("rejected_mechanisms", []),
        "next_candidate": report.get("next_candidate"),
    }
    print(output)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
