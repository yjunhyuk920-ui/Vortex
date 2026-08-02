import json
from pathlib import Path

from vortex_runtime.feasibility import default_gate0_report
from vortex_runtime.gate0_selector_results import apply_selector_falsifications


def test_candidate_coverage_rejects_rank32_hot_representation(
    tmp_path: Path,
) -> None:
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "evidence_level": "E1 test coverage",
                "max_rank": 32,
                "evaluated_tokens": 32,
                "exact_top1_match_rate": 0.15625,
                "coverage_at_k": {"32": 0.40625},
                "rank_statistics": {
                    "minimum": 1,
                    "maximum": 17533,
                    "mean": 2942.9375,
                },
                "first_divergence": {"exact_token_rank": 2},
                "decision": "reject rank-32 hot representation",
                "rejection_scope": "test rejection",
            }
        ),
        encoding="utf-8",
    )

    report = apply_selector_falsifications(
        default_gate0_report(),
        proposal_adjoint_path=tmp_path / "missing-proposal.json",
        margin_bound_path=tmp_path / "missing-margin.json",
        prefill_compiled_path=tmp_path / "missing-prefill.json",
        candidate_coverage_path=coverage,
        rank_frontier_path=tmp_path / "missing-frontier.json",
        session_prefill_path=tmp_path / "missing-session.json",
    )

    assert report["status"] == "rank32-hot-representation-rejected"
    assert report["gates"]["rank32_topk_coverage"] is False
    assert (
        report["hot_representation_coverage"]["coverage_at_k"]["32"]
        == 0.40625
    )
    assert "rank 72" in report["next_candidate"]
