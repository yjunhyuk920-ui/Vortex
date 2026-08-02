from scripts.aggregate_causal_burnin_frontier import aggregate_results


def point(
    burnin: int,
    *,
    top1: float,
    top32: float,
    warm_pass: bool,
    full_pass: bool,
) -> dict[str, object]:
    return {
        "model": "tiny",
        "device": "cpu",
        "evaluated_unseen_tokens": 8,
        "causal_contract": "causal burnin",
        "decision_rule": "coverage gate",
        "exact_burnin_tokens": burnin,
        "burnin_token_ids": list(range(burnin)),
        "compiled_rank_statistics": {
            "minimum": 4 + burnin,
            "maximum": 4 + burnin,
            "mean": 4 + burnin,
        },
        "capsule_bits": 8,
        "prompt_burnin_reconstruction_after_quantization": {
            "maximum_module_output_relative_error": 0.1,
            "mean_module_output_relative_error": 0.05,
            "per_module": {"o_proj": 0.1},
        },
        "quantization": {"aggregate": {"logical_total_bytes": 100}},
        "hot_budget": {"pass_all": True},
        "startup_exact_cost": {
            "minimum_traffic_amortization_horizon": 100 * burnin,
            "minimum_compute_amortization_horizon": 10 * burnin,
            "horizon_4096_pass": full_pass,
        },
        "exact_top1_match_rate": top1,
        "coverage_at_k": {"32": top32},
        "rank_statistics": {"minimum": 1, "maximum": 20, "mean": 5.0},
        "first_divergence": {"exact_token_rank": 2},
        "warm_decode_candidate_pass": warm_pass,
        "full_session_4096_pass": full_pass,
        "elapsed_seconds": 1.0,
    }


def test_aggregate_selects_smallest_high_coverage_survivor() -> None:
    baseline = point(0, top1=0.4, top32=0.8, warm_pass=False, full_pass=False)
    survivor = point(2, top1=0.6, top32=1.0, warm_pass=True, full_pass=True)
    later = point(4, top1=0.7, top32=1.0, warm_pass=True, full_pass=False)

    result = aggregate_results([later, baseline, survivor])

    assert result["tested_burnin_tokens"] == [0, 2, 4]
    assert result["decision"] == "advance causal burnin local trajectory capsule"
    assert len(result["warm_decode_survivors"]) == 2
    assert len(result["full_session_4096_survivors"]) == 1
    assert result["best_observed_point"]["exact_burnin_tokens"] == 2
