from scripts.aggregate_local_affine_dictionary_frontier import aggregate_results


def point(
    *,
    clusters: int,
    local_rank: int,
    bits: int,
    top1: float,
    top32: float,
    qualifies: bool,
) -> dict[str, object]:
    return {
        "model": "tiny",
        "device": "cpu",
        "evaluated_unseen_tokens": 8,
        "causal_contract": "causal dictionary",
        "budget_contract": "stored and active ranks differ",
        "decision_rule": "coverage gate",
        "clusters": clusters,
        "local_rank": local_rank,
        "capsule_bits": bits,
        "captured_vectors_per_module": 32,
        "stored_equivalent_rank": clusters * (local_rank + 1),
        "active_equivalent_rank": clusters + local_rank + 1,
        "stored_budget": {"memory_pass": True},
        "active_budget": {"traffic_pass": True, "compute_pass": True},
        "actual_stored_response_columns": {
            "minimum": 10,
            "maximum": 10,
            "mean": 10.0,
        },
        "actual_active_equivalent_columns": {
            "minimum": 5,
            "maximum": 5,
            "mean": 5.0,
        },
        "post_quantization_training_reconstruction": {
            "maximum_module_output_relative_error": 0.1,
            "mean_module_output_relative_error": 0.05,
            "per_module": {"o_proj": 0.1},
        },
        "exact_top1_match_rate": top1,
        "coverage_at_k": {"32": top32},
        "rank_statistics": {"minimum": 1, "maximum": 20, "mean": 5.0},
        "first_divergence": {"exact_token_rank": 2},
        "decision": (
            "advance routed local affine dictionary"
            if qualifies
            else "reject tested routed local affine dictionary point"
        ),
        "elapsed_seconds": 1.0,
    }


def test_aggregate_selects_surviving_dictionary() -> None:
    failed = point(
        clusters=4,
        local_rank=8,
        bits=8,
        top1=0.5,
        top32=0.8,
        qualifies=False,
    )
    passed = point(
        clusters=8,
        local_rank=8,
        bits=8,
        top1=0.7,
        top32=1.0,
        qualifies=True,
    )

    result = aggregate_results([failed, passed])

    assert result["decision"] == "advance routed local affine dictionary"
    assert len(result["surviving_points"]) == 1
    assert result["best_observed_point"]["clusters"] == 8
