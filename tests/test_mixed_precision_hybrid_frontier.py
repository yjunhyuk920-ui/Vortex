from scripts.aggregate_mixed_precision_hybrid_frontier import aggregate_results


def point(
    *,
    global_rank: int,
    session_rank: int,
    global_bits: int,
    session_bits: int,
    top1: float,
    top32: float,
    decision: str,
) -> dict[str, object]:
    return {
        "model": "tiny",
        "device": "cpu",
        "evaluated_continuation_tokens": 8,
        "compiler_contract": "causal",
        "decision_rule": "gate",
        "global_rank_limit": global_rank,
        "session_rank_limit": session_rank,
        "total_rank_limit": global_rank + session_rank,
        "global_bits": global_bits,
        "session_bits": session_bits,
        "global_rank_statistics": {"minimum": global_rank, "maximum": global_rank, "mean": global_rank},
        "added_session_rank_statistics": {"minimum": session_rank, "maximum": session_rank, "mean": session_rank},
        "final_rank_statistics": {"minimum": global_rank + session_rank, "maximum": global_rank + session_rank, "mean": global_rank + session_rank},
        "prompt_reconstruction_before_quantization": {
            "maximum_final_output_relative_error": 0.0,
            "maximum_final_input_relative_error": 0.0,
        },
        "prompt_reconstruction_after_quantization": {
            "maximum_module_output_relative_error": 0.1,
            "mean_module_output_relative_error": 0.05,
            "per_module": {"o_proj": 0.1},
        },
        "quantization": {"aggregate": {"logical_total_bytes": 10}},
        "budget": {"pass_all": True},
        "exact_top1_match_rate": top1,
        "coverage_at_k": {"32": top32},
        "rank_statistics": {"minimum": 1, "maximum": 10, "mean": 3.0},
        "first_divergence": {"exact_token_rank": 2},
        "decision": decision,
        "elapsed_seconds": 1.0,
    }


def test_aggregate_selects_best_point_and_survivor() -> None:
    failed = point(
        global_rank=58,
        session_rank=45,
        global_bits=4,
        session_bits=8,
        top1=0.4,
        top32=0.8,
        decision="reject tested mixed-precision hybrid allocation",
    )
    passed = point(
        global_rank=80,
        session_rank=45,
        global_bits=4,
        session_bits=6,
        top1=0.5,
        top32=1.0,
        decision="advance mixed-precision hybrid certificate",
    )

    result = aggregate_results([failed, passed])

    assert result["decision"] == "advance mixed-precision hybrid allocation"
    assert len(result["surviving_points"]) == 1
    assert result["best_observed_point"]["global_rank_limit"] == 80
