from vortex_runtime.block_gate import (
    BlockSharedGate,
    maximum_selected_bytes_for_combined_gate,
    maximum_selected_bytes_for_compute,
)


FULL_MODEL_BYTES = 4_400_193_536
HOT_GFLOP = 3.531515136
FULL_REPAIR_GFLOP = 845.521354752
COMPUTE_LIMIT_GFLOP = 12.1327352832
TRAFFIC_EFFICIENCY = 491.29915997929805


def test_512_mib_shared_over_64_passes_traffic_but_fails_compute() -> None:
    gate = BlockSharedGate(
        committed_tokens=64,
        selected_weight_bytes=512 * 1024 * 1024,
        full_model_weight_bytes=FULL_MODEL_BYTES,
        minimum_traffic_efficiency=TRAFFIC_EFFICIENCY,
        hot_gflop_per_token=HOT_GFLOP,
        full_exact_repair_gflop_per_token=FULL_REPAIR_GFLOP,
        compute_limit_gflop_per_token=COMPUTE_LIMIT_GFLOP,
    )
    assert gate.traffic_pass
    assert not gate.compute_pass
    assert not gate.pass_all
    assert gate.projected_total_gflop_per_token > 100.0


def test_compute_gate_limits_selected_weights_to_about_one_percent() -> None:
    maximum = maximum_selected_bytes_for_compute(
        full_model_weight_bytes=FULL_MODEL_BYTES,
        hot_gflop_per_token=HOT_GFLOP,
        full_exact_repair_gflop_per_token=FULL_REPAIR_GFLOP,
        compute_limit_gflop_per_token=COMPUTE_LIMIT_GFLOP,
    )
    fraction = maximum / FULL_MODEL_BYTES
    assert 0.010 < fraction < 0.011
    assert 42 * 1024 * 1024 < maximum < 44 * 1024 * 1024


def test_combined_gate_uses_compute_limit_for_64_token_block() -> None:
    maximum = maximum_selected_bytes_for_combined_gate(
        committed_tokens=64,
        full_model_weight_bytes=FULL_MODEL_BYTES,
        minimum_traffic_efficiency=TRAFFIC_EFFICIENCY,
        hot_gflop_per_token=HOT_GFLOP,
        full_exact_repair_gflop_per_token=FULL_REPAIR_GFLOP,
        compute_limit_gflop_per_token=COMPUTE_LIMIT_GFLOP,
    )
    compute_maximum = maximum_selected_bytes_for_compute(
        full_model_weight_bytes=FULL_MODEL_BYTES,
        hot_gflop_per_token=HOT_GFLOP,
        full_exact_repair_gflop_per_token=FULL_REPAIR_GFLOP,
        compute_limit_gflop_per_token=COMPUTE_LIMIT_GFLOP,
    )
    assert maximum == compute_maximum
