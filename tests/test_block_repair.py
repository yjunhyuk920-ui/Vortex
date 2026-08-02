from vortex_runtime.block_repair import (
    BlockRepairBudget,
    maximum_shared_repair_bytes,
    minimum_committed_tokens,
)


def test_512_mib_shared_across_64_tokens_crosses_gate() -> None:
    full_model_bytes = 4_400_193_536
    selected = 512 * 1024 * 1024
    budget = BlockRepairBudget(
        committed_tokens=64,
        selected_weight_bytes=selected,
        full_model_weight_bytes=full_model_bytes,
    )
    assert budget.tokens_per_full_repair_equivalent is not None
    assert budget.tokens_per_full_repair_equivalent > 491.29915997929805
    assert budget.passes(491.29915997929805)
    assert minimum_committed_tokens(
        selected_weight_bytes=selected,
        full_model_weight_bytes=full_model_bytes,
        minimum_efficiency=491.29915997929805,
    ) == 60


def test_gate_byte_limit_scales_with_committed_block() -> None:
    full_model_bytes = 4_400_193_536
    one = maximum_shared_repair_bytes(
        committed_tokens=1,
        full_model_weight_bytes=full_model_bytes,
        minimum_efficiency=491.29915997929805,
    )
    sixty_four = maximum_shared_repair_bytes(
        committed_tokens=64,
        full_model_weight_bytes=full_model_bytes,
        minimum_efficiency=491.29915997929805,
    )
    assert sixty_four == 64 * one


def test_zero_repair_is_unbounded_efficiency() -> None:
    budget = BlockRepairBudget(
        committed_tokens=64,
        selected_weight_bytes=0,
        full_model_weight_bytes=4_400_193_536,
    )
    assert budget.tokens_per_full_repair_equivalent is None
    assert budget.passes(600.0)
