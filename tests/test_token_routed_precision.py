from __future__ import annotations

import pytest

from vortex_runtime.feasibility import default_specs
from vortex_runtime.token_routed_precision import token_routed_refinement_budget


def test_token_routing_reduces_compute_without_hiding_union_transfer() -> None:
    target, baseline = default_specs()
    uniform = token_routed_refinement_budget(
        target=target,
        baseline=baseline,
        block_positions=4096,
        union_layer_fraction=0.36,
        mean_token_layer_fraction=0.36 * 0.4375,
    )
    routed = token_routed_refinement_budget(
        target=target,
        baseline=baseline,
        block_positions=4096,
        union_layer_fraction=0.36,
        mean_token_layer_fraction=0.05,
    )
    assert routed.union_residual_weight_gib == uniform.union_residual_weight_gib
    assert routed.transfer_seconds_per_block == uniform.transfer_seconds_per_block
    assert (
        routed.routed_residual_compute_seconds_per_block
        < uniform.routed_residual_compute_seconds_per_block
    )
    assert routed.ideal_seconds_per_token < uniform.ideal_seconds_per_token


def test_mean_route_cannot_exceed_union() -> None:
    target, baseline = default_specs()
    with pytest.raises(ValueError):
        token_routed_refinement_budget(
            target=target,
            baseline=baseline,
            block_positions=64,
            union_layer_fraction=0.2,
            mean_token_layer_fraction=0.3,
        )


def test_zero_residual_route_is_valid_consensus_only_point() -> None:
    target, baseline = default_specs()
    point = token_routed_refinement_budget(
        target=target,
        baseline=baseline,
        block_positions=4096,
        union_layer_fraction=0.0,
        mean_token_layer_fraction=0.0,
    )
    assert point.union_residual_weight_gib == 0.0
    assert point.routed_residual_compute_seconds_per_block == 0.0
