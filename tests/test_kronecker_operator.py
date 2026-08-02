from __future__ import annotations

import torch

from vortex_runtime.feasibility import default_specs
from vortex_runtime.kronecker_operator import (
    KroneckerLinear,
    KroneckerShape,
    balanced_factor_pair,
    choose_kronecker_shape,
    fit_kronecker_sum,
    kronecker_operator_budget,
    materialize_kronecker_sum,
)


def test_balanced_shapes_avoid_rank_one_degeneracy() -> None:
    assert balanced_factor_pair(16384) == (128, 128)
    shape = choose_kronecker_shape(2048, 2048)
    assert shape.out_features == 2048
    assert shape.in_features == 2048
    assert min(
        shape.out_first,
        shape.out_second,
        shape.in_first,
        shape.in_second,
    ) > 1
    assert shape.factor_elements_per_term == 4096


def test_rank64_405b_budget_closes_memory_traffic_and_latency() -> None:
    target, baseline = default_specs()
    budget = kronecker_operator_budget(
        target=target,
        baseline=baseline,
        rank=64,
        factor_bits=8,
        embedding_bits=4,
        active_kv_tokens=256,
        workspace_gib=1.5,
        allocator_reserve_gib=1.0,
        resident_hbm_gib_s=300.0,
        effective_tops=160.0,
    )
    assert budget.factor_gib < 2.1
    assert budget.total_memory_gib <= 8.0
    assert budget.total_traffic_gib_per_token <= budget.traffic_limit_gib_per_token
    assert budget.projected_seconds_per_token <= budget.allowed_seconds_per_token
    assert budget.pass_all


def test_fit_recovers_exact_two_term_kronecker_matrix() -> None:
    generator = torch.Generator().manual_seed(41)
    shape = KroneckerShape(
        out_first=4,
        out_second=6,
        in_first=5,
        in_second=3,
    )
    first = torch.randn(2, 4, 5, generator=generator)
    second = torch.randn(2, 6, 3, generator=generator)
    weight = materialize_kronecker_sum(
        first_factors=first,
        second_factors=second,
        shape=shape,
    )
    fitted_first, fitted_second, fitted_shape, stats = fit_kronecker_sum(
        weight,
        rank=2,
        factor_bits=16,
        oversample=2,
        power_iterations=1,
        seed=43,
    )
    reconstructed = materialize_kronecker_sum(
        first_factors=fitted_first,
        second_factors=fitted_second,
        shape=fitted_shape,
    )
    relative_error = torch.linalg.vector_norm(weight - reconstructed) / torch.linalg.vector_norm(weight)
    assert float(relative_error.item()) < 5e-4
    assert stats.relative_l2_error < 5e-4
    assert stats.compression_ratio > 1.0


def test_factorized_linear_matches_materialized_matrix() -> None:
    generator = torch.Generator().manual_seed(47)
    shape = KroneckerShape(
        out_first=3,
        out_second=4,
        in_first=2,
        in_second=5,
    )
    first = torch.randn(3, 3, 2, generator=generator)
    second = torch.randn(3, 4, 5, generator=generator)
    bias = torch.randn(12, generator=generator)
    module = KroneckerLinear(
        first_factors=first,
        second_factors=second,
        shape=shape,
        bias=bias,
    )
    x = torch.randn(2, 7, 10, generator=generator)
    weight = materialize_kronecker_sum(
        first_factors=first,
        second_factors=second,
        shape=shape,
    )
    expected = torch.nn.functional.linear(x, weight, bias)
    actual = module(x)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
