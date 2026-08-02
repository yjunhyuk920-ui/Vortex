from __future__ import annotations

import torch
from torch import nn

from vortex_runtime.block_kronecker import (
    BlockKroneckerLinear,
    block_kronecker_budget,
    fit_block_kronecker_linear,
)
from vortex_runtime.feasibility import default_specs
from vortex_runtime.kronecker_operator import (
    KroneckerShape,
    choose_kronecker_shape,
    materialize_kronecker_sum,
)


def _row_weight(
    first: torch.Tensor,
    second: torch.Tensor,
    shape: KroneckerShape,
) -> torch.Tensor:
    blocks = [
        materialize_kronecker_sum(
            first_factors=first[index],
            second_factors=second[index],
            shape=shape,
        )
        for index in range(first.shape[0])
    ]
    return torch.cat(blocks, dim=0)


def _column_weight(
    first: torch.Tensor,
    second: torch.Tensor,
    shape: KroneckerShape,
) -> torch.Tensor:
    blocks = [
        materialize_kronecker_sum(
            first_factors=first[index],
            second_factors=second[index],
            shape=shape,
        )
        for index in range(first.shape[0])
    ]
    return torch.cat(blocks, dim=1)


def test_405b_semantic_block_budget_closes_gate() -> None:
    target, baseline = default_specs()
    budget = block_kronecker_budget(
        target=target,
        baseline=baseline,
        factor_bits=8,
        embedding_bits=4,
        active_kv_tokens=256,
        attention_terms=4,
        mlp_terms=3,
        lm_head_terms=2,
        mlp_block_size=128,
        lm_head_block_size=256,
    )
    assert budget.factor_gib < 1.8
    assert budget.total_memory_gib < 5.5
    assert budget.total_traffic_gib_per_token < 2.0
    assert budget.projected_seconds_per_token < budget.allowed_seconds_per_token
    assert budget.pass_all


def test_row_block_forward_matches_materialized_weight() -> None:
    generator = torch.Generator().manual_seed(101)
    shape = KroneckerShape(
        out_first=2,
        out_second=3,
        in_first=2,
        in_second=4,
    )
    first = torch.randn(3, 2, 2, 2, generator=generator)
    second = torch.randn(3, 2, 3, 4, generator=generator)
    bias = torch.randn(18, generator=generator)
    module = BlockKroneckerLinear(
        mode="row",
        first_factors=first,
        second_factors=second,
        shape=shape,
        bias=bias,
    )
    weight = _row_weight(first, second, shape)
    x = torch.randn(2, 5, 8, generator=generator)
    expected = torch.nn.functional.linear(x, weight, bias)
    actual = module(x)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_column_block_forward_matches_materialized_weight() -> None:
    generator = torch.Generator().manual_seed(103)
    shape = KroneckerShape(
        out_first=2,
        out_second=3,
        in_first=2,
        in_second=4,
    )
    first = torch.randn(4, 2, 2, 2, generator=generator)
    second = torch.randn(4, 2, 3, 4, generator=generator)
    bias = torch.randn(6, generator=generator)
    module = BlockKroneckerLinear(
        mode="column",
        first_factors=first,
        second_factors=second,
        shape=shape,
        bias=bias,
    )
    weight = _column_weight(first, second, shape)
    x = torch.randn(2, 5, 32, generator=generator)
    expected = torch.nn.functional.linear(x, weight, bias)
    actual = module(x)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_rank_one_semantic_blocks_are_recovered() -> None:
    generator = torch.Generator().manual_seed(107)
    shape = choose_kronecker_shape(8, 6)
    first = torch.randn(
        3,
        1,
        shape.out_first,
        shape.in_first,
        generator=generator,
    )
    second = torch.randn(
        3,
        1,
        shape.out_second,
        shape.in_second,
        generator=generator,
    )
    weight = _row_weight(first, second, shape)
    linear = nn.Linear(6, 24, bias=False)
    with torch.no_grad():
        linear.weight.copy_(weight)
    fitted, stats = fit_block_kronecker_linear(
        linear,
        mode="row",
        block_size=8,
        factor_bits=16,
        power_iterations=6,
        seed=109,
    )
    x = torch.randn(4, 6, generator=generator)
    expected = linear(x)
    actual = fitted(x)
    assert float(stats["relative_l2_error"]) < 1e-3
    assert torch.allclose(actual, expected, atol=3e-3, rtol=3e-3)
