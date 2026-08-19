from __future__ import annotations

import torch
from torch import nn

from vortex_runtime.global_bf16_successive_refinement import (
    BF16_MANTISSA_BITS,
    CompiledGlobalBF16Refinement,
    SuccessiveRefinementError,
    aggregate_oracle_rows,
    bf16_prefix,
    compile_global_bf16_refinement,
    symbol_entropy_bits,
)


class TinySwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(6, 4, bias=False, dtype=torch.bfloat16)
        self.act_fn = torch.nn.functional.silu
        generator = torch.Generator().manual_seed(20260819)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator).to(torch.bfloat16))


def test_bf16_prefix_identity_and_monotone_mask() -> None:
    tensor = torch.tensor([1.0, -2.5, 0.10009765625, 17.75], dtype=torch.bfloat16)
    exact = bf16_prefix(tensor, BF16_MANTISSA_BITS)
    assert torch.equal(exact.view(torch.uint16), tensor.view(torch.uint16))
    previous = bf16_prefix(tensor, 0).view(torch.uint16)
    for bits in range(1, BF16_MANTISSA_BITS + 1):
        current = bf16_prefix(tensor, bits).view(torch.uint16)
        low = BF16_MANTISSA_BITS - bits
        if low:
            assert torch.all((current & ((1 << low) - 1)) == 0)
        assert torch.equal(
            bf16_prefix(current.view(torch.bfloat16), bits - 1).view(torch.uint16),
            previous,
        )
        previous = current


def test_invalid_dtype_and_precision_fail_closed() -> None:
    try:
        bf16_prefix(torch.ones(2, dtype=torch.float32), 3)
    except SuccessiveRefinementError:
        pass
    else:
        raise AssertionError("float32 must fail closed")
    for bits in (-1, 8):
        try:
            bf16_prefix(torch.ones(2, dtype=torch.bfloat16), bits)
        except SuccessiveRefinementError:
            pass
        else:
            raise AssertionError("invalid precision must fail closed")


def test_entropy_prefix_is_bounded_by_full_entropy() -> None:
    tensor = torch.tensor([1.0, 1.0, 1.5, -1.0, -1.5, 2.0, 3.0, 3.0], dtype=torch.bfloat16)
    entropies = [symbol_entropy_bits(tensor, bits) for bits in range(8)]
    assert all(0.0 <= value <= 3.0 for value in entropies)
    assert all(left <= right + 1e-12 for left, right in zip(entropies, entropies[1:]))


def test_compiler_full_precision_control_and_oracles() -> None:
    compiler = compile_global_bf16_refinement(TinySwiGLU())
    x = torch.tensor([
        [0.25, -1.0, 0.5, 2.0],
        [1.0, 0.0, -0.75, 0.125],
        [-2.0, 0.5, 1.5, -0.25],
    ], dtype=torch.bfloat16)
    result = compiler.evaluate(x)
    assert result.controls["full_precision_control_passed"]
    assert result.controls["full_precision_exact_activations"] == 3
    assert all(value == 0 for value in result.controls["prefix_identity_weight_mismatches"].values())
    assert len(result.global_rows) == 3
    assert len(result.row_adaptive_rows) == 3
    assert len(result.precision_grid) == 64
    for row in result.global_rows + result.row_adaptive_rows:
        assert row["exact"]
        assert row["candidate_sha256"] == row["reference_sha256"]
        assert 0.0 < row["entropy_fraction_of_full"] <= 1.0
        assert 0.0 < row["raw_fraction_of_bf16"] <= 1.0
        assert row["minimum_perfect_acceptance_p50"] >= 1
    aggregate = aggregate_oracle_rows(result.row_adaptive_rows)
    assert aggregate["all_exact"] and aggregate["count"] == 3


def test_weight_mutation_changes_fingerprint() -> None:
    mlp = TinySwiGLU()
    first = CompiledGlobalBF16Refinement.from_mlp(mlp).manifest()["fingerprint"]
    with torch.no_grad():
        bits = mlp.gate_proj.weight.view(torch.uint16)
        bits[0, 0] ^= torch.tensor(1, dtype=torch.uint16)
    second = CompiledGlobalBF16Refinement.from_mlp(mlp).manifest()["fingerprint"]
    assert first != second
