from __future__ import annotations

import torch

from vortex_runtime import quadratic_bf16_residual_generator as base
from vortex_runtime.quadratic_bf16_residual_fp32_context import (
    install_fp32_context_semantics,
    select_context_basis_fp32,
)


def _finite_quadratic_outputs(
    bits: torch.Tensor, output_size: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = base.quadratic_features(bits)
    # Restrict XOR coefficients to the seven BF16 mantissa bits and keep a
    # fixed finite exponent. Every generated word is therefore finite.
    coefficient_words = torch.randint(
        0,
        128,
        (features.shape[1], output_size),
        generator=generator,
        dtype=torch.int32,
    )
    residual = torch.zeros((features.shape[0], output_size), dtype=torch.int32)
    for index in range(features.shape[1]):
        mask = features[:, index].bool()
        if bool(mask.any()):
            residual[mask] = torch.bitwise_xor(
                residual[mask], coefficient_words[index]
            )
    baseline = torch.full((output_size,), 0x3F80, dtype=torch.int32)
    return base.words_to_bf16(torch.bitwise_xor(residual, baseline))


def test_returned_context_bits_equal_deployed_fp32_evaluation() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260821)
    x = torch.randn((24, 48), generator=generator).to(torch.bfloat16)
    mean, basis, fitted_bits, fitted_rank = select_context_basis_fp32(
        x, context_bits=12, pool_size=64, seed=20260820
    )
    deployed_bits = ((x.float() - mean) @ basis >= 0).to(torch.uint8)
    assert mean.dtype == torch.float32
    assert basis.dtype == torch.float32
    assert torch.equal(fitted_bits, deployed_bits)
    assert fitted_rank == base.gf2_rank(base.quadratic_features(deployed_bits))


def test_corrected_fit_program_replays_executable_build_semantics() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260822)
    x = torch.randn((24, 48), generator=generator).to(torch.bfloat16)
    _mean, _basis, bits, rank = select_context_basis_fp32(
        x, context_bits=12, pool_size=64, seed=20260820
    )
    assert rank == x.shape[0]
    y = _finite_quadratic_outputs(bits, output_size=16, seed=20260823)
    centroid = base.normalize_rows(x).mean(dim=0)
    centroid = centroid / torch.linalg.vector_norm(centroid).clamp_min(1e-30)

    install_fp32_context_semantics()
    program, exact = base.fit_program(
        x,
        y,
        centroid,
        context_bits=12,
        context_pool=64,
        seed=20260820,
    )
    assert exact
    assert torch.equal(program.query(x).view(torch.int16), y.view(torch.int16))
