from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import torch
from torch.nn import functional as F

from vortex_runtime.feasibility import GIB, ModelSpec
from vortex_runtime.partitioned_signed_dual import _local_silu_lipschitz
from vortex_runtime.signed_dual_mlp import SignedDualTerms


@dataclass(frozen=True)
class SignedResidualCodeBudget:
    block_size: int
    rank: int
    blocks_per_hidden_vector: int
    coefficient_bits: int
    remainder_bits: int
    basis_bits: int
    coefficient_and_remainder_gib: float
    basis_gib: float
    total_metadata_gib: float
    metadata_limit_gib: float
    metadata_pass: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class SignedResidualCodeDiagnostics:
    activation_perpendicular_ratio: float
    dual_perpendicular_ratio: float
    gate_radius_to_global: float
    up_radius_to_global: float
    directional_radius_to_global: float
    gate_signed_center_l1: float
    up_signed_center_l1: float
    directional_signed_center_l1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class MatrixResidualCode:
    exact_weight: torch.Tensor
    hot_weight: torch.Tensor
    global_residual_norms: torch.Tensor
    bases: list[torch.Tensor]
    coefficients: list[torch.Tensor]
    remainder_norms: list[torch.Tensor]
    block_size: int


@dataclass
class BlockSignedResidualKernel:
    gate: MatrixResidualCode
    up: MatrixResidualCode
    down_transposed: MatrixResidualCode
    activation_bases: list[torch.Tensor]
    dual_bases: list[torch.Tensor]
    bits: int
    block_size: int
    rank: int

    @property
    def hidden_size(self) -> int:
        return int(self.gate.exact_weight.shape[1])

    @property
    def intermediate_size(self) -> int:
        return int(self.gate.exact_weight.shape[0])


def _quantize_rows(weight: torch.Tensor, *, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    source = weight.detach().to("cpu", torch.float32).contiguous()
    if source.ndim != 2:
        raise ValueError("weight must be a matrix")
    if bits >= 16:
        return source, torch.zeros_like(source)
    if bits < 2:
        raise ValueError("bits must be at least 2")
    qmax = (1 << (bits - 1)) - 1
    maximum = source.abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
    hot = torch.round(source / scale).clamp(-qmax, qmax) * scale
    return hot.contiguous(), (source - hot).contiguous()


def build_block_bases(
    vectors: list[torch.Tensor],
    *,
    block_size: int,
    rank: int,
) -> list[torch.Tensor]:
    if not vectors:
        raise ValueError("at least one build vector is required")
    if min(block_size, rank) <= 0:
        raise ValueError("block size and rank must be positive")
    flattened = [vector.detach().to("cpu", torch.float32).reshape(-1) for vector in vectors]
    hidden = flattened[0].numel()
    if any(vector.numel() != hidden for vector in flattened):
        raise ValueError("all build vectors must share one hidden size")

    bases: list[torch.Tensor] = []
    for start in range(0, hidden, block_size):
        samples = torch.stack(
            [vector[start : start + block_size] for vector in flattened],
            dim=0,
        )
        requested = min(rank, samples.shape[0], samples.shape[1])
        if requested == 0 or float(torch.linalg.vector_norm(samples).item()) == 0.0:
            bases.append(torch.zeros(samples.shape[1], 0, dtype=torch.float32))
            continue
        _, _, vh = torch.linalg.svd(samples, full_matrices=False)
        bases.append(vh[:requested].T.contiguous())
    return bases


def _compile_matrix_code(
    *,
    weight: torch.Tensor,
    bases: list[torch.Tensor],
    bits: int,
    block_size: int,
) -> MatrixResidualCode:
    exact = weight.detach().to("cpu", torch.float32).contiguous()
    hot, residual = _quantize_rows(exact, bits=bits)
    global_norms = torch.linalg.vector_norm(residual, dim=1)
    coefficients: list[torch.Tensor] = []
    remainder_norms: list[torch.Tensor] = []
    expected_blocks = ceil(exact.shape[1] / block_size)
    if len(bases) != expected_blocks:
        raise ValueError("basis partition does not match matrix columns")

    for block_index, start in enumerate(range(0, exact.shape[1], block_size)):
        block = residual[:, start : start + block_size]
        basis = bases[block_index]
        if basis.shape[0] != block.shape[1]:
            raise ValueError("basis block width does not match residual block")
        if basis.shape[1] == 0:
            coefficient = torch.zeros(block.shape[0], 0, dtype=torch.float32)
            remainder = block
        else:
            coefficient = block @ basis
            remainder = block - coefficient @ basis.T
        coefficients.append(coefficient.contiguous())
        remainder_norms.append(
            torch.linalg.vector_norm(remainder, dim=1).contiguous()
        )
    return MatrixResidualCode(
        exact_weight=exact,
        hot_weight=hot,
        global_residual_norms=global_norms.contiguous(),
        bases=bases,
        coefficients=coefficients,
        remainder_norms=remainder_norms,
        block_size=block_size,
    )


def compile_block_signed_residual_kernel(
    *,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    activation_build_vectors: list[torch.Tensor],
    dual_build_vectors: list[torch.Tensor],
    bits: int,
    block_size: int,
    rank: int,
) -> BlockSignedResidualKernel:
    gate = gate_weight.detach().to("cpu", torch.float32)
    up = up_weight.detach().to("cpu", torch.float32)
    down = down_weight.detach().to("cpu", torch.float32)
    if gate.ndim != 2 or up.shape != gate.shape:
        raise ValueError("gate and up must be matching matrices")
    intermediate, hidden = gate.shape
    if down.shape != (hidden, intermediate):
        raise ValueError("down must have shape [hidden, intermediate]")

    activation_bases = build_block_bases(
        activation_build_vectors,
        block_size=block_size,
        rank=rank,
    )
    dual_bases = build_block_bases(
        dual_build_vectors,
        block_size=block_size,
        rank=rank,
    )
    return BlockSignedResidualKernel(
        gate=_compile_matrix_code(
            weight=gate,
            bases=activation_bases,
            bits=bits,
            block_size=block_size,
        ),
        up=_compile_matrix_code(
            weight=up,
            bases=activation_bases,
            bits=bits,
            block_size=block_size,
        ),
        down_transposed=_compile_matrix_code(
            weight=down.T,
            bases=dual_bases,
            bits=bits,
            block_size=block_size,
        ),
        activation_bases=activation_bases,
        dual_bases=dual_bases,
        bits=bits,
        block_size=block_size,
        rank=rank,
    )


def _apply_matrix_code(
    code: MatrixResidualCode,
    vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    x = vector.detach().to("cpu", torch.float32).reshape(-1)
    if x.numel() != code.exact_weight.shape[1]:
        raise ValueError("vector does not match coded matrix columns")
    center = code.hot_weight @ x
    radius = torch.zeros(code.exact_weight.shape[0], dtype=torch.float32)
    total_perpendicular_sq = 0.0
    total_norm_sq = float(torch.dot(x, x).item())
    signed_center_l1 = 0.0

    for block_index, start in enumerate(range(0, x.numel(), code.block_size)):
        block = x[start : start + code.block_size]
        basis = code.bases[block_index]
        coefficient = code.coefficients[block_index]
        if basis.shape[1] == 0:
            perpendicular = block
        else:
            coordinates = basis.T @ block
            signed = coefficient @ coordinates
            center = center + signed
            signed_center_l1 += float(signed.abs().sum().item())
            perpendicular = block - basis @ coordinates
        perpendicular_norm = torch.linalg.vector_norm(perpendicular)
        radius = radius + code.remainder_norms[block_index] * perpendicular_norm
        total_perpendicular_sq += float(torch.dot(perpendicular, perpendicular).item())

    global_radius = code.global_residual_norms * torch.linalg.vector_norm(x)
    perpendicular_ratio = (
        total_perpendicular_sq / max(total_norm_sq, 1e-24)
    ) ** 0.5
    radius_ratio = float(radius.sum().item()) / max(
        float(global_radius.sum().item()),
        1e-24,
    )
    return center, radius, perpendicular_ratio, radius_ratio, signed_center_l1


def _product_interval(
    left_center: torch.Tensor,
    left_radius: torch.Tensor,
    right_center: torch.Tensor,
    right_radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_low = left_center - left_radius
    left_high = left_center + left_radius
    right_low = right_center - right_radius
    right_high = right_center + right_radius
    products = torch.stack(
        (
            left_low * right_low,
            left_low * right_high,
            left_high * right_low,
            left_high * right_high,
        ),
        dim=0,
    )
    return products.amin(dim=0), products.amax(dim=0)


def build_block_signed_residual_terms(
    kernel: BlockSignedResidualKernel,
    *,
    activation: torch.Tensor,
    output_dual: torch.Tensor,
) -> tuple[SignedDualTerms, SignedResidualCodeDiagnostics]:
    x = activation.detach().to("cpu", torch.float32).reshape(-1)
    q = output_dual.detach().to("cpu", torch.float32).reshape(-1)
    gate_center, gate_radius, x_perp, gate_ratio, gate_signed = _apply_matrix_code(
        kernel.gate,
        x,
    )
    up_center, up_radius, _, up_ratio, up_signed = _apply_matrix_code(
        kernel.up,
        x,
    )
    directional_center, directional_radius, q_perp, down_ratio, down_signed = (
        _apply_matrix_code(kernel.down_transposed, q)
    )

    gate_lower = gate_center - gate_radius
    gate_upper = gate_center + gate_radius
    local_lipschitz = _local_silu_lipschitz(gate_lower, gate_upper)
    silu_center = F.silu(gate_center)
    activation_center = silu_center * up_center
    activation_radius = (
        local_lipschitz * gate_radius * (up_center.abs() + up_radius)
        + silu_center.abs() * up_radius
    )
    approximate = activation_center * directional_center
    lower, upper = _product_interval(
        activation_center,
        activation_radius,
        directional_center,
        directional_radius,
    )

    exact_activation = F.silu(kernel.gate.exact_weight @ x) * (
        kernel.up.exact_weight @ x
    )
    exact_directional = kernel.down_transposed.exact_weight @ q
    exact = exact_activation * exact_directional
    terms = SignedDualTerms(
        exact_contributions=exact.contiguous(),
        approximate_contributions=approximate.contiguous(),
        lower_contributions=lower.contiguous(),
        upper_contributions=upper.contiguous(),
        activation_error_bounds=activation_radius.contiguous(),
        directional_error_bounds=directional_radius.contiguous(),
    )
    diagnostics = SignedResidualCodeDiagnostics(
        activation_perpendicular_ratio=x_perp,
        dual_perpendicular_ratio=q_perp,
        gate_radius_to_global=gate_ratio,
        up_radius_to_global=up_ratio,
        directional_radius_to_global=down_ratio,
        gate_signed_center_l1=gate_signed,
        up_signed_center_l1=up_signed,
        directional_signed_center_l1=down_signed,
    )
    return terms, diagnostics


def signed_residual_code_budget(
    *,
    target: ModelSpec,
    block_size: int,
    rank: int,
    coefficient_bits: int = 32,
    remainder_bits: int = 32,
    basis_bits: int = 32,
    metadata_limit_gib: float = 6.0,
) -> SignedResidualCodeBudget:
    if min(block_size, rank, coefficient_bits, remainder_bits, basis_bits) <= 0:
        raise ValueError("code dimensions and precisions must be positive")
    blocks = ceil(target.hidden_size / block_size)
    rows = target.layers * target.intermediate_size * 3
    coefficient_values = rows * blocks * rank
    remainder_values = rows * blocks
    coefficient_and_remainder = (
        coefficient_values * coefficient_bits / 8
        + remainder_values * remainder_bits / 8
    ) / GIB
    # Gate and up share one activation basis; down uses one dual basis.
    basis_values = target.layers * 2 * target.hidden_size * rank
    basis_gib = basis_values * basis_bits / 8 / GIB
    total = coefficient_and_remainder + basis_gib
    return SignedResidualCodeBudget(
        block_size=block_size,
        rank=rank,
        blocks_per_hidden_vector=blocks,
        coefficient_bits=coefficient_bits,
        remainder_bits=remainder_bits,
        basis_bits=basis_bits,
        coefficient_and_remainder_gib=coefficient_and_remainder,
        basis_gib=basis_gib,
        total_metadata_gib=total,
        metadata_limit_gib=metadata_limit_gib,
        metadata_pass=total <= metadata_limit_gib,
    )
