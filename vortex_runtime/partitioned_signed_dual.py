from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import torch
from torch.nn import functional as F

from vortex_runtime.feasibility import GIB, ModelSpec
from vortex_runtime.signed_dual_mlp import (
    SILU_GLOBAL_LIPSCHITZ,
    SignedDualTerms,
)


@dataclass(frozen=True)
class PartitionedConeMetadataBudget:
    block_size: int
    blocks_per_hidden_vector: int
    metadata_bits: int
    metadata_values: int
    metadata_gib: float
    metadata_limit_gib: float
    metadata_pass: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class PartitionedConeDiagnostics:
    global_gate_radius_sum: float
    partitioned_gate_radius_sum: float
    global_up_radius_sum: float
    partitioned_up_radius_sum: float
    global_directional_radius_sum: float
    partitioned_directional_radius_sum: float
    mean_local_silu_lipschitz: float
    maximum_local_silu_lipschitz: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class PartitionedSignedDualKernel:
    gate_exact: torch.Tensor
    up_exact: torch.Tensor
    down_exact: torch.Tensor
    gate_hat: torch.Tensor
    up_hat: torch.Tensor
    down_hat: torch.Tensor
    gate_global_residual_norms: torch.Tensor
    up_global_residual_norms: torch.Tensor
    down_global_residual_norms: torch.Tensor
    gate_block_residual_norms: torch.Tensor
    up_block_residual_norms: torch.Tensor
    down_block_residual_norms: torch.Tensor
    bits: int
    block_size: int

    @property
    def intermediate_size(self) -> int:
        return int(self.gate_exact.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.gate_exact.shape[1])

    @property
    def blocks(self) -> int:
        return int(self.gate_block_residual_norms.shape[1])


def _symmetric_quantize_rows(
    weight: torch.Tensor,
    *,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    source = weight.detach().to("cpu", torch.float32).contiguous()
    if bits >= 16:
        residual = torch.zeros_like(source)
        return source, residual, torch.zeros(source.shape[0], dtype=torch.float32)
    if bits < 2:
        raise ValueError("bits must be at least 2")
    qmax = (1 << (bits - 1)) - 1
    maximum = source.abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
    restored = torch.round(source / scale).clamp(-qmax, qmax) * scale
    residual = source - restored
    global_norm = torch.linalg.vector_norm(residual, dim=1)
    return restored.contiguous(), residual.contiguous(), global_norm.contiguous()


def _block_row_norms(residual: torch.Tensor, *, block_size: int) -> torch.Tensor:
    if residual.ndim != 2:
        raise ValueError("residual must be a matrix")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    columns = residual.shape[1]
    norms = [
        torch.linalg.vector_norm(residual[:, start : start + block_size], dim=1)
        for start in range(0, columns, block_size)
    ]
    return torch.stack(norms, dim=1).contiguous()


def _block_vector_norms(vector: torch.Tensor, *, block_size: int) -> torch.Tensor:
    source = vector.detach().to("cpu", torch.float32).reshape(-1)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return torch.stack(
        [
            torch.linalg.vector_norm(source[start : start + block_size])
            for start in range(0, source.numel(), block_size)
        ]
    ).contiguous()


def _partitioned_dot_radius(
    block_residual_norms: torch.Tensor,
    vector: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    vector_norms = _block_vector_norms(vector, block_size=block_size)
    if block_residual_norms.shape[1] != vector_norms.numel():
        raise ValueError("residual metadata and vector partition do not match")
    return (block_residual_norms * vector_norms[None, :]).sum(dim=1)


def _local_silu_lipschitz(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Sound interval-local SiLU derivative bound.

    `f'(x) = sigmoid(x) + x sigmoid(x)(1-sigmoid(x))` and
    `sigmoid(x)(1-sigmoid(x)) <= 1/4`. On [l, u], therefore
    `|f'(x)| <= sigmoid(u) + max(|l|, |u|)/4`. The global proven bound is
    also valid, so taking the minimum tightens without weakening soundness.
    """

    maximum_abs = torch.maximum(lower.abs(), upper.abs())
    interval_bound = torch.sigmoid(upper) + 0.25 * maximum_abs
    return torch.minimum(
        interval_bound,
        torch.full_like(interval_bound, SILU_GLOBAL_LIPSCHITZ),
    )


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


def compile_partitioned_signed_dual_kernel(
    *,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    bits: int,
    block_size: int,
) -> PartitionedSignedDualKernel:
    gate = gate_weight.detach().to("cpu", torch.float32).contiguous()
    up = up_weight.detach().to("cpu", torch.float32).contiguous()
    down = down_weight.detach().to("cpu", torch.float32).contiguous()
    if gate.ndim != 2 or up.shape != gate.shape:
        raise ValueError("gate and up weights must be matching matrices")
    intermediate, hidden = gate.shape
    if down.shape != (hidden, intermediate):
        raise ValueError("down weight must have shape [hidden, intermediate]")

    gate_hat, gate_residual, gate_global = _symmetric_quantize_rows(gate, bits=bits)
    up_hat, up_residual, up_global = _symmetric_quantize_rows(up, bits=bits)
    down_hat_t, down_residual_t, down_global = _symmetric_quantize_rows(
        down.T,
        bits=bits,
    )
    return PartitionedSignedDualKernel(
        gate_exact=gate,
        up_exact=up,
        down_exact=down,
        gate_hat=gate_hat,
        up_hat=up_hat,
        down_hat=down_hat_t.T.contiguous(),
        gate_global_residual_norms=gate_global,
        up_global_residual_norms=up_global,
        down_global_residual_norms=down_global,
        gate_block_residual_norms=_block_row_norms(
            gate_residual,
            block_size=block_size,
        ),
        up_block_residual_norms=_block_row_norms(
            up_residual,
            block_size=block_size,
        ),
        down_block_residual_norms=_block_row_norms(
            down_residual_t,
            block_size=block_size,
        ),
        bits=bits,
        block_size=block_size,
    )


def build_partitioned_signed_dual_terms(
    kernel: PartitionedSignedDualKernel,
    *,
    activation: torch.Tensor,
    output_dual: torch.Tensor,
) -> tuple[SignedDualTerms, PartitionedConeDiagnostics]:
    x = activation.detach().to("cpu", torch.float32).reshape(-1)
    q = output_dual.detach().to("cpu", torch.float32).reshape(-1)
    if x.numel() != kernel.hidden_size or q.numel() != kernel.hidden_size:
        raise ValueError("activation and output_dual must match hidden size")

    x_norm = torch.linalg.vector_norm(x)
    q_norm = torch.linalg.vector_norm(q)
    global_gate_radius = kernel.gate_global_residual_norms * x_norm
    global_up_radius = kernel.up_global_residual_norms * x_norm
    global_directional_radius = kernel.down_global_residual_norms * q_norm

    gate_radius = _partitioned_dot_radius(
        kernel.gate_block_residual_norms,
        x,
        block_size=kernel.block_size,
    )
    up_radius = _partitioned_dot_radius(
        kernel.up_block_residual_norms,
        x,
        block_size=kernel.block_size,
    )
    directional_radius = _partitioned_dot_radius(
        kernel.down_block_residual_norms,
        q,
        block_size=kernel.block_size,
    )

    gate_hat_value = kernel.gate_hat @ x
    up_hat_value = kernel.up_hat @ x
    gate_lower = gate_hat_value - gate_radius
    gate_upper = gate_hat_value + gate_radius
    local_lipschitz = _local_silu_lipschitz(gate_lower, gate_upper)
    silu_hat = F.silu(gate_hat_value)
    activation_hat = silu_hat * up_hat_value
    activation_radius = (
        local_lipschitz
        * gate_radius
        * (up_hat_value.abs() + up_radius)
        + silu_hat.abs() * up_radius
    )

    directional_hat = kernel.down_hat.T @ q
    approximate = activation_hat * directional_hat
    lower, upper = _product_interval(
        activation_hat,
        activation_radius,
        directional_hat,
        directional_radius,
    )

    exact_activation = F.silu(kernel.gate_exact @ x) * (kernel.up_exact @ x)
    exact_directional = kernel.down_exact.T @ q
    exact = exact_activation * exact_directional

    terms = SignedDualTerms(
        exact_contributions=exact.contiguous(),
        approximate_contributions=approximate.contiguous(),
        lower_contributions=lower.contiguous(),
        upper_contributions=upper.contiguous(),
        activation_error_bounds=activation_radius.contiguous(),
        directional_error_bounds=directional_radius.contiguous(),
    )
    diagnostics = PartitionedConeDiagnostics(
        global_gate_radius_sum=float(global_gate_radius.sum().item()),
        partitioned_gate_radius_sum=float(gate_radius.sum().item()),
        global_up_radius_sum=float(global_up_radius.sum().item()),
        partitioned_up_radius_sum=float(up_radius.sum().item()),
        global_directional_radius_sum=float(global_directional_radius.sum().item()),
        partitioned_directional_radius_sum=float(directional_radius.sum().item()),
        mean_local_silu_lipschitz=float(local_lipschitz.mean().item()),
        maximum_local_silu_lipschitz=float(local_lipschitz.max().item()),
    )
    return terms, diagnostics


def partitioned_cone_metadata_budget(
    *,
    target: ModelSpec,
    block_size: int,
    metadata_bits: int = 8,
    metadata_limit_gib: float = 1.5,
) -> PartitionedConeMetadataBudget:
    if min(block_size, metadata_bits) <= 0 or metadata_limit_gib <= 0:
        raise ValueError("block size, metadata precision, and limit must be positive")
    blocks = ceil(target.hidden_size / block_size)
    # Per layer and MLP neuron: gate residual block norms, up residual block
    # norms, and down-column residual block norms.
    values = target.layers * target.intermediate_size * 3 * blocks
    metadata_gib = values * metadata_bits / 8 / GIB
    return PartitionedConeMetadataBudget(
        block_size=block_size,
        blocks_per_hidden_vector=blocks,
        metadata_bits=metadata_bits,
        metadata_values=values,
        metadata_gib=metadata_gib,
        metadata_limit_gib=metadata_limit_gib,
        metadata_pass=metadata_gib <= metadata_limit_gib,
    )
