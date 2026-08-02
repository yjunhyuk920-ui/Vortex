from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn

from vortex_runtime.feasibility import GIB, ModelSpec
from vortex_runtime.kronecker_operator import KroneckerShape, choose_kronecker_shape

BlockMode = Literal["row", "column"]


@dataclass(frozen=True)
class BlockKroneckerPlan:
    name: str
    mode: BlockMode
    block_size: int
    terms_per_block: int
    blocks: int
    shape: KroneckerShape

    @property
    def factor_elements(self) -> int:
        return self.blocks * self.terms_per_block * self.shape.factor_elements_per_term

    @property
    def flops_per_token(self) -> int:
        return self.blocks * self.terms_per_block * self.shape.flops_per_term

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mode": self.mode,
            "block_size": self.block_size,
            "terms_per_block": self.terms_per_block,
            "blocks": self.blocks,
            "shape": self.shape.to_dict(),
            "factor_elements": self.factor_elements,
            "flops_per_token": self.flops_per_token,
        }


@dataclass(frozen=True)
class BlockKroneckerBudget:
    factor_bits: int
    embedding_bits: int
    active_kv_tokens: int
    factor_elements: int
    factor_gib: float
    total_memory_gib: float
    total_traffic_gib_per_token: float
    total_flops_per_token: float
    projected_seconds_per_token: float
    allowed_seconds_per_token: float
    memory_pass: bool
    traffic_pass: bool
    latency_pass: bool
    pass_all: bool
    plans: tuple[BlockKroneckerPlan, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "factor_bits": self.factor_bits,
            "embedding_bits": self.embedding_bits,
            "active_kv_tokens": self.active_kv_tokens,
            "factor_elements": self.factor_elements,
            "factor_gib": self.factor_gib,
            "total_memory_gib": self.total_memory_gib,
            "total_traffic_gib_per_token": self.total_traffic_gib_per_token,
            "total_flops_per_token": self.total_flops_per_token,
            "projected_seconds_per_token": self.projected_seconds_per_token,
            "allowed_seconds_per_token": self.allowed_seconds_per_token,
            "memory_pass": self.memory_pass,
            "traffic_pass": self.traffic_pass,
            "latency_pass": self.latency_pass,
            "pass_all": self.pass_all,
            "plans": [plan.to_dict() for plan in self.plans],
        }


def make_block_plan(
    *,
    name: str,
    out_features: int,
    in_features: int,
    mode: BlockMode,
    block_size: int,
    terms_per_block: int,
) -> BlockKroneckerPlan:
    if block_size <= 0 or terms_per_block <= 0:
        raise ValueError("block size and terms must be positive")
    if mode == "row":
        if out_features % block_size:
            raise ValueError("row block size must divide out_features")
        blocks = out_features // block_size
        shape = choose_kronecker_shape(block_size, in_features)
    elif mode == "column":
        if in_features % block_size:
            raise ValueError("column block size must divide in_features")
        blocks = in_features // block_size
        shape = choose_kronecker_shape(out_features, block_size)
    else:
        raise ValueError(f"unsupported block mode: {mode}")
    return BlockKroneckerPlan(
        name=name,
        mode=mode,
        block_size=block_size,
        terms_per_block=terms_per_block,
        blocks=blocks,
        shape=shape,
    )


def llama_block_kronecker_plans(
    model: ModelSpec,
    *,
    attention_terms: int = 4,
    mlp_terms: int = 3,
    lm_head_terms: int = 2,
    mlp_block_size: int = 128,
    lm_head_block_size: int = 256,
) -> tuple[BlockKroneckerPlan, ...]:
    hidden = model.hidden_size
    head_dim = model.head_dim
    kv = model.kv_dim
    intermediate = model.intermediate_size
    return (
        make_block_plan(name="q_proj", out_features=hidden, in_features=hidden, mode="row", block_size=head_dim, terms_per_block=attention_terms),
        make_block_plan(name="k_proj", out_features=kv, in_features=hidden, mode="row", block_size=head_dim, terms_per_block=attention_terms),
        make_block_plan(name="v_proj", out_features=kv, in_features=hidden, mode="row", block_size=head_dim, terms_per_block=attention_terms),
        make_block_plan(name="o_proj", out_features=hidden, in_features=hidden, mode="column", block_size=head_dim, terms_per_block=attention_terms),
        make_block_plan(name="gate_proj", out_features=intermediate, in_features=hidden, mode="row", block_size=mlp_block_size, terms_per_block=mlp_terms),
        make_block_plan(name="up_proj", out_features=intermediate, in_features=hidden, mode="row", block_size=mlp_block_size, terms_per_block=mlp_terms),
        make_block_plan(name="down_proj", out_features=hidden, in_features=intermediate, mode="column", block_size=mlp_block_size, terms_per_block=mlp_terms),
        make_block_plan(name="lm_head", out_features=model.vocab_size, in_features=hidden, mode="row", block_size=lm_head_block_size, terms_per_block=lm_head_terms),
    )


def block_kronecker_budget(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    factor_bits: int = 8,
    embedding_bits: int = 4,
    active_kv_tokens: int = 256,
    attention_terms: int = 4,
    mlp_terms: int = 3,
    lm_head_terms: int = 2,
    mlp_block_size: int = 128,
    lm_head_block_size: int = 256,
    workspace_gib: float = 1.5,
    allocator_reserve_gib: float = 1.0,
    misc_traffic_gib_per_token: float = 0.05,
    misc_flops_per_token: float = 1.0e9,
    resident_hbm_gib_s: float = 300.0,
    effective_tops: float = 160.0,
    target_ratio: float = 1.2,
    traffic_ratio: float = 1.2,
) -> BlockKroneckerBudget:
    plans = llama_block_kronecker_plans(
        target,
        attention_terms=attention_terms,
        mlp_terms=mlp_terms,
        lm_head_terms=lm_head_terms,
        mlp_block_size=mlp_block_size,
        lm_head_block_size=lm_head_block_size,
    )
    layer_plans = plans[:-1]
    lm_plan = plans[-1]
    factor_elements = target.layers * sum(p.factor_elements for p in layer_plans) + lm_plan.factor_elements
    factor_gib = factor_elements * factor_bits / 8 / GIB
    embedding_gib = target.vocab_size * target.hidden_size * embedding_bits / 8 / GIB
    active_kv_gib = target.layers * active_kv_tokens * 2 * target.kv_dim * target.kv_bits / 8 / GIB
    norm_gib = (2 * target.layers + 1) * target.hidden_size * 16 / 8 / GIB
    total_memory = factor_gib + embedding_gib + active_kv_gib + norm_gib + workspace_gib + allocator_reserve_gib
    embedding_row_gib = target.hidden_size * embedding_bits / 8 / GIB
    total_traffic = factor_gib + active_kv_gib + embedding_row_gib + misc_traffic_gib_per_token
    traffic_limit = traffic_ratio * (baseline.weight_bytes / GIB + baseline.kv_bytes / GIB)
    factor_flops = target.layers * sum(p.flops_per_token for p in layer_plans) + lm_plan.flops_per_token
    attention_flops = 4.0 * target.layers * target.hidden_size * active_kv_tokens
    total_flops = factor_flops + attention_flops + misc_flops_per_token
    projected_seconds = max(factor_gib / resident_hbm_gib_s, total_flops / (effective_tops * 1e12))
    baseline_seconds = max(
        baseline.weight_bytes / GIB / resident_hbm_gib_s,
        (baseline.dense_linear_flops_per_token + baseline.dense_attention_flops_per_token) / (40.0 * 1e12),
    )
    allowed_seconds = target_ratio * baseline_seconds
    memory_pass = total_memory <= 8.0
    traffic_pass = total_traffic <= traffic_limit
    latency_pass = projected_seconds <= allowed_seconds
    return BlockKroneckerBudget(
        factor_bits=factor_bits,
        embedding_bits=embedding_bits,
        active_kv_tokens=active_kv_tokens,
        factor_elements=factor_elements,
        factor_gib=factor_gib,
        total_memory_gib=total_memory,
        total_traffic_gib_per_token=total_traffic,
        total_flops_per_token=total_flops,
        projected_seconds_per_token=projected_seconds,
        allowed_seconds_per_token=allowed_seconds,
        memory_pass=memory_pass,
        traffic_pass=traffic_pass,
        latency_pass=latency_pass,
        pass_all=memory_pass and traffic_pass and latency_pass,
        plans=plans,
    )


def _fake_quantize_blocks(tensor: torch.Tensor, *, bits: int) -> torch.Tensor:
    if not 2 <= bits <= 16:
        raise ValueError("factor bits must be in [2, 16]")
    levels = (1 << (bits - 1)) - 1
    maximum = tensor.abs().amax(dim=(-2, -1), keepdim=True)
    scale = torch.where(maximum > 0, maximum / levels, torch.ones_like(maximum))
    return torch.clamp(torch.round(tensor / scale), -levels, levels) * scale


def _rank_one_block_factors(
    blocks: torch.Tensor,
    *,
    shape: KroneckerShape,
    factor_bits: int,
    power_iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    block_count = blocks.shape[0]
    source = blocks.to("cpu", torch.float32)
    rearranged = source.reshape(
        block_count,
        shape.out_first,
        shape.out_second,
        shape.in_first,
        shape.in_second,
    ).permute(0, 1, 3, 2, 4).reshape(
        block_count,
        shape.out_first * shape.in_first,
        shape.out_second * shape.in_second,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    right = torch.randn(block_count, rearranged.shape[2], 1, generator=generator)
    right = right / torch.linalg.vector_norm(right, dim=1, keepdim=True).clamp_min(1e-12)
    for _ in range(max(1, power_iterations + 1)):
        left = rearranged @ right
        left = left / torch.linalg.vector_norm(left, dim=1, keepdim=True).clamp_min(1e-12)
        right = rearranged.transpose(1, 2) @ left
        right = right / torch.linalg.vector_norm(right, dim=1, keepdim=True).clamp_min(1e-12)
    signed_value = (left.transpose(1, 2) @ rearranged @ right).reshape(block_count)
    root = torch.sqrt(signed_value.abs().clamp_min(0))
    sign = torch.sign(signed_value)
    first = (left.squeeze(-1) * root.unsqueeze(1)).reshape(block_count, 1, shape.out_first, shape.in_first)
    second = (right.squeeze(-1) * (root * sign).unsqueeze(1)).reshape(block_count, 1, shape.out_second, shape.in_second)
    first = _fake_quantize_blocks(first, bits=factor_bits)
    second = _fake_quantize_blocks(second, bits=factor_bits)
    reconstructed = torch.einsum("ptai,ptcj->pacij", first, second).reshape_as(source)
    relative = torch.linalg.vector_norm(source - reconstructed) / torch.linalg.vector_norm(source).clamp_min(1e-12)
    return first.contiguous(), second.contiguous(), float(relative.item())


class BlockKroneckerLinear(nn.Module):
    def __init__(
        self,
        *,
        mode: BlockMode,
        first_factors: torch.Tensor,
        second_factors: torch.Tensor,
        shape: KroneckerShape,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.shape = shape
        self.register_buffer("first_factors", first_factors.contiguous())
        self.register_buffer("second_factors", second_factors.contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    @property
    def blocks(self) -> int:
        return int(self.first_factors.shape[0])

    @property
    def out_features(self) -> int:
        return self.blocks * self.shape.out_features if self.mode == "row" else self.shape.out_features

    @property
    def in_features(self) -> int:
        return self.blocks * self.shape.in_features if self.mode == "column" else self.shape.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.first_factors.to(device=x.device, dtype=x.dtype)
        second = self.second_factors.to(device=x.device, dtype=x.dtype)
        leading = x.shape[:-1]
        if self.mode == "row":
            flat = x.reshape(-1, self.shape.in_first, self.shape.in_second)
            output = torch.einsum("ptai,nij,ptcj->npac", first, flat, second).reshape(*leading, self.out_features)
        else:
            flat = x.reshape(-1, self.blocks, self.shape.in_first, self.shape.in_second)
            per_block = torch.einsum("ptai,npij,ptcj->npac", first, flat, second)
            output = per_block.sum(dim=1).reshape(*leading, self.out_features)
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        return output


def fit_block_kronecker_linear(
    linear: nn.Linear,
    *,
    mode: BlockMode,
    block_size: int,
    factor_bits: int = 8,
    power_iterations: int = 2,
    seed: int = 0,
) -> tuple[BlockKroneckerLinear, dict[str, int | float | str]]:
    weight = linear.weight.detach().to("cpu", torch.float32)
    if mode == "row":
        if weight.shape[0] % block_size:
            raise ValueError("row block size must divide output features")
        blocks = weight.reshape(-1, block_size, weight.shape[1])
        shape = choose_kronecker_shape(block_size, weight.shape[1])
    elif mode == "column":
        if weight.shape[1] % block_size:
            raise ValueError("column block size must divide input features")
        blocks = weight.reshape(weight.shape[0], -1, block_size).permute(1, 0, 2).contiguous()
        shape = choose_kronecker_shape(weight.shape[0], block_size)
    else:
        raise ValueError(f"unsupported block mode: {mode}")
    first, second, relative_error = _rank_one_block_factors(
        blocks,
        shape=shape,
        factor_bits=factor_bits,
        power_iterations=power_iterations,
        seed=seed,
    )
    module = BlockKroneckerLinear(
        mode=mode,
        first_factors=first,
        second_factors=second,
        shape=shape,
        bias=linear.bias,
    )
    factor_elements = first.numel() + second.numel()
    return module, {
        "mode": mode,
        "block_size": block_size,
        "blocks": int(blocks.shape[0]),
        "terms_per_block": 1,
        "original_elements": int(weight.numel()),
        "factor_elements": int(factor_elements),
        "compression_ratio": weight.numel() / factor_elements,
        "relative_l2_error": relative_error,
    }
