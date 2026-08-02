from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isqrt
from typing import Iterable

import torch
from torch import nn

from vortex_runtime.feasibility import GIB, ModelSpec


@dataclass(frozen=True)
class KroneckerShape:
    out_first: int
    out_second: int
    in_first: int
    in_second: int

    @property
    def out_features(self) -> int:
        return self.out_first * self.out_second

    @property
    def in_features(self) -> int:
        return self.in_first * self.in_second

    @property
    def factor_elements_per_term(self) -> int:
        return (
            self.out_first * self.in_first
            + self.out_second * self.in_second
        )

    @property
    def flops_per_term(self) -> int:
        # Y = A @ X @ B.T, where X is [in_first, in_second].
        return 2 * (
            self.out_first * self.in_first * self.in_second
            + self.out_first * self.in_second * self.out_second
        )

    def to_dict(self) -> dict[str, int]:
        payload = asdict(self)
        payload["out_features"] = self.out_features
        payload["in_features"] = self.in_features
        payload["factor_elements_per_term"] = self.factor_elements_per_term
        payload["flops_per_term"] = self.flops_per_term
        return payload


@dataclass(frozen=True)
class KroneckerOperatorBudget:
    rank: int
    factor_bits: int
    embedding_bits: int
    active_kv_tokens: int
    factor_elements: int
    factor_gib: float
    embedding_storage_gib: float
    active_kv_gib: float
    norm_gib: float
    workspace_gib: float
    allocator_reserve_gib: float
    total_memory_gib: float
    memory_limit_gib: float
    factor_traffic_gib_per_token: float
    kv_traffic_gib_per_token: float
    misc_traffic_gib_per_token: float
    total_traffic_gib_per_token: float
    traffic_limit_gib_per_token: float
    kronecker_flops_per_token: float
    attention_flops_per_token: float
    misc_flops_per_token: float
    total_flops_per_token: float
    factor_seconds_per_token: float
    compute_seconds_per_token: float
    projected_seconds_per_token: float
    baseline_seconds_per_token: float
    allowed_seconds_per_token: float
    memory_pass: bool
    traffic_pass: bool
    latency_pass: bool
    pass_all: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def balanced_factor_pair(size: int) -> tuple[int, int]:
    if size <= 0:
        raise ValueError("size must be positive")
    for first in range(isqrt(size), 0, -1):
        if size % first == 0:
            return first, size // first
    raise RuntimeError("positive integer must have a factor pair")


def choose_kronecker_shape(
    out_features: int,
    in_features: int,
) -> KroneckerShape:
    """Choose balanced tensor dimensions and the cheapest orientation.

    Balanced pairs avoid the degenerate `1 x N` case, which collapses a
    Kronecker term back into an ordinary rank-one matrix and loses the intended
    high-rank structured expressivity.
    """

    out_pair = balanced_factor_pair(out_features)
    in_pair = balanced_factor_pair(in_features)
    candidates: list[KroneckerShape] = []
    for out_first, out_second in (out_pair, out_pair[::-1]):
        for in_first, in_second in (in_pair, in_pair[::-1]):
            candidates.append(
                KroneckerShape(
                    out_first=out_first,
                    out_second=out_second,
                    in_first=in_first,
                    in_second=in_second,
                )
            )
    return min(
        candidates,
        key=lambda item: (
            item.factor_elements_per_term,
            item.flops_per_term,
            max(
                item.out_first * item.in_first,
                item.out_second * item.in_second,
            ),
        ),
    )


def llama_linear_shapes(model: ModelSpec) -> tuple[tuple[int, int], ...]:
    hidden = model.hidden_size
    intermediate = model.intermediate_size
    kv = model.kv_dim
    return (
        (hidden, hidden),
        (kv, hidden),
        (kv, hidden),
        (hidden, hidden),
        (intermediate, hidden),
        (intermediate, hidden),
        (hidden, intermediate),
    )


def kronecker_operator_budget(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    rank: int = 64,
    factor_bits: int = 8,
    embedding_bits: int = 4,
    active_kv_tokens: int = 256,
    workspace_gib: float = 1.5,
    allocator_reserve_gib: float = 1.0,
    misc_traffic_gib_per_token: float = 0.05,
    misc_flops_per_token: float = 1.0e9,
    memory_limit_gib: float = 8.0,
    traffic_ratio: float = 1.2,
    target_ratio: float = 1.2,
    resident_hbm_gib_s: float = 300.0,
    effective_tops: float = 160.0,
    baseline_memory_gib_s: float = 300.0,
    baseline_effective_tflops: float = 40.0,
) -> KroneckerOperatorBudget:
    """Budget a fully factorized Llama-like 405B execution path.

    Every dense linear, including the LM head, is executed directly from a sum
    of Kronecker products. No full-size representative matrix is read at decode
    time. The embedding table is stored at low precision but only one row is
    read per input token.
    """

    if rank <= 0:
        raise ValueError("rank must be positive")
    if factor_bits <= 0 or embedding_bits <= 0:
        raise ValueError("storage precision must be positive")
    if active_kv_tokens <= 0:
        raise ValueError("active_kv_tokens must be positive")
    positive = (
        memory_limit_gib,
        traffic_ratio,
        target_ratio,
        resident_hbm_gib_s,
        effective_tops,
        baseline_memory_gib_s,
        baseline_effective_tflops,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("limits, ratios and hardware values must be positive")

    layer_shapes = [
        choose_kronecker_shape(out_features, in_features)
        for out_features, in_features in llama_linear_shapes(target)
    ]
    lm_head_shape = choose_kronecker_shape(
        target.vocab_size,
        target.hidden_size,
    )
    layer_elements_per_term = sum(
        shape.factor_elements_per_term for shape in layer_shapes
    )
    layer_flops_per_term = sum(shape.flops_per_term for shape in layer_shapes)
    factor_elements = rank * (
        target.layers * layer_elements_per_term
        + lm_head_shape.factor_elements_per_term
    )
    factor_gib = factor_elements * factor_bits / 8 / GIB

    embedding_storage_gib = (
        target.vocab_size
        * target.hidden_size
        * embedding_bits
        / 8
        / GIB
    )
    active_kv_gib = (
        target.layers
        * active_kv_tokens
        * 2
        * target.kv_dim
        * target.kv_bits
        / 8
        / GIB
    )
    norm_gib = (
        (2 * target.layers + 1)
        * target.hidden_size
        * 16
        / 8
        / GIB
    )
    total_memory_gib = (
        factor_gib
        + embedding_storage_gib
        + active_kv_gib
        + norm_gib
        + workspace_gib
        + allocator_reserve_gib
    )

    embedding_row_gib = target.hidden_size * embedding_bits / 8 / GIB
    factor_traffic = factor_gib
    kv_traffic = active_kv_gib
    total_traffic = (
        factor_traffic
        + kv_traffic
        + embedding_row_gib
        + misc_traffic_gib_per_token
    )
    baseline_traffic = baseline.weight_bytes / GIB + baseline.kv_bytes / GIB
    traffic_limit = traffic_ratio * baseline_traffic

    kronecker_flops = rank * (
        target.layers * layer_flops_per_term
        + lm_head_shape.flops_per_term
    )
    attention_flops = (
        4.0
        * target.layers
        * target.hidden_size
        * active_kv_tokens
    )
    total_flops = kronecker_flops + attention_flops + misc_flops_per_token
    factor_seconds = factor_traffic / resident_hbm_gib_s
    compute_seconds = total_flops / (effective_tops * 1e12)
    projected_seconds = max(factor_seconds, compute_seconds)

    baseline_weight_seconds = baseline.weight_bytes / GIB / baseline_memory_gib_s
    baseline_ops = (
        baseline.dense_linear_flops_per_token
        + baseline.dense_attention_flops_per_token
    )
    baseline_compute_seconds = baseline_ops / (
        baseline_effective_tflops * 1e12
    )
    baseline_seconds = max(baseline_weight_seconds, baseline_compute_seconds)
    allowed_seconds = target_ratio * baseline_seconds

    memory_pass = total_memory_gib <= memory_limit_gib
    traffic_pass = total_traffic <= traffic_limit
    latency_pass = projected_seconds <= allowed_seconds
    return KroneckerOperatorBudget(
        rank=rank,
        factor_bits=factor_bits,
        embedding_bits=embedding_bits,
        active_kv_tokens=active_kv_tokens,
        factor_elements=factor_elements,
        factor_gib=factor_gib,
        embedding_storage_gib=embedding_storage_gib,
        active_kv_gib=active_kv_gib,
        norm_gib=norm_gib,
        workspace_gib=workspace_gib,
        allocator_reserve_gib=allocator_reserve_gib,
        total_memory_gib=total_memory_gib,
        memory_limit_gib=memory_limit_gib,
        factor_traffic_gib_per_token=factor_traffic,
        kv_traffic_gib_per_token=kv_traffic,
        misc_traffic_gib_per_token=misc_traffic_gib_per_token,
        total_traffic_gib_per_token=total_traffic,
        traffic_limit_gib_per_token=traffic_limit,
        kronecker_flops_per_token=kronecker_flops,
        attention_flops_per_token=attention_flops,
        misc_flops_per_token=misc_flops_per_token,
        total_flops_per_token=total_flops,
        factor_seconds_per_token=factor_seconds,
        compute_seconds_per_token=compute_seconds,
        projected_seconds_per_token=projected_seconds,
        baseline_seconds_per_token=baseline_seconds,
        allowed_seconds_per_token=allowed_seconds,
        memory_pass=memory_pass,
        traffic_pass=traffic_pass,
        latency_pass=latency_pass,
        pass_all=memory_pass and traffic_pass and latency_pass,
    )


def _fake_quantize_tensor(tensor: torch.Tensor, *, bits: int) -> torch.Tensor:
    if not 2 <= bits <= 16:
        raise ValueError("factor bits must be in [2, 16]")
    levels = (1 << (bits - 1)) - 1
    maximum = tensor.abs().amax()
    if float(maximum.item()) == 0.0:
        return tensor.clone()
    scale = maximum / levels
    return torch.clamp(torch.round(tensor / scale), -levels, levels) * scale


@dataclass(frozen=True)
class KroneckerFitStats:
    out_features: int
    in_features: int
    rank: int
    factor_bits: int
    original_elements: int
    factor_elements: int
    relative_l2_error: float
    maximum_absolute_error: float

    @property
    def compression_ratio(self) -> float:
        return self.original_elements / max(1, self.factor_elements)

    def to_dict(self) -> dict[str, int | float]:
        payload = asdict(self)
        payload["compression_ratio"] = self.compression_ratio
        return payload


def _randomized_svd(
    matrix: torch.Tensor,
    *,
    rank: int,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, cols = matrix.shape
    sample_rank = min(rows, cols, rank + oversample)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    omega = torch.randn(cols, sample_rank, generator=generator)
    sample = matrix @ omega
    for _ in range(power_iterations):
        sample = matrix @ (matrix.T @ sample)
    basis = torch.linalg.qr(sample, mode="reduced").Q
    small = basis.T @ matrix
    left, singular_values, right_transpose = torch.linalg.svd(
        small,
        full_matrices=False,
    )
    effective_rank = min(rank, singular_values.numel())
    return (
        basis @ left[:, :effective_rank],
        singular_values[:effective_rank],
        right_transpose[:effective_rank, :],
    )


def fit_kronecker_sum(
    weight: torch.Tensor,
    *,
    rank: int,
    factor_bits: int = 8,
    oversample: int = 4,
    power_iterations: int = 1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, KroneckerShape, KroneckerFitStats]:
    """Fit `weight ~= sum_k kron(A_k, B_k)` using rearranged SVD."""

    if weight.ndim != 2:
        raise ValueError("weight must be two-dimensional")
    out_features, in_features = weight.shape
    shape = choose_kronecker_shape(out_features, in_features)
    source = weight.detach().to("cpu", torch.float32)
    rearranged = source.reshape(
        shape.out_first,
        shape.out_second,
        shape.in_first,
        shape.in_second,
    ).permute(0, 2, 1, 3).reshape(
        shape.out_first * shape.in_first,
        shape.out_second * shape.in_second,
    )
    left, singular_values, right_transpose = _randomized_svd(
        rearranged,
        rank=rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed,
    )
    roots = torch.sqrt(torch.clamp(singular_values, min=0))
    first = (
        left * roots.reshape(1, -1)
    ).T.reshape(-1, shape.out_first, shape.in_first)
    second = (
        right_transpose.T * roots.reshape(1, -1)
    ).T.reshape(-1, shape.out_second, shape.in_second)
    first = torch.stack(
        [_fake_quantize_tensor(item, bits=factor_bits) for item in first]
    )
    second = torch.stack(
        [_fake_quantize_tensor(item, bits=factor_bits) for item in second]
    )
    reconstructed = materialize_kronecker_sum(
        first_factors=first,
        second_factors=second,
        shape=shape,
    )
    numerator = torch.linalg.vector_norm(source - reconstructed)
    denominator = torch.linalg.vector_norm(source).clamp_min(1e-12)
    stats = KroneckerFitStats(
        out_features=out_features,
        in_features=in_features,
        rank=int(first.shape[0]),
        factor_bits=factor_bits,
        original_elements=source.numel(),
        factor_elements=first.numel() + second.numel(),
        relative_l2_error=float((numerator / denominator).item()),
        maximum_absolute_error=float((source - reconstructed).abs().max().item()),
    )
    return first.contiguous(), second.contiguous(), shape, stats


def materialize_kronecker_sum(
    *,
    first_factors: torch.Tensor,
    second_factors: torch.Tensor,
    shape: KroneckerShape,
) -> torch.Tensor:
    if first_factors.ndim != 3 or second_factors.ndim != 3:
        raise ValueError("Kronecker factors must be rank-three tensors")
    if first_factors.shape[0] != second_factors.shape[0]:
        raise ValueError("Kronecker factor counts must match")
    rearranged = torch.einsum(
        "kab,kcd->acbd",
        first_factors,
        second_factors,
    )
    return rearranged.reshape(shape.out_features, shape.in_features)


class KroneckerLinear(nn.Module):
    def __init__(
        self,
        *,
        first_factors: torch.Tensor,
        second_factors: torch.Tensor,
        shape: KroneckerShape,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.register_buffer("first_factors", first_factors.contiguous())
        self.register_buffer("second_factors", second_factors.contiguous())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())

    @property
    def in_features(self) -> int:
        return self.shape.in_features

    @property
    def out_features(self) -> int:
        return self.shape.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError("input feature dimension mismatch")
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.shape.in_first, self.shape.in_second)
        first = self.first_factors.to(device=x.device, dtype=x.dtype)
        second = self.second_factors.to(device=x.device, dtype=x.dtype)
        # Per term: A @ X @ B.T. Sum terms before restoring leading dims.
        output = torch.einsum("kai,nij,kbj->nab", first, flat, second)
        output = output.reshape(*original_shape, self.out_features)
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        return output


def replace_linears_with_kronecker(
    model: nn.Module,
    *,
    rank: int,
    factor_bits: int = 8,
    oversample: int = 4,
    power_iterations: int = 1,
    seed: int = 0,
    suffixes: Iterable[str] | None = None,
) -> dict[str, KroneckerFitStats]:
    selected = None if suffixes is None else tuple(suffixes)
    matches: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if selected is not None and not any(name.endswith(item) for item in selected):
            continue
        matches.append((name, module))

    result: dict[str, KroneckerFitStats] = {}
    for index, (name, linear) in enumerate(matches):
        first, second, shape, stats = fit_kronecker_sum(
            linear.weight,
            rank=rank,
            factor_bits=factor_bits,
            oversample=oversample,
            power_iterations=power_iterations,
            seed=seed + index * 1009,
        )
        parts = name.split(".")
        parent: nn.Module = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(
            parent,
            parts[-1],
            KroneckerLinear(
                first_factors=first,
                second_factors=second,
                shape=shape,
                bias=linear.bias,
            ),
        )
        result[name] = stats
    return result
