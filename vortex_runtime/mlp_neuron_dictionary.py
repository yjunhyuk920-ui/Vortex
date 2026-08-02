from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from vortex_runtime.feasibility import GIB, ModelSpec
from vortex_runtime.progressive_precision import symmetric_per_row_fake_quantize


@dataclass(frozen=True)
class MLPDictionaryBudget:
    prototypes_per_layer: int
    factor_bits: int
    factor_elements: int
    factor_gib: float
    factor_traffic_gib_per_token: float
    factor_flops_per_token: float
    baseline_traffic_limit_gib_per_token: float
    remaining_traffic_gib_per_token: float
    baseline_allowed_seconds_per_token: float
    factor_hbm_seconds_per_token: float
    factor_compute_seconds_per_token: float
    memory_pass: bool
    partial_traffic_pass: bool
    partial_latency_pass: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class MLPDictionaryFitStats:
    neurons: int
    prototypes: int
    projection_dim: int
    iterations: int
    factor_bits: int
    gate_up_relative_l2_error: float
    gate_relative_l2_error: float
    up_relative_l2_error: float
    minimum_cluster_size: int
    maximum_cluster_size: int
    mean_cluster_size: float
    empty_clusters: int
    factor_elements: int
    factor_bytes: int

    @property
    def compression_ratio(self) -> float:
        original_elements = self.neurons * 3
        prototype_equivalent = self.prototypes * 3
        return original_elements / max(1, prototype_equivalent)

    def to_dict(self) -> dict[str, int | float]:
        payload = asdict(self)
        payload["compression_ratio"] = self.compression_ratio
        return payload


def mlp_neuron_dictionary_budget(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    prototypes_per_layer: int,
    factor_bits: int = 8,
    resident_hbm_gib_s: float = 300.0,
    effective_tops: float = 160.0,
    traffic_ratio: float = 1.2,
    target_ratio: float = 1.2,
    workspace_gib: float = 1.0,
    allocator_reserve_gib: float = 1.0,
    memory_limit_gib: float = 8.0,
) -> MLPDictionaryBudget:
    """Budget a per-layer compiled SwiGLU neuron dictionary.

    Each prototype stores one gate row, one up row and one aggregated down
    column. All prototypes are read exactly once per layer and token. Attention,
    embeddings, KV and LM-head cost are intentionally omitted and exposed as
    remaining traffic headroom; this is a partial-family Gate, not a complete
    architecture pass.
    """

    if prototypes_per_layer <= 0 or factor_bits <= 0:
        raise ValueError("prototype count and precision must be positive")
    positive = (
        resident_hbm_gib_s,
        effective_tops,
        traffic_ratio,
        target_ratio,
        memory_limit_gib,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("hardware and Gate values must be positive")

    factor_elements = (
        target.layers
        * prototypes_per_layer
        * 3
        * target.hidden_size
    )
    factor_gib = factor_elements * factor_bits / 8 / GIB
    factor_flops = (
        6.0
        * target.layers
        * prototypes_per_layer
        * target.hidden_size
    )

    baseline_traffic = baseline.weight_bytes / GIB + baseline.kv_bytes / GIB
    traffic_limit = traffic_ratio * baseline_traffic
    remaining_traffic = traffic_limit - factor_gib

    baseline_weight_seconds = baseline.weight_bytes / GIB / resident_hbm_gib_s
    baseline_ops = (
        baseline.dense_linear_flops_per_token
        + baseline.dense_attention_flops_per_token
    )
    baseline_compute_seconds = baseline_ops / (40.0 * 1e12)
    baseline_seconds = max(baseline_weight_seconds, baseline_compute_seconds)
    allowed_seconds = target_ratio * baseline_seconds
    hbm_seconds = factor_gib / resident_hbm_gib_s
    compute_seconds = factor_flops / (effective_tops * 1e12)
    resident_memory = factor_gib + workspace_gib + allocator_reserve_gib

    return MLPDictionaryBudget(
        prototypes_per_layer=prototypes_per_layer,
        factor_bits=factor_bits,
        factor_elements=factor_elements,
        factor_gib=factor_gib,
        factor_traffic_gib_per_token=factor_gib,
        factor_flops_per_token=factor_flops,
        baseline_traffic_limit_gib_per_token=traffic_limit,
        remaining_traffic_gib_per_token=remaining_traffic,
        baseline_allowed_seconds_per_token=allowed_seconds,
        factor_hbm_seconds_per_token=hbm_seconds,
        factor_compute_seconds_per_token=compute_seconds,
        memory_pass=resident_memory <= memory_limit_gib,
        partial_traffic_pass=remaining_traffic > 0,
        partial_latency_pass=max(hbm_seconds, compute_seconds) <= allowed_seconds,
    )


def _relative_l2(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(reference - estimate)
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _random_sign_projection(
    features: torch.Tensor,
    *,
    projection_dim: int,
    seed: int,
    chunk_size: int = 1024,
) -> torch.Tensor:
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randint(
        0,
        2,
        (features.shape[1], projection_dim),
        generator=generator,
        dtype=torch.int8,
    ).to(torch.float32)
    projection.mul_(2).sub_(1).div_(sqrt(projection_dim))
    rows: list[torch.Tensor] = []
    for start in range(0, features.shape[0], chunk_size):
        rows.append(features[start : start + chunk_size] @ projection)
    return torch.cat(rows, dim=0)


def _squared_distance(
    points: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    point_norm = points.square().sum(dim=1, keepdim=True)
    center_norm = centers.square().sum(dim=1).unsqueeze(0)
    return torch.clamp(
        point_norm + center_norm - 2.0 * points @ centers.T,
        min=0.0,
    )


def _initialize_farthest(
    points: torch.Tensor,
    *,
    clusters: int,
) -> torch.Tensor:
    first = int(torch.argmax(points.square().sum(dim=1)).item())
    indices = [first]
    minimum_distance = _squared_distance(points, points[first : first + 1]).squeeze(1)
    for _ in range(1, clusters):
        index = int(torch.argmax(minimum_distance).item())
        indices.append(index)
        candidate = _squared_distance(points, points[index : index + 1]).squeeze(1)
        minimum_distance = torch.minimum(minimum_distance, candidate)
    return torch.tensor(indices, dtype=torch.long)


def _cluster_projected_features(
    points: torch.Tensor,
    *,
    clusters: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not 0 < clusters <= points.shape[0]:
        raise ValueError("clusters must be in [1, number of points]")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    center_indices = _initialize_farthest(points, clusters=clusters)
    centers = points[center_indices].clone()
    assignments = torch.zeros(points.shape[0], dtype=torch.long)
    empty_total = 0
    for _ in range(iterations):
        distances = _squared_distance(points, centers)
        assignments = torch.argmin(distances, dim=1)
        counts = torch.bincount(assignments, minlength=clusters)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, points)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        empty = torch.nonzero(~nonempty, as_tuple=False).reshape(-1)
        empty_total += int(empty.numel())
        if empty.numel():
            nearest = distances.gather(1, assignments.unsqueeze(1)).squeeze(1)
            replacement_order = torch.argsort(nearest, descending=True)
            for position, cluster_index in enumerate(empty.tolist()):
                point_index = int(replacement_order[position].item())
                centers[cluster_index] = points[point_index]
                assignments[point_index] = cluster_index
    return assignments, centers, empty_total


class CompiledSwiGLUDictionary(nn.Module):
    def __init__(
        self,
        *,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        if gate_weight.shape != up_weight.shape:
            raise ValueError("gate and up prototype shapes must match")
        if down_weight.shape != (gate_weight.shape[1], gate_weight.shape[0]):
            raise ValueError("down weight must have shape [hidden, prototypes]")
        self.register_buffer("gate_weight", gate_weight.contiguous())
        self.register_buffer("up_weight", up_weight.contiguous())
        self.register_buffer("down_weight", down_weight.contiguous())

    @property
    def hidden_size(self) -> int:
        return int(self.gate_weight.shape[1])

    @property
    def prototypes(self) -> int:
        return int(self.gate_weight.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, self.gate_weight)
        up = F.linear(x, self.up_weight)
        activated = F.silu(gate) * up
        return F.linear(activated, self.down_weight)


def compile_swiglu_dictionary(
    *,
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
    down_proj: nn.Linear,
    prototypes: int,
    projection_dim: int = 64,
    iterations: int = 5,
    factor_bits: int = 8,
    seed: int = 0,
) -> tuple[CompiledSwiGLUDictionary, MLPDictionaryFitStats]:
    """Compile one exact-permutation-invariant SwiGLU neuron dictionary.

    Gate and up rows define the neuron input function. Neurons assigned to the
    same prototype share the centroid input function, while their original down
    columns are summed exactly. This converts permutation-equivalent neuron sets
    into a smaller executable nonlinear program.
    """

    gate = gate_proj.weight.detach().to("cpu", torch.float32)
    up = up_proj.weight.detach().to("cpu", torch.float32)
    down = down_proj.weight.detach().to("cpu", torch.float32)
    if gate.shape != up.shape:
        raise ValueError("gate and up weights must match")
    neurons, hidden = gate.shape
    if down.shape != (hidden, neurons):
        raise ValueError("down weight must have shape [hidden, neurons]")
    if not 0 < prototypes <= neurons:
        raise ValueError("prototypes must be in [1, neurons]")

    features = torch.cat((gate, up), dim=1)
    projected = _random_sign_projection(
        features,
        projection_dim=projection_dim,
        seed=seed,
    )
    assignments, _, empty_total = _cluster_projected_features(
        projected,
        clusters=prototypes,
        iterations=iterations,
    )
    counts = torch.bincount(assignments, minlength=prototypes)

    gate_sum = torch.zeros(prototypes, hidden, dtype=torch.float32)
    up_sum = torch.zeros_like(gate_sum)
    gate_sum.index_add_(0, assignments, gate)
    up_sum.index_add_(0, assignments, up)
    denominator = counts.clamp_min(1).to(torch.float32).unsqueeze(1)
    gate_centers = gate_sum / denominator
    up_centers = up_sum / denominator

    down_aggregated = torch.zeros(hidden, prototypes, dtype=torch.float32)
    down_aggregated.index_add_(1, assignments, down)

    reconstructed_gate = gate_centers[assignments]
    reconstructed_up = up_centers[assignments]
    reconstructed_features = torch.cat((reconstructed_gate, reconstructed_up), dim=1)

    quantized_gate, _ = symmetric_per_row_fake_quantize(
        gate_centers,
        bits=factor_bits,
        source_bits=16,
        name="compiled_swiglu.gate",
        row_chunk=max(1, prototypes),
    )
    quantized_up, _ = symmetric_per_row_fake_quantize(
        up_centers,
        bits=factor_bits,
        source_bits=16,
        name="compiled_swiglu.up",
        row_chunk=max(1, prototypes),
    )
    quantized_down, _ = symmetric_per_row_fake_quantize(
        down_aggregated,
        bits=factor_bits,
        source_bits=16,
        name="compiled_swiglu.down",
        row_chunk=128,
    )

    module = CompiledSwiGLUDictionary(
        gate_weight=quantized_gate,
        up_weight=quantized_up,
        down_weight=quantized_down,
    )
    factor_elements = quantized_gate.numel() + quantized_up.numel() + quantized_down.numel()
    stats = MLPDictionaryFitStats(
        neurons=neurons,
        prototypes=prototypes,
        projection_dim=projection_dim,
        iterations=iterations,
        factor_bits=factor_bits,
        gate_up_relative_l2_error=_relative_l2(features, reconstructed_features),
        gate_relative_l2_error=_relative_l2(gate, reconstructed_gate),
        up_relative_l2_error=_relative_l2(up, reconstructed_up),
        minimum_cluster_size=int(counts.min().item()),
        maximum_cluster_size=int(counts.max().item()),
        mean_cluster_size=float(counts.to(torch.float32).mean().item()),
        empty_clusters=empty_total,
        factor_elements=factor_elements,
        factor_bytes=(factor_elements * factor_bits + 7) // 8,
    )
    return module, stats
