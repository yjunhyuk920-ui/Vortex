from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from vortex_runtime.mlp_heavy_hitter import OracleHeavyHitterSwiGLU


@dataclass(frozen=True)
class AdjointAllocation:
    total_neurons: int
    minimum_per_layer: int
    layer_counts: tuple[int, ...]
    active_layers: int
    minimum_count: int
    maximum_count: int
    allocation_entropy_bits: float
    selected_score_fraction: float

    def to_dict(self) -> dict[str, int | float | list[int]]:
        payload = asdict(self)
        payload["layer_counts"] = list(self.layer_counts)
        return payload


def allocate_global_neuron_budget(
    layer_scores: list[torch.Tensor],
    *,
    total_neurons: int,
    minimum_per_layer: int = 0,
) -> AdjointAllocation:
    """Allocate equal-cost exact neurons by global decision-adjoint utility.

    Each score tensor contains one nonnegative utility per MLP neuron. A fixed
    minimum can be reserved per layer. Remaining units are assigned to the
    globally largest neuron scores, which is the exact unit-cost knapsack
    solution for the measured first-order margin objective.
    """

    if not layer_scores:
        raise ValueError("at least one layer score tensor is required")
    if total_neurons <= 0 or minimum_per_layer < 0:
        raise ValueError("total_neurons must be positive and minimum nonnegative")
    sizes = [int(scores.numel()) for scores in layer_scores]
    if any(size <= 0 for size in sizes):
        raise ValueError("every layer must expose at least one neuron")
    reserved = minimum_per_layer * len(layer_scores)
    capacity = sum(sizes)
    if total_neurons < reserved or total_neurons > capacity:
        raise ValueError("total_neurons is outside the feasible allocation range")
    if any(minimum_per_layer > size for size in sizes):
        raise ValueError("minimum_per_layer exceeds a layer's neuron count")

    normalized: list[torch.Tensor] = []
    counts = [minimum_per_layer for _ in layer_scores]
    selected_score = 0.0
    total_score = 0.0
    candidates: list[tuple[float, int]] = []
    layer_offsets: list[int] = []
    offset = 0
    for layer_index, source in enumerate(layer_scores):
        scores = source.detach().to("cpu", torch.float64).reshape(-1)
        if torch.any(scores < 0):
            raise ValueError("neuron utility scores must be nonnegative")
        normalized.append(scores)
        total_score += float(scores.sum().item())
        order = torch.argsort(scores, descending=True)
        if minimum_per_layer:
            selected_score += float(scores[order[:minimum_per_layer]].sum().item())
        for neuron_index in order[minimum_per_layer:].tolist():
            candidates.append((float(scores[neuron_index].item()), layer_index))
        layer_offsets.append(offset)
        offset += scores.numel()

    remaining = total_neurons - reserved
    candidates.sort(key=lambda item: item[0], reverse=True)
    for score, layer_index in candidates[:remaining]:
        counts[layer_index] += 1
        selected_score += score

    count_tensor = torch.tensor(counts, dtype=torch.float64)
    probabilities = count_tensor[count_tensor > 0] / max(float(total_neurons), 1.0)
    entropy = float((-(probabilities * torch.log2(probabilities))).sum().item())
    return AdjointAllocation(
        total_neurons=total_neurons,
        minimum_per_layer=minimum_per_layer,
        layer_counts=tuple(counts),
        active_layers=sum(count > 0 for count in counts),
        minimum_count=min(counts),
        maximum_count=max(counts),
        allocation_entropy_bits=entropy,
        selected_score_fraction=(selected_score / total_score if total_score > 0 else 0.0),
    )


def uniform_neuron_allocation(
    *,
    layers: int,
    intermediate_neurons: int,
    total_neurons: int,
) -> tuple[int, ...]:
    if min(layers, intermediate_neurons, total_neurons) <= 0:
        raise ValueError("allocation dimensions must be positive")
    if total_neurons > layers * intermediate_neurons:
        raise ValueError("total_neurons exceeds model capacity")
    base, remainder = divmod(total_neurons, layers)
    if base > intermediate_neurons or (base == intermediate_neurons and remainder):
        raise ValueError("uniform allocation exceeds layer capacity")
    return tuple(base + int(index < remainder) for index in range(layers))


def replace_llama_mlp_with_count_allocation(
    model: nn.Module,
    *,
    layer_counts: tuple[int, ...] | list[int],
) -> list[OracleHeavyHitterSwiGLU]:
    root = getattr(model, "model", None)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise ValueError("expected a Llama-style model.model.layers stack")
    if len(layer_counts) != len(layers):
        raise ValueError("one neuron count is required per decoder layer")

    replacements: list[OracleHeavyHitterSwiGLU] = []
    for layer, requested_count in zip(layers, layer_counts):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError("decoder layer has no mlp module")
        intermediate = int(mlp.gate_proj.out_features)
        count = int(requested_count)
        if count <= 0 or count > intermediate:
            raise ValueError("each active layer count must be in [1, intermediate]")
        replacement = OracleHeavyHitterSwiGLU(
            gate_proj=mlp.gate_proj,
            up_proj=mlp.up_proj,
            down_proj=mlp.down_proj,
            act_fn=mlp.act_fn,
            selected_fraction=count / intermediate,
        )
        layer.mlp = replacement
        replacements.append(replacement)
    return replacements
