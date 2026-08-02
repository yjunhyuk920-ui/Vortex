from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import torch
from torch import nn

from vortex_runtime.feasibility import GIB, ModelSpec


@dataclass(frozen=True)
class MLPHeavyHitterBudget:
    selected_fraction: float
    selected_neurons_per_layer: int
    source_bits: int
    selected_weight_elements: int
    selected_weight_gib_per_token: float
    selector_metadata_gib: float
    partial_memory_pass: bool
    partial_traffic_pass: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class MLPHeavyHitterStats:
    modules: int
    forward_calls: int
    activation_vectors: int
    intermediate_neurons: int
    selected_neurons_per_vector: int
    selected_fraction: float
    mean_score_coverage: float
    mean_output_relative_l2_error: float
    maximum_output_relative_l2_error: float
    mean_unique_neuron_fraction_per_module: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def mlp_heavy_hitter_budget(
    *,
    target: ModelSpec,
    selected_fraction: float,
    source_bits: int = 16,
    selector_bits_per_neuron: int = 32,
    memory_limit_gib: float = 8.0,
    partial_traffic_limit_gib: float = 1.5,
) -> MLPHeavyHitterBudget:
    """Budget exact selected SwiGLU neuron rows and columns.

    For each selected neuron the runtime needs one exact gate row, one exact up
    row and one exact down column, or ``3 * hidden_size`` source values. This is
    only the MLP partial gate. Attention, embeddings, KV, the selector and proof
    refinement must fit in the remaining envelope.
    """

    if not 0 < selected_fraction <= 1:
        raise ValueError("selected_fraction must be in (0, 1]")
    if min(source_bits, selector_bits_per_neuron) <= 0:
        raise ValueError("precision values must be positive")
    if min(memory_limit_gib, partial_traffic_limit_gib) <= 0:
        raise ValueError("budget limits must be positive")

    selected = max(1, ceil(target.intermediate_size * selected_fraction))
    weight_elements = target.layers * selected * 3 * target.hidden_size
    weight_gib = weight_elements * source_bits / 8 / GIB
    selector_elements = target.layers * target.intermediate_size
    selector_gib = selector_elements * selector_bits_per_neuron / 8 / GIB
    return MLPHeavyHitterBudget(
        selected_fraction=selected_fraction,
        selected_neurons_per_layer=selected,
        source_bits=source_bits,
        selected_weight_elements=weight_elements,
        selected_weight_gib_per_token=weight_gib,
        selector_metadata_gib=selector_gib,
        partial_memory_pass=weight_gib + selector_gib <= memory_limit_gib,
        partial_traffic_pass=weight_gib <= partial_traffic_limit_gib,
    )


class OracleHeavyHitterSwiGLU(nn.Module):
    """Exact-activation oracle that executes only top-contribution down neurons.

    This diagnostic deliberately computes the full gate/up activations to reveal
    the best possible token-dependent neuron subset. It is not yet a fast
    runtime. A promoted architecture must predict the same subset from compact
    metadata before loading exact gate/up/down rows and must certify the omitted
    tail.
    """

    def __init__(
        self,
        *,
        gate_proj: nn.Linear,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
        act_fn: nn.Module | object,
        selected_fraction: float,
    ) -> None:
        super().__init__()
        if gate_proj.out_features != up_proj.out_features:
            raise ValueError("gate and up intermediate dimensions must match")
        if down_proj.in_features != gate_proj.out_features:
            raise ValueError("down input dimension must match gate/up output")
        if down_proj.out_features != gate_proj.in_features:
            raise ValueError("down output dimension must match hidden size")
        if not 0 < selected_fraction <= 1:
            raise ValueError("selected_fraction must be in (0, 1]")

        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn
        self.selected_fraction = float(selected_fraction)
        self.selected_neurons = max(
            1,
            ceil(gate_proj.out_features * selected_fraction),
        )
        column_norms = torch.linalg.vector_norm(
            down_proj.weight.detach().to("cpu", torch.float32),
            dim=0,
        )
        self.register_buffer(
            "down_column_norms",
            column_norms.to(device=down_proj.weight.device),
            persistent=False,
        )
        self.register_buffer(
            "ever_selected",
            torch.zeros(gate_proj.out_features, dtype=torch.bool),
            persistent=False,
        )
        self.forward_calls = 0
        self.activation_vectors = 0
        self.score_coverage_sum = 0.0
        self.output_relative_error_sum = 0.0
        self.output_relative_error_max = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = self.act_fn(gate) * up
        scores = activated.abs() * self.down_column_norms.to(
            device=activated.device,
            dtype=activated.dtype,
        )
        indices = torch.topk(
            scores,
            k=self.selected_neurons,
            dim=-1,
            sorted=False,
        ).indices
        selected_values = torch.gather(activated, dim=-1, index=indices)
        sparse = torch.zeros_like(activated)
        sparse.scatter_(dim=-1, index=indices, src=selected_values)

        sparse_output = self.down_proj(sparse)
        with torch.no_grad():
            exact_output = self.down_proj(activated)
            tail = exact_output - sparse_output
            flat_exact = exact_output.reshape(-1, exact_output.shape[-1])
            flat_tail = tail.reshape(-1, tail.shape[-1])
            relative = torch.linalg.vector_norm(flat_tail, dim=1) / torch.linalg.vector_norm(
                flat_exact,
                dim=1,
            ).clamp_min(1e-12)
            flat_scores = scores.reshape(-1, scores.shape[-1]).to(torch.float32)
            selected_scores = torch.gather(
                flat_scores,
                dim=1,
                index=indices.reshape(-1, indices.shape[-1]),
            )
            coverage = selected_scores.sum(dim=1) / flat_scores.sum(dim=1).clamp_min(
                1e-12
            )
            vectors = int(flat_scores.shape[0])
            self.forward_calls += 1
            self.activation_vectors += vectors
            self.score_coverage_sum += float(coverage.sum().item())
            self.output_relative_error_sum += float(relative.sum().item())
            self.output_relative_error_max = max(
                self.output_relative_error_max,
                float(relative.max().item()),
            )
            unique = torch.unique(indices.detach().to("cpu").reshape(-1))
            self.ever_selected[unique] = True
        return sparse_output

    def reset_statistics(self) -> None:
        self.forward_calls = 0
        self.activation_vectors = 0
        self.score_coverage_sum = 0.0
        self.output_relative_error_sum = 0.0
        self.output_relative_error_max = 0.0
        self.ever_selected.zero_()

    def statistics(self) -> dict[str, int | float]:
        vectors = max(self.activation_vectors, 1)
        return {
            "forward_calls": self.forward_calls,
            "activation_vectors": self.activation_vectors,
            "intermediate_neurons": self.gate_proj.out_features,
            "selected_neurons": self.selected_neurons,
            "selected_fraction": self.selected_neurons / self.gate_proj.out_features,
            "mean_score_coverage": self.score_coverage_sum / vectors,
            "mean_output_relative_l2_error": self.output_relative_error_sum / vectors,
            "maximum_output_relative_l2_error": self.output_relative_error_max,
            "unique_neuron_fraction": float(self.ever_selected.float().mean().item()),
        }


def replace_llama_mlp_with_oracle_heavy_hitters(
    model: nn.Module,
    *,
    selected_fraction: float,
) -> list[OracleHeavyHitterSwiGLU]:
    root = getattr(model, "model", None)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise ValueError("expected a Llama-style model.model.layers stack")
    replacements: list[OracleHeavyHitterSwiGLU] = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError("decoder layer has no mlp module")
        replacement = OracleHeavyHitterSwiGLU(
            gate_proj=mlp.gate_proj,
            up_proj=mlp.up_proj,
            down_proj=mlp.down_proj,
            act_fn=mlp.act_fn,
            selected_fraction=selected_fraction,
        )
        layer.mlp = replacement
        replacements.append(replacement)
    return replacements


def aggregate_heavy_hitter_stats(
    modules: list[OracleHeavyHitterSwiGLU],
) -> MLPHeavyHitterStats:
    if not modules:
        raise ValueError("at least one oracle module is required")
    calls = sum(module.forward_calls for module in modules)
    vectors = sum(module.activation_vectors for module in modules)
    weighted_coverage = sum(module.score_coverage_sum for module in modules)
    weighted_error = sum(module.output_relative_error_sum for module in modules)
    maximum_error = max(module.output_relative_error_max for module in modules)
    unique_fraction = sum(
        float(module.ever_selected.float().mean().item()) for module in modules
    ) / len(modules)
    first = modules[0]
    return MLPHeavyHitterStats(
        modules=len(modules),
        forward_calls=calls,
        activation_vectors=vectors,
        intermediate_neurons=first.gate_proj.out_features,
        selected_neurons_per_vector=first.selected_neurons,
        selected_fraction=first.selected_neurons / first.gate_proj.out_features,
        mean_score_coverage=weighted_coverage / max(vectors, 1),
        mean_output_relative_l2_error=weighted_error / max(vectors, 1),
        maximum_output_relative_l2_error=maximum_error,
        mean_unique_neuron_fraction_per_module=unique_fraction,
    )
