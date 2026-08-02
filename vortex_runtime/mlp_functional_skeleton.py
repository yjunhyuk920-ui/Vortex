from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from vortex_runtime.mlp_gauge_dictionary import (
    UpDownGaugeStats,
    normalize_swiglu_up_down_gauge,
)
from vortex_runtime.mlp_neuron_dictionary import CompiledSwiGLUDictionary
from vortex_runtime.progressive_precision import symmetric_per_row_fake_quantize


@dataclass(frozen=True)
class FunctionalSkeletonStats:
    neurons: int
    prototypes_requested: int
    prototypes_selected: int
    probe_count: int
    heldout_probe_count: int
    factor_bits: int
    probe_activation_relative_l2_error: float
    heldout_activation_relative_l2_error: float
    probe_output_relative_l2_error: float
    heldout_output_relative_l2_error: float
    coefficient_maximum_absolute_value: float
    coefficient_mean_absolute_value: float
    selected_response_condition_number: float
    factor_elements: int
    factor_bytes: int
    selected_indices: tuple[int, ...]

    @property
    def neuron_collapse_ratio(self) -> float:
        return self.neurons / max(1, self.prototypes_selected)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["selected_indices"] = list(self.selected_indices)
        payload["neuron_collapse_ratio"] = self.neuron_collapse_ratio
        return payload


def deterministic_rademacher_probes(
    *,
    count: int,
    hidden_size: int,
    seed: int,
) -> torch.Tensor:
    """Generate RMSNorm-like deterministic synthetic compile probes.

    Entries are +/-1, so every row has exact root-mean-square one. The probes
    are independent of user prompts, datasets and target labels.
    """

    if count <= 0 or hidden_size <= 0:
        raise ValueError("probe count and hidden size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probes = torch.randint(
        0,
        2,
        (count, hidden_size),
        generator=generator,
        dtype=torch.int8,
    ).to(torch.float32)
    return probes.mul_(2).sub_(1)


def swiglu_neuron_responses(
    probes: torch.Tensor,
    *,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    if probes.ndim != 2 or gate_weight.ndim != 2 or up_weight.ndim != 2:
        raise ValueError("probes and weights must be matrices")
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate and up weights must match")
    if probes.shape[1] != gate_weight.shape[1]:
        raise ValueError("probe hidden dimension mismatch")
    return F.silu(probes @ gate_weight.T) * (probes @ up_weight.T)


def select_response_skeleton(
    responses: torch.Tensor,
    *,
    prototypes: int,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select actual neuron columns by deterministic greedy residual pivoting."""

    if responses.ndim != 2:
        raise ValueError("responses must have shape [probes, neurons]")
    probes, neurons = responses.shape
    if not 0 < prototypes <= neurons:
        raise ValueError("prototypes must be in [1, neurons]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    basis = torch.empty(probes, 0, dtype=responses.dtype)
    residual_energy = responses.square().sum(dim=0)
    selected: list[int] = []
    available = torch.ones(neurons, dtype=torch.bool)

    for _ in range(prototypes):
        masked = torch.where(
            available,
            residual_energy,
            torch.full_like(residual_energy, -1.0),
        )
        index = int(torch.argmax(masked).item())
        if float(masked[index].item()) <= epsilon:
            break
        vector = responses[:, index].clone()
        if basis.shape[1]:
            vector = vector - basis @ (basis.T @ vector)
            # A second pass controls loss of orthogonality in float32.
            vector = vector - basis @ (basis.T @ vector)
        norm = torch.linalg.vector_norm(vector)
        available[index] = False
        if float(norm.item()) <= epsilon:
            residual_energy[index] = -1.0
            continue
        direction = vector / norm
        basis = torch.cat((basis, direction.unsqueeze(1)), dim=1)
        selected.append(index)
        projection = direction @ responses
        residual_energy = torch.clamp(
            residual_energy - projection.square(),
            min=0.0,
        )

    if not selected:
        raise RuntimeError("functional skeleton selected no neurons")
    indices = torch.tensor(selected, dtype=torch.long)
    return indices, basis


def _relative_l2(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(reference - estimate)
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _condition_number(matrix: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(matrix)
    if singular_values.numel() == 0:
        return float("inf")
    smallest = singular_values[-1]
    if float(smallest.item()) <= 1e-12:
        return float("inf")
    return float((singular_values[0] / smallest).item())


def compile_swiglu_functional_skeleton(
    *,
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
    down_proj: nn.Linear,
    prototypes: int,
    probe_count: int = 256,
    heldout_probe_count: int = 128,
    factor_bits: int = 8,
    ridge: float = 1e-5,
    seed: int = 0,
) -> tuple[
    CompiledSwiGLUDictionary,
    FunctionalSkeletonStats,
    UpDownGaugeStats,
]:
    """Compile actual SwiGLU neuron functions by interpolative decomposition.

    The procedure is automatic checkpoint compilation:

    1. apply the exact up/down gauge normalization;
    2. evaluate all neuron functions on deterministic synthetic RMS-one probes;
    3. greedily select actual neuron response columns;
    4. solve a linear interpolation of all responses from selected responses;
    5. absorb interpolation coefficients into the original down matrix.

    Only selected gate/up rows and one aggregated down matrix remain at runtime.
    No user data, prompts, labels or gradient optimization are used.
    """

    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    gate, normalized_up, scaled_down, gauge_stats = (
        normalize_swiglu_up_down_gauge(
            gate_weight=gate_proj.weight,
            up_weight=up_proj.weight,
            down_weight=down_proj.weight,
        )
    )
    neurons, hidden = gate.shape
    if not 0 < prototypes <= neurons:
        raise ValueError("prototypes must be in [1, neurons]")

    probes = deterministic_rademacher_probes(
        count=probe_count,
        hidden_size=hidden,
        seed=seed,
    )
    responses = swiglu_neuron_responses(
        probes,
        gate_weight=gate,
        up_weight=normalized_up,
    )
    selected_indices, _ = select_response_skeleton(
        responses,
        prototypes=prototypes,
    )
    selected = responses[:, selected_indices]

    if ridge > 0:
        gram = selected.T @ selected
        right_hand = selected.T @ responses
        regularized = gram + ridge * torch.eye(
            gram.shape[0],
            dtype=gram.dtype,
        )
        coefficients = torch.linalg.solve(regularized, right_hand)
    else:
        coefficients = torch.linalg.lstsq(selected, responses).solution

    reconstructed_responses = selected @ coefficients
    aggregated_down = scaled_down @ coefficients.T

    heldout = deterministic_rademacher_probes(
        count=heldout_probe_count,
        hidden_size=hidden,
        seed=seed + 1,
    )
    heldout_responses = swiglu_neuron_responses(
        heldout,
        gate_weight=gate,
        up_weight=normalized_up,
    )
    heldout_selected = heldout_responses[:, selected_indices]
    heldout_reconstructed = heldout_selected @ coefficients

    probe_output = responses @ scaled_down.T
    probe_compiled_output = selected @ aggregated_down.T
    heldout_output = heldout_responses @ scaled_down.T
    heldout_compiled_output = heldout_selected @ aggregated_down.T

    selected_gate = gate[selected_indices]
    selected_up = normalized_up[selected_indices]
    quantized_gate, _ = symmetric_per_row_fake_quantize(
        selected_gate,
        bits=factor_bits,
        source_bits=16,
        name="functional_skeleton.gate",
        row_chunk=max(1, selected_gate.shape[0]),
    )
    quantized_up, _ = symmetric_per_row_fake_quantize(
        selected_up,
        bits=factor_bits,
        source_bits=16,
        name="functional_skeleton.up",
        row_chunk=max(1, selected_up.shape[0]),
    )
    quantized_down, _ = symmetric_per_row_fake_quantize(
        aggregated_down,
        bits=factor_bits,
        source_bits=16,
        name="functional_skeleton.down",
        row_chunk=128,
    )
    module = CompiledSwiGLUDictionary(
        gate_weight=quantized_gate,
        up_weight=quantized_up,
        down_weight=quantized_down,
    )
    factor_elements = (
        quantized_gate.numel()
        + quantized_up.numel()
        + quantized_down.numel()
    )
    stats = FunctionalSkeletonStats(
        neurons=neurons,
        prototypes_requested=prototypes,
        prototypes_selected=int(selected_indices.numel()),
        probe_count=probe_count,
        heldout_probe_count=heldout_probe_count,
        factor_bits=factor_bits,
        probe_activation_relative_l2_error=_relative_l2(
            responses,
            reconstructed_responses,
        ),
        heldout_activation_relative_l2_error=_relative_l2(
            heldout_responses,
            heldout_reconstructed,
        ),
        probe_output_relative_l2_error=_relative_l2(
            probe_output,
            probe_compiled_output,
        ),
        heldout_output_relative_l2_error=_relative_l2(
            heldout_output,
            heldout_compiled_output,
        ),
        coefficient_maximum_absolute_value=float(
            coefficients.abs().max().item()
        ),
        coefficient_mean_absolute_value=float(
            coefficients.abs().mean().item()
        ),
        selected_response_condition_number=_condition_number(selected),
        factor_elements=factor_elements,
        factor_bytes=(factor_elements * factor_bits + 7) // 8,
        selected_indices=tuple(int(item) for item in selected_indices.tolist()),
    )
    return module, stats, gauge_stats
