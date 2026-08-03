from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from statistics import mean, median

import torch

GIB = 1024**3


@dataclass(frozen=True)
class TargetModelShape:
    parameters: int = 405_849_243_648
    hidden_size: int = 16_384
    intermediate_size: int = 53_248
    layers: int = 126


@dataclass(frozen=True)
class SemanticProgramBudget:
    block_size: int
    rank: int
    blocks_per_hidden_vector: int
    coefficient_bits: int
    remainder_bits: int
    scale_bits: int
    basis_bits: int
    coefficient_gib: float
    remainder_gib: float
    scale_gib: float
    basis_gib: float
    active_program_gib: float
    active_program_limit_gib: float
    active_program_pass: bool
    switch_traffic_limit_gib_per_token: float
    minimum_mean_run_length: float

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class RoutingRunStatistics:
    sequences: int
    tokens: int
    program_loads: int
    transitions: int
    transition_switches: int
    load_fraction_per_token: float
    transition_switch_fraction: float
    mean_run_length: float
    median_run_length: float
    p10_run_length: float
    p90_run_length: float
    maximum_run_length: int
    projected_switch_traffic_gib_per_token: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class RatioSummary:
    count: int
    mean: float
    p50: float
    p95: float
    maximum: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def semantic_program_budget(
    *,
    target: TargetModelShape | None = None,
    block_size: int,
    rank: int,
    coefficient_bits: int = 4,
    remainder_bits: int = 8,
    scale_bits: int = 16,
    basis_bits: int = 8,
    active_program_limit_gib: float = 2.0,
    switch_traffic_limit_gib_per_token: float = 0.4,
) -> SemanticProgramBudget:
    """Project one active semantic signed-residual program at 405B scale.

    Three coded matrices are charged for every MLP neuron: gate rows, up rows,
    and down columns. Each matrix row stores signed basis coefficients, one
    remainder norm per hidden block, one coefficient scale, and one remainder
    scale. Gate and up share an activation basis; down uses one dual basis.
    """

    spec = target or TargetModelShape()
    if min(
        block_size,
        rank,
        coefficient_bits,
        remainder_bits,
        scale_bits,
        basis_bits,
    ) <= 0:
        raise ValueError("program dimensions and precisions must be positive")
    if active_program_limit_gib <= 0 or switch_traffic_limit_gib_per_token <= 0:
        raise ValueError("memory and traffic limits must be positive")

    blocks = ceil(spec.hidden_size / block_size)
    coded_rows = spec.layers * spec.intermediate_size * 3
    coefficient_values = coded_rows * blocks * rank
    remainder_values = coded_rows * blocks
    # One signed-coefficient scale and one upward remainder-norm scale per row.
    scale_values = coded_rows * 2
    # Activation and dual bases: block count cancels because blocks tile H.
    basis_values = spec.layers * 2 * spec.hidden_size * rank

    coefficient_gib = coefficient_values * coefficient_bits / 8 / GIB
    remainder_gib = remainder_values * remainder_bits / 8 / GIB
    scale_gib = scale_values * scale_bits / 8 / GIB
    basis_gib = basis_values * basis_bits / 8 / GIB
    active = coefficient_gib + remainder_gib + scale_gib + basis_gib
    return SemanticProgramBudget(
        block_size=block_size,
        rank=rank,
        blocks_per_hidden_vector=blocks,
        coefficient_bits=coefficient_bits,
        remainder_bits=remainder_bits,
        scale_bits=scale_bits,
        basis_bits=basis_bits,
        coefficient_gib=coefficient_gib,
        remainder_gib=remainder_gib,
        scale_gib=scale_gib,
        basis_gib=basis_gib,
        active_program_gib=active,
        active_program_limit_gib=active_program_limit_gib,
        active_program_pass=active <= active_program_limit_gib,
        switch_traffic_limit_gib_per_token=switch_traffic_limit_gib_per_token,
        minimum_mean_run_length=active / switch_traffic_limit_gib_per_token,
    )


def deterministic_projection(
    hidden_size: int,
    signature_size: int,
    *,
    seed: int = 20260803,
) -> torch.Tensor:
    if hidden_size <= 0 or signature_size <= 0 or signature_size > hidden_size:
        raise ValueError("signature size must lie in [1, hidden_size]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(hidden_size, signature_size, generator=generator)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.to(torch.float32).contiguous()


def project_and_normalize_signatures(
    hidden_vectors: list[torch.Tensor],
    projection: torch.Tensor,
) -> torch.Tensor:
    if not hidden_vectors:
        raise ValueError("at least one hidden vector is required")
    if projection.ndim != 2:
        raise ValueError("projection must be a matrix")
    hidden_size = projection.shape[0]
    stacked = torch.stack(
        [vector.detach().to("cpu", torch.float32).reshape(-1) for vector in hidden_vectors],
        dim=0,
    )
    if stacked.shape[1] != hidden_size:
        raise ValueError("hidden vectors do not match projection input size")
    signatures = stacked @ projection
    norms = torch.linalg.vector_norm(signatures, dim=1, keepdim=True)
    return (signatures / torch.clamp(norms, min=1e-12)).contiguous()


def assign_semantic_states(
    signatures: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if signatures.ndim != 2 or centroids.ndim != 2:
        raise ValueError("signatures and centroids must be matrices")
    if signatures.shape[1] != centroids.shape[1]:
        raise ValueError("signature dimensions do not match")
    similarities = signatures @ centroids.T
    best_similarity, labels = torch.max(similarities, dim=1)
    return labels.to(torch.int64), best_similarity.to(torch.float32)


def spherical_state_centroids(
    signatures: torch.Tensor,
    *,
    states: int,
    iterations: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic farthest-point initialization plus spherical Lloyd steps."""

    if signatures.ndim != 2 or signatures.shape[0] == 0:
        raise ValueError("signatures must be a non-empty matrix")
    if states <= 0 or states > signatures.shape[0]:
        raise ValueError("state count must lie in [1, number of signatures]")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    normalized = signatures.to("cpu", torch.float32)
    normalized = normalized / torch.clamp(
        torch.linalg.vector_norm(normalized, dim=1, keepdim=True), min=1e-12
    )
    selected = [0]
    while len(selected) < states:
        selected_tensor = torch.tensor(selected, dtype=torch.long)
        maximum_similarity = (normalized @ normalized[selected_tensor].T).amax(dim=1)
        maximum_similarity[selected_tensor] = 1.0
        next_index = int(torch.argmin(maximum_similarity).item())
        selected.append(next_index)

    centroids = normalized[torch.tensor(selected, dtype=torch.long)].clone()
    for _ in range(iterations):
        labels, _ = assign_semantic_states(normalized, centroids)
        updated: list[torch.Tensor] = []
        for state in range(states):
            members = normalized[labels == state]
            if members.shape[0] == 0:
                nearest, _ = assign_semantic_states(normalized, centroids)
                similarity = (normalized * centroids[nearest]).sum(dim=1)
                candidate = normalized[int(torch.argmin(similarity).item())]
                updated.append(candidate)
                continue
            center = members.mean(dim=0)
            norm = torch.linalg.vector_norm(center)
            if float(norm.item()) <= 1e-12:
                updated.append(members[0])
            else:
                updated.append(center / norm)
        next_centroids = torch.stack(updated, dim=0)
        if torch.allclose(next_centroids, centroids, atol=1e-6, rtol=0.0):
            centroids = next_centroids
            break
        centroids = next_centroids

    labels, _ = assign_semantic_states(normalized, centroids)
    return centroids.contiguous(), labels.contiguous()


def build_block_bases(
    vectors: list[torch.Tensor],
    *,
    block_size: int,
    rank: int,
) -> list[torch.Tensor]:
    if not vectors:
        raise ValueError("at least one vector is required")
    if min(block_size, rank) <= 0:
        raise ValueError("block size and rank must be positive")
    flattened = [vector.detach().to("cpu", torch.float32).reshape(-1) for vector in vectors]
    hidden_size = flattened[0].numel()
    if any(vector.numel() != hidden_size for vector in flattened):
        raise ValueError("all vectors must share one hidden size")

    bases: list[torch.Tensor] = []
    for start in range(0, hidden_size, block_size):
        samples = torch.stack(
            [vector[start : start + block_size] for vector in flattened], dim=0
        )
        usable_rank = min(rank, samples.shape[0], samples.shape[1])
        if usable_rank == 0 or float(torch.linalg.vector_norm(samples).item()) <= 1e-12:
            bases.append(torch.zeros(samples.shape[1], 0, dtype=torch.float32))
            continue
        _, _, vh = torch.linalg.svd(samples, full_matrices=False)
        bases.append(vh[:usable_rank].T.contiguous())
    return bases


def block_perpendicular_ratio(
    vector: torch.Tensor,
    bases: list[torch.Tensor],
    *,
    block_size: int,
) -> float:
    source = vector.detach().to("cpu", torch.float32).reshape(-1)
    expected_blocks = ceil(source.numel() / block_size)
    if len(bases) != expected_blocks:
        raise ValueError("basis partition does not match vector")
    total_energy = float(torch.dot(source, source).item())
    perpendicular_energy = 0.0
    for block_index, start in enumerate(range(0, source.numel(), block_size)):
        block = source[start : start + block_size]
        basis = bases[block_index]
        if basis.shape[0] != block.numel():
            raise ValueError("basis width does not match vector block")
        if basis.shape[1] == 0:
            perpendicular = block
        else:
            coordinates = basis.T @ block
            perpendicular = block - basis @ coordinates
        perpendicular_energy += float(torch.dot(perpendicular, perpendicular).item())
    return (perpendicular_energy / max(total_energy, 1e-24)) ** 0.5


def routing_run_statistics(
    labels_by_sequence: list[list[int]],
    *,
    active_program_gib: float,
) -> RoutingRunStatistics:
    if active_program_gib < 0:
        raise ValueError("program size must be nonnegative")
    if not labels_by_sequence or any(not sequence for sequence in labels_by_sequence):
        raise ValueError("every sequence must contain at least one routed token")

    run_lengths: list[int] = []
    transitions = 0
    transition_switches = 0
    program_loads = 0
    tokens = 0
    for sequence in labels_by_sequence:
        tokens += len(sequence)
        program_loads += 1
        current = sequence[0]
        run = 1
        for label in sequence[1:]:
            transitions += 1
            if label == current:
                run += 1
            else:
                transition_switches += 1
                program_loads += 1
                run_lengths.append(run)
                current = label
                run = 1
        run_lengths.append(run)

    sorted_runs = torch.tensor(sorted(run_lengths), dtype=torch.float64)
    return RoutingRunStatistics(
        sequences=len(labels_by_sequence),
        tokens=tokens,
        program_loads=program_loads,
        transitions=transitions,
        transition_switches=transition_switches,
        load_fraction_per_token=program_loads / tokens,
        transition_switch_fraction=transition_switches / max(transitions, 1),
        mean_run_length=mean(run_lengths),
        median_run_length=median(run_lengths),
        p10_run_length=float(torch.quantile(sorted_runs, 0.10).item()),
        p90_run_length=float(torch.quantile(sorted_runs, 0.90).item()),
        maximum_run_length=max(run_lengths),
        projected_switch_traffic_gib_per_token=(program_loads / tokens)
        * active_program_gib,
    )


def summarize_ratios(values: list[float]) -> RatioSummary:
    if not values:
        raise ValueError("at least one ratio is required")
    tensor = torch.tensor(values, dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("ratios must be finite")
    return RatioSummary(
        count=len(values),
        mean=float(tensor.mean().item()),
        p50=float(torch.quantile(tensor, 0.50).item()),
        p95=float(torch.quantile(tensor, 0.95).item()),
        maximum=float(tensor.max().item()),
    )
