from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

from vortex_runtime.feasibility import GIB, ModelSpec
from vortex_runtime.substitute_draft_budget import (
    SubstituteDraftBudget,
    llama_layer_parameter_count,
    select_layer_indices,
    substitute_draft_budget,
)


@dataclass(frozen=True)
class RecurrentLayerSchedule:
    total_positions: int
    representative_indices: tuple[int, ...]
    assignment: tuple[int, ...]
    strategy: str

    @property
    def unique_layers(self) -> int:
        return len(self.representative_indices)

    def to_dict(self) -> dict[str, int | str | list[int]]:
        payload = asdict(self)
        payload["representative_indices"] = list(self.representative_indices)
        payload["assignment"] = list(self.assignment)
        payload["unique_layers"] = self.unique_layers
        return payload


@dataclass(frozen=True)
class RecurrentDraftBudget:
    memory: SubstituteDraftBudget
    parallel_nodes: int
    resident_hbm_gib_s: float
    recurrent_weight_read_gib_per_decode_step: float
    single_stream_weight_seconds_per_token: float
    throughput_weight_seconds_per_node: float
    minimum_parallel_nodes_for_throughput: int
    effective_tops: float
    operations_per_token: float
    compute_seconds_per_token: float
    single_stream_projected_seconds_per_token: float
    throughput_projected_seconds_per_node: float
    baseline_seconds_per_token: float
    allowed_seconds_per_token: float
    latency_pass: bool
    throughput_pass: bool
    compute_pass: bool
    memory_pass: bool
    pass_all: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "memory": self.memory.to_dict(),
            "parallel_nodes": self.parallel_nodes,
            "resident_hbm_gib_s": self.resident_hbm_gib_s,
            "recurrent_weight_read_gib_per_decode_step": (
                self.recurrent_weight_read_gib_per_decode_step
            ),
            "single_stream_weight_seconds_per_token": (
                self.single_stream_weight_seconds_per_token
            ),
            "throughput_weight_seconds_per_node": (
                self.throughput_weight_seconds_per_node
            ),
            "minimum_parallel_nodes_for_throughput": (
                self.minimum_parallel_nodes_for_throughput
            ),
            "effective_tops": self.effective_tops,
            "operations_per_token": self.operations_per_token,
            "compute_seconds_per_token": self.compute_seconds_per_token,
            "single_stream_projected_seconds_per_token": (
                self.single_stream_projected_seconds_per_token
            ),
            "throughput_projected_seconds_per_node": (
                self.throughput_projected_seconds_per_node
            ),
            "baseline_seconds_per_token": self.baseline_seconds_per_token,
            "allowed_seconds_per_token": self.allowed_seconds_per_token,
            "latency_pass": self.latency_pass,
            "throughput_pass": self.throughput_pass,
            "compute_pass": self.compute_pass,
            "memory_pass": self.memory_pass,
            "pass_all": self.pass_all,
        }


def nearest_representative_assignment(
    *,
    total_layers: int,
    representative_indices: tuple[int, ...],
) -> tuple[int, ...]:
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if not representative_indices:
        raise ValueError("at least one representative is required")
    if any(index < 0 or index >= total_layers for index in representative_indices):
        raise ValueError("representative index out of range")
    if len(set(representative_indices)) != len(representative_indices):
        raise ValueError("representative indices must be unique")

    ordered = tuple(sorted(representative_indices))
    return tuple(
        min(
            ordered,
            key=lambda representative: (
                abs(position - representative),
                representative,
            ),
        )
        for position in range(total_layers)
    )


def cyclic_representative_assignment(
    *,
    total_layers: int,
    representative_indices: tuple[int, ...],
) -> tuple[int, ...]:
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if not representative_indices:
        raise ValueError("at least one representative is required")
    if any(index < 0 or index >= total_layers for index in representative_indices):
        raise ValueError("representative index out of range")
    return tuple(
        representative_indices[position % len(representative_indices)]
        for position in range(total_layers)
    )


def recurrent_layer_schedule(
    *,
    total_layers: int,
    unique_layers: int,
    representative_strategy: str,
    assignment_strategy: str = "nearest",
) -> RecurrentLayerSchedule:
    representatives = select_layer_indices(
        total_layers=total_layers,
        retained_layers=unique_layers,
        strategy=representative_strategy,
    )
    if assignment_strategy == "nearest":
        assignment = nearest_representative_assignment(
            total_layers=total_layers,
            representative_indices=representatives,
        )
    elif assignment_strategy == "cyclic":
        assignment = cyclic_representative_assignment(
            total_layers=total_layers,
            representative_indices=representatives,
        )
    else:
        raise ValueError(f"unsupported assignment strategy: {assignment_strategy}")
    return RecurrentLayerSchedule(
        total_positions=total_layers,
        representative_indices=representatives,
        assignment=assignment,
        strategy=f"{representative_strategy}:{assignment_strategy}",
    )


def recurrent_draft_budget(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    unique_layers: int,
    weight_bits: int = 4,
    tie_word_embeddings: bool = False,
    workspace_gib: float = 1.0,
    memory_limit_gib: float = 8.0,
    parallel_nodes: int = 1,
    resident_hbm_gib_s: float = 300.0,
    effective_tops: float = 160.0,
    baseline_memory_gib_s: float = 300.0,
    baseline_effective_tflops: float = 40.0,
    target_ratio: float = 1.2,
) -> RecurrentDraftBudget:
    """Budget full-depth reuse of a small resident layer dictionary.

    Storage is reduced to `unique_layers`, but one autoregressive depth step
    still reads one complete representative matrix set at every original layer
    position. Parallel branches improve aggregate throughput only; they do not
    reduce latency of the single committed path because the next depth cannot be
    constructed until the current depth finishes.
    """

    if parallel_nodes <= 0:
        raise ValueError("parallel_nodes must be positive")
    positive = (
        resident_hbm_gib_s,
        effective_tops,
        baseline_memory_gib_s,
        baseline_effective_tflops,
        target_ratio,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("bandwidth, throughput and target ratio must be positive")

    memory = substitute_draft_budget(
        model=target,
        retained_layers=unique_layers,
        weight_bits=weight_bits,
        tie_word_embeddings=tie_word_embeddings,
        workspace_gib=workspace_gib,
        memory_limit_gib=memory_limit_gib,
    )

    per_layer_parameters = llama_layer_parameter_count(target)
    repeated_layer_bytes = (
        target.layers * per_layer_parameters * weight_bits / 8
    )
    lm_head_bytes = target.vocab_size * target.hidden_size * weight_bits / 8
    recurrent_weight_read_gib = (repeated_layer_bytes + lm_head_bytes) / GIB
    single_stream_weight_seconds = (
        recurrent_weight_read_gib / resident_hbm_gib_s
    )
    throughput_weight_seconds = (
        single_stream_weight_seconds / parallel_nodes
    )

    operations = (
        target.dense_linear_flops_per_token
        + target.dense_attention_flops_per_token
    )
    compute_seconds = operations / (effective_tops * 1e12)

    baseline_weight_seconds = baseline.weight_bytes / GIB / baseline_memory_gib_s
    baseline_operations = (
        baseline.dense_linear_flops_per_token
        + baseline.dense_attention_flops_per_token
    )
    baseline_compute_seconds = baseline_operations / (
        baseline_effective_tflops * 1e12
    )
    baseline_seconds = max(baseline_weight_seconds, baseline_compute_seconds)
    allowed_seconds = target_ratio * baseline_seconds
    minimum_parallel_nodes = ceil(single_stream_weight_seconds / allowed_seconds)

    single_stream_projected = max(single_stream_weight_seconds, compute_seconds)
    throughput_projected = max(throughput_weight_seconds, compute_seconds)
    latency_pass = single_stream_projected <= allowed_seconds
    throughput_pass = throughput_projected <= allowed_seconds
    compute_pass = compute_seconds <= allowed_seconds
    memory_pass = memory.fits_memory

    return RecurrentDraftBudget(
        memory=memory,
        parallel_nodes=parallel_nodes,
        resident_hbm_gib_s=resident_hbm_gib_s,
        recurrent_weight_read_gib_per_decode_step=recurrent_weight_read_gib,
        single_stream_weight_seconds_per_token=single_stream_weight_seconds,
        throughput_weight_seconds_per_node=throughput_weight_seconds,
        minimum_parallel_nodes_for_throughput=minimum_parallel_nodes,
        effective_tops=effective_tops,
        operations_per_token=operations,
        compute_seconds_per_token=compute_seconds,
        single_stream_projected_seconds_per_token=single_stream_projected,
        throughput_projected_seconds_per_node=throughput_projected,
        baseline_seconds_per_token=baseline_seconds,
        allowed_seconds_per_token=allowed_seconds,
        latency_pass=latency_pass,
        throughput_pass=throughput_pass,
        compute_pass=compute_pass,
        memory_pass=memory_pass,
        pass_all=latency_pass and compute_pass and memory_pass,
    )
