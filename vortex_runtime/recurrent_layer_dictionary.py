from __future__ import annotations

from dataclasses import asdict, dataclass

from vortex_runtime.feasibility import GIB, ModelSpec
from vortex_runtime.substitute_draft_budget import (
    SubstituteDraftBudget,
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
    effective_tops: float
    operations_per_token: float
    compute_seconds_per_token: float
    baseline_seconds_per_token: float
    allowed_seconds_per_token: float
    compute_pass: bool
    memory_pass: bool
    pass_all: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "memory": self.memory.to_dict(),
            "effective_tops": self.effective_tops,
            "operations_per_token": self.operations_per_token,
            "compute_seconds_per_token": self.compute_seconds_per_token,
            "baseline_seconds_per_token": self.baseline_seconds_per_token,
            "allowed_seconds_per_token": self.allowed_seconds_per_token,
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
    effective_tops: float = 160.0,
    baseline_memory_gib_s: float = 300.0,
    baseline_effective_tflops: float = 40.0,
    target_ratio: float = 1.2,
) -> RecurrentDraftBudget:
    """Budget a full-depth draft backed by a small resident layer dictionary.

    Every target layer position is still executed, so arithmetic remains close
    to the original dense model. Only `unique_layers` weight sets plus IO are
    resident, allowing repeated weight reuse to replace host-to-device streams.
    """

    positive = (
        effective_tops,
        baseline_memory_gib_s,
        baseline_effective_tflops,
        target_ratio,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("throughput and target ratio must be positive")

    memory = substitute_draft_budget(
        model=target,
        retained_layers=unique_layers,
        weight_bits=weight_bits,
        tie_word_embeddings=tie_word_embeddings,
        workspace_gib=workspace_gib,
        memory_limit_gib=memory_limit_gib,
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
    compute_pass = compute_seconds <= allowed_seconds
    memory_pass = memory.fits_memory

    return RecurrentDraftBudget(
        memory=memory,
        effective_tops=effective_tops,
        operations_per_token=operations,
        compute_seconds_per_token=compute_seconds,
        baseline_seconds_per_token=baseline_seconds,
        allowed_seconds_per_token=allowed_seconds,
        compute_pass=compute_pass,
        memory_pass=memory_pass,
        pass_all=compute_pass and memory_pass,
    )
