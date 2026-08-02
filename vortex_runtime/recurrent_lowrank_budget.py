from __future__ import annotations

from dataclasses import asdict, dataclass

from vortex_runtime.diagonal_transport import (
    DiagonalTransportMetadataBudget,
    diagonal_transport_metadata_budget,
)
from vortex_runtime.feasibility import GIB, ModelSpec
from vortex_runtime.recurrent_layer_dictionary import (
    RecurrentDraftBudget,
    recurrent_draft_budget,
)


@dataclass(frozen=True)
class RecurrentLowRankBudget:
    rank: int
    residual_bits: int
    residual_elements: int
    residual_gib: float
    dictionary: RecurrentDraftBudget
    diagonal_metadata: DiagonalTransportMetadataBudget
    total_gib: float
    memory_limit_gib: float
    extra_residual_flops_per_token: float
    total_compute_seconds_per_token: float
    allowed_seconds_per_token: float
    memory_pass: bool
    compute_pass: bool
    pass_all: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "residual_bits": self.residual_bits,
            "residual_elements": self.residual_elements,
            "residual_gib": self.residual_gib,
            "dictionary": self.dictionary.to_dict(),
            "diagonal_metadata": self.diagonal_metadata.to_dict(),
            "total_gib": self.total_gib,
            "memory_limit_gib": self.memory_limit_gib,
            "extra_residual_flops_per_token": self.extra_residual_flops_per_token,
            "total_compute_seconds_per_token": self.total_compute_seconds_per_token,
            "allowed_seconds_per_token": self.allowed_seconds_per_token,
            "memory_pass": self.memory_pass,
            "compute_pass": self.compute_pass,
            "pass_all": self.pass_all,
        }


def low_rank_residual_elements_per_layer(model: ModelSpec, rank: int) -> int:
    if rank <= 0:
        raise ValueError("rank must be positive")
    hidden = model.hidden_size
    intermediate = model.intermediate_size
    kv = model.kv_dim
    shapes = (
        (hidden, hidden),
        (kv, hidden),
        (kv, hidden),
        (hidden, hidden),
        (intermediate, hidden),
        (intermediate, hidden),
        (hidden, intermediate),
    )
    return rank * sum(rows + cols for rows, cols in shapes)


def recurrent_low_rank_budget(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    unique_layers: int,
    rank: int,
    dictionary_bits: int = 4,
    residual_bits: int = 8,
    metadata_bits: int = 16,
    workspace_gib: float = 1.0,
    memory_limit_gib: float = 8.0,
    effective_tops: float = 160.0,
    baseline_memory_gib_s: float = 300.0,
    baseline_effective_tflops: float = 40.0,
    target_ratio: float = 1.2,
) -> RecurrentLowRankBudget:
    """Budget ``diag(a) D diag(b) + U V.T`` at every target layer position."""

    if rank <= 0 or residual_bits <= 0:
        raise ValueError("rank and residual bits must be positive")
    dictionary = recurrent_draft_budget(
        target=target,
        baseline=baseline,
        unique_layers=unique_layers,
        weight_bits=dictionary_bits,
        tie_word_embeddings=False,
        workspace_gib=workspace_gib,
        memory_limit_gib=memory_limit_gib,
        effective_tops=effective_tops,
        baseline_memory_gib_s=baseline_memory_gib_s,
        baseline_effective_tflops=baseline_effective_tflops,
        target_ratio=target_ratio,
    )
    metadata = diagonal_transport_metadata_budget(
        model=target,
        metadata_bits=metadata_bits,
    )
    residual_elements = target.layers * low_rank_residual_elements_per_layer(
        target,
        rank,
    )
    residual_gib = residual_elements * residual_bits / 8 / GIB
    total_gib = dictionary.memory.total_gib + metadata.metadata_gib + residual_gib

    extra_flops = 2.0 * residual_elements
    total_compute_seconds = (
        dictionary.operations_per_token + extra_flops
    ) / (effective_tops * 1e12)
    memory_pass = total_gib <= memory_limit_gib
    compute_pass = total_compute_seconds <= dictionary.allowed_seconds_per_token

    return RecurrentLowRankBudget(
        rank=rank,
        residual_bits=residual_bits,
        residual_elements=residual_elements,
        residual_gib=residual_gib,
        dictionary=dictionary,
        diagonal_metadata=metadata,
        total_gib=total_gib,
        memory_limit_gib=memory_limit_gib,
        extra_residual_flops_per_token=extra_flops,
        total_compute_seconds_per_token=total_compute_seconds,
        allowed_seconds_per_token=dictionary.allowed_seconds_per_token,
        memory_pass=memory_pass,
        compute_pass=compute_pass,
        pass_all=memory_pass and compute_pass,
    )


def maximum_feasible_residual_rank(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    unique_layers: int = 3,
    dictionary_bits: int = 4,
    residual_bits: int = 8,
    metadata_bits: int = 16,
    workspace_gib: float = 1.0,
    memory_limit_gib: float = 8.0,
    effective_tops: float = 160.0,
    maximum_rank: int = 256,
) -> int:
    feasible = [
        rank
        for rank in range(1, maximum_rank + 1)
        if recurrent_low_rank_budget(
            target=target,
            baseline=baseline,
            unique_layers=unique_layers,
            rank=rank,
            dictionary_bits=dictionary_bits,
            residual_bits=residual_bits,
            metadata_bits=metadata_bits,
            workspace_gib=workspace_gib,
            memory_limit_gib=memory_limit_gib,
            effective_tops=effective_tops,
        ).pass_all
    ]
    return max(feasible, default=0)
