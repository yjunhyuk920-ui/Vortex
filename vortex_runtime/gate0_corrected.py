from __future__ import annotations

from dataclasses import dataclass

from vortex_runtime.feasibility import (
    ModelSpec,
    ObservedMechanism,
    WaveCandidate,
    architecture_gate0_report,
    default_specs,
    linear_capsule_elements,
)

DenseShape = ModelSpec
TARGET_405B, BASELINE_4B = default_specs()


@dataclass(frozen=True)
class CorrectedCandidate:
    """Compatibility configuration for the corrected Gate 0 calculator.

    `selected_repair_fraction` is exact arithmetic performed for every token.
    `committed_tokens_per_shared_repair` amortizes storage traffic only.
    """

    linear_rank: int = 32
    capsule_bits: int = 8
    attention_rank: int = 64
    attention_summary_bits: int = 8
    recent_exact_kv_tokens: int = 128
    embedding_cache_rows: int = 4096
    workspace_gib: float = 1.25
    repair_window_gib: float = 0.5
    allocator_reserve_gib: float = 1.0
    certificate_state_gib: float = 0.25
    misc_hot_traffic_gib: float = 0.05
    misc_hot_gflop: float = 0.5
    selected_repair_fraction: float = 0.25
    committed_tokens_per_shared_repair: int = 160

    def as_wave_candidate(self) -> WaveCandidate:
        return WaveCandidate(
            linear_rank=self.linear_rank,
            capsule_bits=self.capsule_bits,
            attention_rank=self.attention_rank,
            attention_summary_bits=self.attention_summary_bits,
            recent_exact_kv_tokens=self.recent_exact_kv_tokens,
            embedding_cache_rows=self.embedding_cache_rows,
            workspace_gib=self.workspace_gib,
            repair_window_gib=self.repair_window_gib,
            allocator_reserve_gib=self.allocator_reserve_gib,
            certificate_state_gib=self.certificate_state_gib,
            misc_hot_traffic_gib=self.misc_hot_traffic_gib,
            misc_hot_flops=self.misc_hot_gflop * 1e9,
            repair_fraction=self.selected_repair_fraction,
            committed_tokens_per_repair=(
                self.committed_tokens_per_shared_repair
            ),
        )


def capsule_bytes(shape: DenseShape, candidate: CorrectedCandidate) -> int:
    elements = linear_capsule_elements(shape, candidate.linear_rank)
    return int(elements * candidate.capsule_bits / 8)


def capsule_linear_gflop(
    shape: DenseShape,
    candidate: CorrectedCandidate,
) -> float:
    return 2.0 * linear_capsule_elements(
        shape,
        candidate.linear_rank,
    ) / 1e9


def exact_recent_kv_bytes(
    shape: DenseShape,
    candidate: CorrectedCandidate,
) -> int:
    return int(
        shape.layers
        * candidate.recent_exact_kv_tokens
        * 2
        * shape.kv_dim
        * shape.kv_bits
        / 8
    )


def attention_summary_bytes(
    shape: DenseShape,
    candidate: CorrectedCandidate,
) -> int:
    return int(
        shape.layers
        * candidate.attention_rank
        * 2
        * shape.kv_dim
        * candidate.attention_summary_bits
        / 8
    )


def baseline_kv_read_bytes(shape: DenseShape) -> int:
    return int(shape.kv_bytes)


def exact_projection_gflop(shape: DenseShape) -> float:
    return (
        shape.dense_linear_flops_per_token
        + shape.dense_attention_flops_per_token
    ) / 1e9


def attention_hot_gflop(
    shape: DenseShape,
    candidate: CorrectedCandidate,
) -> float:
    return (
        4.0
        * shape.layers
        * shape.hidden_size
        * candidate.attention_rank
        / 1e9
    )


def baseline_gflop(shape: DenseShape) -> float:
    return (
        shape.dense_linear_flops_per_token
        + shape.dense_attention_flops_per_token
    ) / 1e9


def corrected_gate0_report(
    candidate: CorrectedCandidate = CorrectedCandidate(),
) -> dict[str, object]:
    """Delegate to the single active Gate 0 accounting implementation."""

    wave = candidate.as_wave_candidate()
    return architecture_gate0_report(
        target=TARGET_405B,
        baseline=BASELINE_4B,
        candidate=wave,
        observed=ObservedMechanism(
            committed_tokens_per_repair=(
                candidate.committed_tokens_per_shared_repair
            ),
            repair_fraction=candidate.selected_repair_fraction,
            source="gate0_corrected compatibility candidate",
        ),
    )
