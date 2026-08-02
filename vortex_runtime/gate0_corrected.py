from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

GIB = 1024**3


@dataclass(frozen=True)
class DenseShape:
    parameters: int
    layers: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    kv_heads: int
    vocab_size: int
    context_tokens: int
    weight_bits: int
    kv_bits: int

    @property
    def kv_width(self) -> int:
        return self.hidden_size * self.kv_heads // self.attention_heads


@dataclass(frozen=True)
class CorrectedCandidate:
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


TARGET_405B = DenseShape(
    parameters=405_849_243_648,
    layers=126,
    hidden_size=16_384,
    intermediate_size=53_248,
    attention_heads=128,
    kv_heads=8,
    vocab_size=128_256,
    context_tokens=4096,
    weight_bits=16,
    kv_bits=16,
)

BASELINE_4B = DenseShape(
    parameters=4_000_000_000,
    layers=32,
    hidden_size=3072,
    intermediate_size=8192,
    attention_heads=24,
    kv_heads=8,
    vocab_size=128_256,
    context_tokens=4096,
    weight_bits=4,
    kv_bits=16,
)


def _projection_shapes(shape: DenseShape) -> list[tuple[int, int]]:
    h = shape.hidden_size
    i = shape.intermediate_size
    k = shape.kv_width
    return [
        (h, h),
        (k, h),
        (k, h),
        (h, h),
        (i, h),
        (i, h),
        (h, i),
    ]


def capsule_bytes(shape: DenseShape, candidate: CorrectedCandidate) -> int:
    values = 0
    for output_features, input_features in _projection_shapes(shape):
        values += shape.layers * candidate.linear_rank * (
            input_features + output_features
        )
    values += candidate.linear_rank * (
        shape.hidden_size + shape.vocab_size
    )
    return values * candidate.capsule_bits // 8


def capsule_linear_gflop(shape: DenseShape, candidate: CorrectedCandidate) -> float:
    operations = 0
    for output_features, input_features in _projection_shapes(shape):
        operations += shape.layers * 2 * candidate.linear_rank * (
            input_features + output_features
        )
    operations += 2 * candidate.linear_rank * (
        shape.hidden_size + shape.vocab_size
    )
    return operations / 1e9


def exact_recent_kv_bytes(shape: DenseShape, candidate: CorrectedCandidate) -> int:
    return (
        shape.layers
        * candidate.recent_exact_kv_tokens
        * 2
        * shape.kv_width
        * shape.kv_bits
        // 8
    )


def attention_summary_bytes(shape: DenseShape, candidate: CorrectedCandidate) -> int:
    return (
        shape.layers
        * candidate.attention_rank
        * 2
        * shape.kv_width
        * candidate.attention_summary_bits
        // 8
    )


def baseline_kv_read_bytes(shape: DenseShape) -> int:
    return (
        shape.layers
        * shape.context_tokens
        * 2
        * shape.kv_width
        * shape.kv_bits
        // 8
    )


def exact_projection_gflop(shape: DenseShape) -> float:
    parameters = 0
    for output_features, input_features in _projection_shapes(shape):
        parameters += shape.layers * output_features * input_features
    parameters += shape.vocab_size * shape.hidden_size
    return 2 * parameters / 1e9


def attention_hot_gflop(shape: DenseShape, candidate: CorrectedCandidate) -> float:
    recent = (
        4
        * shape.layers
        * candidate.recent_exact_kv_tokens
        * shape.hidden_size
    )
    summary = (
        4
        * shape.layers
        * candidate.attention_rank
        * shape.hidden_size
    )
    return (recent + summary) / 1e9


def baseline_gflop(shape: DenseShape) -> float:
    weight = 2 * shape.parameters
    attention = 4 * shape.layers * shape.context_tokens * shape.hidden_size
    return (weight + attention) / 1e9


def corrected_gate0_report(
    candidate: CorrectedCandidate = CorrectedCandidate(),
) -> dict[str, object]:
    target = TARGET_405B
    baseline = BASELINE_4B

    linear_capsule = capsule_bytes(target, candidate)
    attention_summary = attention_summary_bytes(target, candidate)
    recent_kv = exact_recent_kv_bytes(target, candidate)
    embedding_cache = (
        candidate.embedding_cache_rows
        * target.hidden_size
        * candidate.capsule_bits
        // 8
    )
    memory_components = {
        "linear_capsules": linear_capsule / GIB,
        "attention_summaries": attention_summary / GIB,
        "recent_exact_kv": recent_kv / GIB,
        "embedding_cache": embedding_cache / GIB,
        "workspace": candidate.workspace_gib,
        "repair_window": candidate.repair_window_gib,
        "allocator_reserve": candidate.allocator_reserve_gib,
        "certificate_state": candidate.certificate_state_gib,
    }
    memory_total = sum(memory_components.values())

    hot_traffic_components = {
        "linear_capsule_read": linear_capsule / GIB,
        "attention_summary_read": attention_summary / GIB,
        "recent_exact_kv_read": recent_kv / GIB,
        "embedding_row_read": (
            target.hidden_size * candidate.capsule_bits // 8
        ) / GIB,
        "misc": candidate.misc_hot_traffic_gib,
    }
    hot_traffic = sum(hot_traffic_components.values())
    baseline_traffic_components = {
        "weights": (
            baseline.parameters * baseline.weight_bits / 8 / GIB
        ),
        "kv": baseline_kv_read_bytes(baseline) / GIB,
    }
    baseline_traffic = sum(baseline_traffic_components.values())
    traffic_limit = 1.2 * baseline_traffic

    cold_weight_gib = target.parameters * target.weight_bits / 8 / GIB
    selected_weight_gib = (
        candidate.selected_repair_fraction * cold_weight_gib
    )
    shared_traffic_per_token = (
        selected_weight_gib
        / candidate.committed_tokens_per_shared_repair
    )
    projected_traffic = hot_traffic + shared_traffic_per_token
    required_efficiency = cold_weight_gib / max(
        1e-12,
        traffic_limit - hot_traffic,
    )
    observed_efficiency = (
        candidate.committed_tokens_per_shared_repair
        / candidate.selected_repair_fraction
    )

    hot_linear_gflop = capsule_linear_gflop(target, candidate)
    hot_attention_gflop = attention_hot_gflop(target, candidate)
    hot_compute = (
        hot_linear_gflop
        + hot_attention_gflop
        + candidate.misc_hot_gflop
    )
    full_exact_gflop = exact_projection_gflop(target)
    exact_selected_gflop = (
        candidate.selected_repair_fraction * full_exact_gflop
    )
    projected_compute = hot_compute + exact_selected_gflop
    baseline_compute = baseline_gflop(baseline)
    compute_limit = 1.2 * baseline_compute
    maximum_compute_fraction = max(
        0.0,
        (compute_limit - hot_compute) / full_exact_gflop,
    )
    maximum_compute_selected_gib = maximum_compute_fraction * cold_weight_gib

    memory_pass = memory_total <= 8.0
    traffic_pass = projected_traffic <= traffic_limit
    compute_pass = projected_compute <= compute_limit
    status = (
        "analytic-envelope-closed"
        if memory_pass and traffic_pass and compute_pass
        else "analytic-envelope-rejected"
    )

    return {
        "evidence_level": "E0 corrected analytic certificate",
        "candidate": "VORTEX-WAVE-1",
        "status": status,
        "target": asdict(target),
        "baseline": asdict(baseline),
        "candidate_parameters": asdict(candidate),
        "memory": {
            "components_gib": memory_components,
            "total_gib": memory_total,
            "limit_gib": 8.0,
            "pass": memory_pass,
        },
        "traffic": {
            "hot_components_gib_per_token": hot_traffic_components,
            "hot_gib_per_token": hot_traffic,
            "baseline_components_gib_per_token": baseline_traffic_components,
            "baseline_gib_per_token": baseline_traffic,
            "limit_gib_per_token": traffic_limit,
            "full_model_weight_gib": cold_weight_gib,
            "selected_shared_weight_gib": selected_weight_gib,
            "shared_repair_gib_per_token": shared_traffic_per_token,
            "projected_gib_per_token": projected_traffic,
            "required_tokens_per_full_repair_equivalent": required_efficiency,
            "candidate_tokens_per_full_repair_equivalent": observed_efficiency,
            "pass": traffic_pass,
        },
        "compute": {
            "hot_linear_gflop_per_token": hot_linear_gflop,
            "hot_attention_gflop_per_token": hot_attention_gflop,
            "hot_total_gflop_per_token": hot_compute,
            "full_exact_projection_gflop_per_token": full_exact_gflop,
            "selected_exact_gflop_per_token": exact_selected_gflop,
            "projected_gflop_per_token": projected_compute,
            "baseline_gflop_per_token": baseline_compute,
            "limit_gflop_per_token": compute_limit,
            "maximum_repair_fraction": maximum_compute_fraction,
            "maximum_selected_weight_gib": maximum_compute_selected_gib,
            "minimum_block_tokens_at_max_compute_fraction": ceil(
                required_efficiency * maximum_compute_fraction
            ),
            "pass": compute_pass,
        },
        "gates": {
            "memory": memory_pass,
            "traffic": traffic_pass,
            "compute": compute_pass,
        },
        "correction": (
            "Shared weight traffic is divided by committed block length. "
            "Selected exact tile arithmetic is charged every token and is "
            "therefore proportional to repair fraction, not 1/E."
        ),
    }
