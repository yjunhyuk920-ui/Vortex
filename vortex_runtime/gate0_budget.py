from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

GIB = 1024**3


def _bytes_for(elements: int, bits: float) -> float:
    if elements < 0:
        raise ValueError("elements must be non-negative")
    if bits <= 0:
        raise ValueError("bits must be positive")
    return elements * bits / 8.0


def _gib(value: float) -> float:
    return value / GIB


@dataclass(frozen=True)
class DenseModelGeometry:
    name: str
    parameter_count: int
    layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        for field_name in (
            "parameter_count",
            "layers",
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("attention heads times head_dim must equal hidden_size")

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim


@dataclass(frozen=True)
class BaselineMeasurement:
    name: str
    traffic_gib_per_token: float
    compute_gflops_per_token: float
    p50_ms_per_token: float | None = None
    p95_ms_per_token: float | None = None
    source: str = "measured"

    def __post_init__(self) -> None:
        if self.traffic_gib_per_token <= 0:
            raise ValueError("traffic_gib_per_token must be positive")
        if self.compute_gflops_per_token <= 0:
            raise ValueError("compute_gflops_per_token must be positive")
        if self.p50_ms_per_token is not None and self.p50_ms_per_token <= 0:
            raise ValueError("p50_ms_per_token must be positive")
        if self.p95_ms_per_token is not None and self.p95_ms_per_token <= 0:
            raise ValueError("p95_ms_per_token must be positive")


@dataclass(frozen=True)
class ProjectedCapsuleCandidate:
    name: str = "vortex-cascade-capsule-v0"
    pre_attention_rank: int = 64
    attention_output_rank: int = 48
    pre_mlp_rank: int = 64
    down_projection_rank: int = 48
    lm_head_rank: int = 64
    basis_bits: float = 8.0
    image_bits: float = 3.0
    capsule_metadata_fraction: float = 0.05
    active_attention_tokens: int = 256
    kv_bits: float = 2.0
    cold_weight_bits: float = 4.0
    embedding_cache_gib: float = 0.125
    runtime_state_gib: float = 0.125
    workspace_gib: float = 1.25
    allocator_reserve_gib: float = 0.50
    repair_tile_gib: float = 0.75
    other_hot_traffic_gib_per_token: float = 0.15
    other_hot_compute_gflops_per_token: float = 0.40
    target_amortized_tokens_per_full_stream: float = 512.0
    memory_limit_gib: float = 8.0
    traffic_ratio_limit: float = 1.2
    compute_ratio_limit: float = 1.2

    def __post_init__(self) -> None:
        rank_fields = (
            "pre_attention_rank",
            "attention_output_rank",
            "pre_mlp_rank",
            "down_projection_rank",
            "lm_head_rank",
            "active_attention_tokens",
        )
        for field_name in rank_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        positive_fields = (
            "basis_bits",
            "image_bits",
            "kv_bits",
            "cold_weight_bits",
            "target_amortized_tokens_per_full_stream",
            "memory_limit_gib",
            "traffic_ratio_limit",
            "compute_ratio_limit",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.capsule_metadata_fraction < 0:
            raise ValueError("capsule_metadata_fraction must be non-negative")


@dataclass(frozen=True)
class Gate0Certificate:
    evidence_level: str
    status: str
    model: dict[str, Any]
    candidate: dict[str, Any]
    baseline: dict[str, Any]
    memory: dict[str, float | bool]
    traffic: dict[str, float | bool]
    compute: dict[str, float | bool]
    decisive_unknowns: list[str]
    falsification_thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_gate0_certificate(
    model: DenseModelGeometry,
    candidate: ProjectedCapsuleCandidate,
    baseline: BaselineMeasurement,
) -> Gate0Certificate:
    h = model.hidden_size
    m = model.intermediate_size
    kv = model.kv_width
    layers = model.layers

    basis_elements_per_layer = (
        h * candidate.pre_attention_rank
        + h * candidate.attention_output_rank
        + h * candidate.pre_mlp_rank
        + m * candidate.down_projection_rank
    )
    image_elements_per_layer = (
        (h + 2 * kv) * candidate.pre_attention_rank
        + h * candidate.attention_output_rank
        + 2 * m * candidate.pre_mlp_rank
        + h * candidate.down_projection_rank
    )
    basis_elements = layers * basis_elements_per_layer
    image_elements = (
        layers * image_elements_per_layer
        + model.vocab_size * candidate.lm_head_rank
    )

    basis_bytes = _bytes_for(basis_elements, candidate.basis_bits)
    image_bytes = _bytes_for(image_elements, candidate.image_bits)
    metadata_bytes = (
        basis_bytes + image_bytes
    ) * candidate.capsule_metadata_fraction
    capsule_bytes = basis_bytes + image_bytes + metadata_bytes

    kv_elements = layers * 2 * kv * candidate.active_attention_tokens
    kv_bytes = _bytes_for(kv_elements, candidate.kv_bits)
    kv_write_bytes = _bytes_for(layers * 2 * kv, candidate.kv_bits)

    m_hot = (
        _gib(capsule_bytes)
        + candidate.embedding_cache_gib
        + candidate.runtime_state_gib
    )
    m_kv = _gib(kv_bytes)
    m_work = candidate.workspace_gib + candidate.allocator_reserve_gib
    m_repair = candidate.repair_tile_gib
    memory_total = m_hot + m_kv + m_work + m_repair
    memory_pass = memory_total <= candidate.memory_limit_gib

    b_hot = (
        _gib(capsule_bytes)
        + _gib(kv_bytes + kv_write_bytes)
        + candidate.other_hot_traffic_gib_per_token
    )
    b_cold = _gib(_bytes_for(model.parameter_count, candidate.cold_weight_bits))
    b_limit = candidate.traffic_ratio_limit * baseline.traffic_gib_per_token
    b_headroom = b_limit - b_hot
    required_a_bandwidth = b_cold / b_headroom if b_headroom > 0 else math.inf
    projected_b_total = (
        b_hot + b_cold / candidate.target_amortized_tokens_per_full_stream
    )
    traffic_pass_at_target = projected_b_total <= b_limit

    projected_linear_gflops = 2.0 * (basis_elements + image_elements) / 1e9
    attention_gflops = (
        4.0 * layers * h * candidate.active_attention_tokens / 1e9
    )
    c_hot = (
        projected_linear_gflops
        + attention_gflops
        + candidate.other_hot_compute_gflops_per_token
    )
    c_repair = 2.0 * model.parameter_count / 1e9
    c_limit = candidate.compute_ratio_limit * baseline.compute_gflops_per_token
    c_headroom = c_limit - c_hot
    required_a_compute = c_repair / c_headroom if c_headroom > 0 else math.inf
    projected_c_total = (
        c_hot + c_repair / candidate.target_amortized_tokens_per_full_stream
    )
    compute_pass_at_target = projected_c_total <= c_limit

    required_a = max(required_a_bandwidth, required_a_compute)
    conditional = (
        memory_pass
        and traffic_pass_at_target
        and compute_pass_at_target
        and math.isfinite(required_a)
    )
    status = "conditional_pass" if conditional else "rejected_by_budget"

    return Gate0Certificate(
        evidence_level="E0-budget/E1-calculator",
        status=status,
        model=asdict(model),
        candidate=asdict(candidate),
        baseline=asdict(baseline),
        memory={
            "M_hot_gib": m_hot,
            "M_kv_gib": m_kv,
            "M_work_gib": m_work,
            "M_repair_gib": m_repair,
            "total_gib": memory_total,
            "limit_gib": candidate.memory_limit_gib,
            "passes": memory_pass,
            "capsule_gib": _gib(capsule_bytes),
            "basis_gib": _gib(basis_bytes),
            "projected_image_gib": _gib(image_bytes),
            "metadata_gib": _gib(metadata_bytes),
        },
        traffic={
            "B_hot_gib_per_token": b_hot,
            "B_cold_gib_per_full_stream": b_cold,
            "B_4B_gib_per_token": baseline.traffic_gib_per_token,
            "limit_gib_per_token": b_limit,
            "required_A_from_bandwidth": required_a_bandwidth,
            "target_A": candidate.target_amortized_tokens_per_full_stream,
            "projected_total_gib_per_token_at_target_A": projected_b_total,
            "passes_at_target_A": traffic_pass_at_target,
        },
        compute={
            "C_hot_gflops_per_token": c_hot,
            "C_repair_gflops_per_full_stream": c_repair,
            "C_4B_gflops_per_token": baseline.compute_gflops_per_token,
            "limit_gflops_per_token": c_limit,
            "projected_linear_gflops_per_token": projected_linear_gflops,
            "compressed_attention_gflops_per_token": attention_gflops,
            "required_A_from_compute": required_a_compute,
            "target_A": candidate.target_amortized_tokens_per_full_stream,
            "projected_total_gflops_per_token_at_target_A": projected_c_total,
            "passes_at_target_A": compute_pass_at_target,
        },
        decisive_unknowns=[
            "weighted cold-repair rate on disjoint real-model decode traces",
            "quality and token agreement at the declared ranks and bit widths",
            "whether active attention can remain bounded at the declared token budget",
            "measured native 4B traffic, compute, and wall-clock on the flagship machine",
            "real CUDA kernel efficiency and overlap of storage, host, and device transfers",
        ],
        falsification_thresholds={
            "minimum_amortized_tokens_per_full_stream": required_a,
            "maximum_full_stream_equivalents_per_token": (
                0.0 if not math.isfinite(required_a) else 1.0 / required_a
            ),
            "maximum_peak_vram_gib": candidate.memory_limit_gib,
            "maximum_traffic_ratio_to_4b": candidate.traffic_ratio_limit,
            "maximum_compute_ratio_to_4b": candidate.compute_ratio_limit,
        },
    )


def llama_31_405b_geometry() -> DenseModelGeometry:
    return DenseModelGeometry(
        name="Llama-3.1-405B-class",
        parameter_count=405_849_243_648,
        layers=126,
        hidden_size=16_384,
        intermediate_size=53_248,
        vocab_size=128_256,
        num_attention_heads=128,
        num_key_value_heads=8,
        head_dim=128,
    )


def conservative_4b_proxy() -> BaselineMeasurement:
    return BaselineMeasurement(
        name="native-4B-Q4 proxy",
        traffic_gib_per_token=2.0,
        compute_gflops_per_token=8.0,
        source="conservative_proxy_not_measured",
    )
