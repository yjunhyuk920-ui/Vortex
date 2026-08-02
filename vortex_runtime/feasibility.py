from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from typing import Any

GIB = 1024**3


@dataclass(frozen=True)
class ModelSpec:
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
    def head_dim(self) -> int:
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        return self.hidden_size // self.attention_heads

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def weight_bytes(self) -> float:
        return self.parameters * self.weight_bits / 8

    @property
    def kv_bytes(self) -> float:
        return (
            self.layers
            * self.context_tokens
            * 2
            * self.kv_dim
            * self.kv_bits
            / 8
        )

    @property
    def dense_linear_flops_per_token(self) -> float:
        return 2.0 * self.parameters

    @property
    def dense_attention_flops_per_token(self) -> float:
        return 4.0 * self.layers * self.hidden_size * self.context_tokens


@dataclass(frozen=True)
class WaveCandidate:
    linear_rank: int = 32
    capsule_bits: int = 8
    attention_rank: int = 64
    attention_summary_bits: int = 8
    recent_exact_kv_tokens: int = 128
    embedding_cache_rows: int = 4096
    workspace_gib: float = 1.25
    repair_window_gib: float = 0.50
    allocator_reserve_gib: float = 1.00
    certificate_state_gib: float = 0.25
    misc_hot_traffic_gib: float = 0.05
    misc_hot_flops: float = 0.50e9
    repair_fraction: float = 0.25
    committed_tokens_per_repair: float = 160.0

    def validate(self) -> None:
        if self.linear_rank <= 0 or self.attention_rank <= 0:
            raise ValueError("ranks must be positive")
        if self.capsule_bits <= 0 or self.attention_summary_bits <= 0:
            raise ValueError("storage bit widths must be positive")
        if not 0 < self.repair_fraction <= 1:
            raise ValueError("repair_fraction must be in (0, 1]")
        if self.committed_tokens_per_repair <= 0:
            raise ValueError("committed_tokens_per_repair must be positive")


@dataclass(frozen=True)
class GateTargets:
    memory_limit_gib: float = 8.0
    traffic_ratio: float = 1.2
    compute_ratio: float = 1.2


@dataclass(frozen=True)
class ObservedMechanism:
    committed_tokens_per_repair: float
    repair_fraction: float
    source: str

    @property
    def repair_efficiency(self) -> float:
        if self.repair_fraction <= 0:
            return inf
        return self.committed_tokens_per_repair / self.repair_fraction


def _linear_shapes(model: ModelSpec) -> tuple[tuple[int, int], ...]:
    kv = model.kv_dim
    h = model.hidden_size
    m = model.intermediate_size
    return (
        (h, h),
        (kv, h),
        (kv, h),
        (h, h),
        (m, h),
        (m, h),
        (h, m),
    )


def linear_capsule_elements(model: ModelSpec, rank: int) -> int:
    per_layer = sum(
        (in_features + out_features) * rank
        for out_features, in_features in _linear_shapes(model)
    )
    lm_head = (model.hidden_size + model.vocab_size) * rank
    return model.layers * per_layer + lm_head


def architecture_gate0_report(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    candidate: WaveCandidate,
    observed: ObservedMechanism,
    targets: GateTargets = GateTargets(),
) -> dict[str, Any]:
    """Build the Gate 0 budget using non-negotiable traffic/compute accounting.

    Cold storage traffic may be shared across a committed token block. Exact
    arithmetic performed with the selected cold weights is still charged for
    every token and therefore scales with repair_fraction, not 1/E.
    """

    candidate.validate()

    capsule_elements = linear_capsule_elements(target, candidate.linear_rank)
    linear_capsule_bytes = capsule_elements * candidate.capsule_bits / 8
    attention_summary_bytes = (
        target.layers
        * 2
        * target.kv_dim
        * candidate.attention_rank
        * candidate.attention_summary_bits
        / 8
    )
    recent_exact_kv_bytes = (
        target.layers
        * candidate.recent_exact_kv_tokens
        * 2
        * target.kv_dim
        * target.kv_bits
        / 8
    )
    embedding_cache_bytes = (
        candidate.embedding_cache_rows
        * target.hidden_size
        * candidate.capsule_bits
        / 8
    )

    memory_components = {
        "linear_capsules": linear_capsule_bytes,
        "attention_summaries": attention_summary_bytes,
        "recent_exact_kv": recent_exact_kv_bytes,
        "embedding_cache": embedding_cache_bytes,
        "workspace": candidate.workspace_gib * GIB,
        "repair_window": candidate.repair_window_gib * GIB,
        "allocator_reserve": candidate.allocator_reserve_gib * GIB,
        "certificate_state": candidate.certificate_state_gib * GIB,
    }
    memory_total = sum(memory_components.values())

    hot_traffic_components = {
        "linear_capsule_read": linear_capsule_bytes,
        "attention_summary_read": attention_summary_bytes,
        "recent_exact_kv_read": recent_exact_kv_bytes,
        "embedding_row_read": target.hidden_size * candidate.capsule_bits / 8,
        "misc": candidate.misc_hot_traffic_gib * GIB,
    }
    hot_bytes_per_token = sum(hot_traffic_components.values())

    baseline_traffic_components = {
        "weights": baseline.weight_bytes,
        "kv": baseline.kv_bytes,
    }
    baseline_bytes_per_token = sum(baseline_traffic_components.values())
    traffic_limit = targets.traffic_ratio * baseline_bytes_per_token
    traffic_headroom = traffic_limit - hot_bytes_per_token

    cold_repair_bytes = target.weight_bytes + target.kv_bytes
    required_efficiency_bytes = (
        cold_repair_bytes / traffic_headroom if traffic_headroom > 0 else inf
    )

    hot_linear_flops = 2.0 * capsule_elements
    hot_attention_flops = (
        4.0 * target.layers * target.hidden_size * candidate.attention_rank
    )
    hot_flops_per_token = (
        hot_linear_flops + hot_attention_flops + candidate.misc_hot_flops
    )

    baseline_flops_per_token = (
        baseline.dense_linear_flops_per_token
        + baseline.dense_attention_flops_per_token
        + candidate.misc_hot_flops
    )
    compute_limit = targets.compute_ratio * baseline_flops_per_token
    compute_headroom = compute_limit - hot_flops_per_token
    cold_repair_flops = (
        target.dense_linear_flops_per_token
        + target.dense_attention_flops_per_token
    )
    maximum_compute_repair_fraction = (
        max(0.0, compute_headroom / cold_repair_flops)
        if cold_repair_flops > 0
        else 0.0
    )

    target_efficiency = (
        candidate.committed_tokens_per_repair / candidate.repair_fraction
    )
    observed_efficiency = observed.repair_efficiency

    projected_bytes_per_token = (
        hot_bytes_per_token
        + cold_repair_bytes
        * candidate.repair_fraction
        / candidate.committed_tokens_per_repair
    )
    projected_flops_per_token = (
        hot_flops_per_token
        + cold_repair_flops * candidate.repair_fraction
    )

    memory_pass = memory_total <= targets.memory_limit_gib * GIB
    analytic_traffic_pass = projected_bytes_per_token <= traffic_limit
    analytic_compute_pass = projected_flops_per_token <= compute_limit
    observed_traffic_pass = observed_efficiency >= required_efficiency_bytes
    observed_compute_pass = (
        observed.repair_fraction <= maximum_compute_repair_fraction
    )
    observed_mechanism_pass = observed_traffic_pass and observed_compute_pass

    if not memory_pass:
        status = "rejected-memory"
    elif not analytic_traffic_pass:
        status = "rejected-analytic-traffic"
    elif not analytic_compute_pass:
        status = "rejected-analytic-compute"
    elif not observed_mechanism_pass:
        status = "blocked-mechanism-unproven"
    else:
        status = "gate0-candidate-ready-for-e2-falsification"

    def gib(value: float) -> float:
        return value / GIB

    return {
        "evidence_level": "E0/E1",
        "candidate": "VORTEX-WAVE-1",
        "status": status,
        "accounting_contract": {
            "traffic": (
                "selected cold bytes are divided by committed block tokens"
            ),
            "compute": (
                "selected exact arithmetic is charged every token and is not "
                "divided by committed block tokens"
            ),
        },
        "target": asdict(target),
        "baseline": asdict(baseline),
        "candidate_parameters": asdict(candidate),
        "targets": asdict(targets),
        "memory": {
            "components_gib": {
                key: gib(value) for key, value in memory_components.items()
            },
            "total_gib": gib(memory_total),
            "limit_gib": targets.memory_limit_gib,
            "pass": memory_pass,
        },
        "traffic": {
            "hot_components_gib_per_token": {
                key: gib(value) for key, value in hot_traffic_components.items()
            },
            "hot_gib_per_token": gib(hot_bytes_per_token),
            "baseline_components_gib_per_token": {
                key: gib(value)
                for key, value in baseline_traffic_components.items()
            },
            "baseline_gib_per_token": gib(baseline_bytes_per_token),
            "limit_gib_per_token": gib(traffic_limit),
            "cold_full_repair_gib": gib(cold_repair_bytes),
            "required_tokens_per_full_repair_equivalent": (
                required_efficiency_bytes
            ),
            "candidate_tokens_per_full_repair_equivalent": target_efficiency,
            "candidate_minimum_committed_tokens": (
                required_efficiency_bytes * candidate.repair_fraction
            ),
            "projected_gib_per_token": gib(projected_bytes_per_token),
            "analytic_pass": analytic_traffic_pass,
        },
        "compute": {
            "hot_linear_gflop_per_token": hot_linear_flops / 1e9,
            "hot_attention_gflop_per_token": hot_attention_flops / 1e9,
            "hot_total_gflop_per_token": hot_flops_per_token / 1e9,
            "baseline_gflop_per_token": baseline_flops_per_token / 1e9,
            "limit_gflop_per_token": compute_limit / 1e9,
            "cold_full_repair_gflop": cold_repair_flops / 1e9,
            "candidate_repair_fraction": candidate.repair_fraction,
            "maximum_repair_fraction": maximum_compute_repair_fraction,
            "maximum_selected_weight_gib": (
                gib(target.weight_bytes * maximum_compute_repair_fraction)
            ),
            "projected_gflop_per_token": projected_flops_per_token / 1e9,
            "analytic_pass": analytic_compute_pass,
        },
        "mechanism": {
            "required_tokens_per_full_repair_equivalent": (
                required_efficiency_bytes
            ),
            "maximum_compute_repair_fraction": (
                maximum_compute_repair_fraction
            ),
            "candidate_target": target_efficiency,
            "candidate_repair_fraction": candidate.repair_fraction,
            "observed": observed_efficiency,
            "observed_repair_fraction": observed.repair_fraction,
            "observed_source": observed.source,
            "traffic_pass": observed_traffic_pass,
            "compute_pass": observed_compute_pass,
            "pass": observed_mechanism_pass,
            "traffic_shortfall_factor": (
                required_efficiency_bytes / observed_efficiency
                if observed_efficiency > 0
                else inf
            ),
            "compute_excess_factor": (
                observed.repair_fraction / maximum_compute_repair_fraction
                if maximum_compute_repair_fraction > 0
                else inf
            ),
        },
        "gates": {
            "memory": memory_pass,
            "analytic_traffic": analytic_traffic_pass,
            "analytic_compute": analytic_compute_pass,
            "observed_repair_traffic": observed_traffic_pass,
            "observed_repair_compute": observed_compute_pass,
            "observed_repair_efficiency": observed_mechanism_pass,
        },
    }


def default_specs() -> tuple[ModelSpec, ModelSpec]:
    target = ModelSpec(
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
    baseline = ModelSpec(
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
    return target, baseline


def default_gate0_report(
    observed_committed_block: float = 1.2751790996462853,
) -> dict[str, Any]:
    target, baseline = default_specs()
    return architecture_gate0_report(
        target=target,
        baseline=baseline,
        candidate=WaveCandidate(),
        observed=ObservedMechanism(
            committed_tokens_per_repair=observed_committed_block,
            repair_fraction=1.0,
            source=(
                "validation_results.json:jacobi.mean_committed_block; "
                "full-repair equivalent"
            ),
        ),
    )
