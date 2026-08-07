"""Reference accounting for weight-stationary speculative block execution.

The candidate keeps an unchanged target checkpoint in cold storage, proposes a
causal token block, groups routed MoE tokens by expert, and streams each
required weight page once per verification block.  This module deliberately
models only optimistic logical weight-equivalent traffic.  It does not claim a
physical kernel, proposal accuracy, or target-hardware performance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


class WeightStationaryBudgetError(ValueError):
    """Raised when a block-accounting contract is invalid."""


@dataclass(frozen=True)
class MoeActiveSet:
    """Declared active-set structure for one MoE target."""

    total_parameters: float
    active_parameters: float
    baseline_parameters: float
    hidden_size: int
    expert_intermediate_size: int
    layer_count: int
    expert_count: int
    routed_experts_per_token: int
    shared_experts_per_token: int = 1

    def validate(self) -> None:
        positive = (
            self.total_parameters,
            self.active_parameters,
            self.baseline_parameters,
            self.hidden_size,
            self.expert_intermediate_size,
            self.layer_count,
            self.expert_count,
            self.routed_experts_per_token,
        )
        if any(value <= 0 for value in positive):
            raise WeightStationaryBudgetError("model quantities must be positive")
        if self.active_parameters > self.total_parameters:
            raise WeightStationaryBudgetError("active parameters exceed total parameters")
        if self.routed_experts_per_token > self.expert_count:
            raise WeightStationaryBudgetError("routed experts exceed expert population")
        if self.shared_experts_per_token < 0:
            raise WeightStationaryBudgetError("shared expert count cannot be negative")
        if self.single_token_active_expert_parameters > self.active_parameters:
            raise WeightStationaryBudgetError(
                "derived active expert parameters exceed declared active parameters"
            )

    @property
    def parameters_per_expert_per_layer(self) -> int:
        # Qwen3.5 MoE uses gate/up/down projections for each expert.
        return 3 * self.hidden_size * self.expert_intermediate_size

    @property
    def parameters_per_expert_model_wide(self) -> int:
        return self.parameters_per_expert_per_layer * self.layer_count

    @property
    def single_token_active_expert_parameters(self) -> int:
        return self.parameters_per_expert_model_wide * (
            self.routed_experts_per_token + self.shared_experts_per_token
        )

    @property
    def nonexpert_active_parameters(self) -> float:
        return self.active_parameters - self.single_token_active_expert_parameters


def expected_independent_routed_union(
    *, expert_count: int, routed_per_token: int, block_tokens: int
) -> float:
    """Expected unique routed experts for independent uniform token routes."""
    if expert_count <= 0 or routed_per_token <= 0 or block_tokens <= 0:
        raise WeightStationaryBudgetError("union inputs must be positive")
    if routed_per_token > expert_count:
        raise WeightStationaryBudgetError("routed experts exceed expert population")
    probability_absent_one_token = 1.0 - routed_per_token / expert_count
    return expert_count * (1.0 - probability_absent_one_token**block_tokens)


def maximum_distinct_routed_union(
    *, expert_count: int, routed_per_token: int, block_tokens: int
) -> int:
    if expert_count <= 0 or routed_per_token <= 0 or block_tokens <= 0:
        raise WeightStationaryBudgetError("union inputs must be positive")
    if routed_per_token > expert_count:
        raise WeightStationaryBudgetError("routed experts exceed expert population")
    return min(expert_count, routed_per_token * block_tokens)


def block_target_parameters(
    model: MoeActiveSet,
    *,
    routed_expert_union: float,
) -> float:
    """Logical target parameters read once for one verification block.

    All non-expert active weights and the shared expert set are read once.
    Each distinct routed expert is charged once across the block.
    """
    model.validate()
    if not math.isfinite(routed_expert_union):
        raise WeightStationaryBudgetError("expert union must be finite")
    if not model.routed_experts_per_token <= routed_expert_union <= model.expert_count:
        raise WeightStationaryBudgetError("expert union is outside valid bounds")
    expert_sets = routed_expert_union + model.shared_experts_per_token
    return (
        model.nonexpert_active_parameters
        + model.parameters_per_expert_model_wide * expert_sets
    )


def normalized_weight_traffic(
    model: MoeActiveSet,
    *,
    accepted_tokens: int,
    routed_expert_union: float,
    draft_parameters_per_token: float = 0.0,
) -> float:
    """Return optimistic weight-equivalent traffic / native baseline traffic."""
    model.validate()
    if accepted_tokens <= 0:
        raise WeightStationaryBudgetError("accepted token count must be positive")
    if not math.isfinite(draft_parameters_per_token) or draft_parameters_per_token < 0:
        raise WeightStationaryBudgetError("draft cost must be finite and non-negative")
    target_per_token = block_target_parameters(
        model, routed_expert_union=routed_expert_union
    ) / accepted_tokens
    return (target_per_token + draft_parameters_per_token) / model.baseline_parameters


def minimum_accepted_tokens(
    *,
    block_target_parameter_reads: float,
    baseline_parameters: float,
    allowed_multiplier: float,
    draft_parameters_per_token: float = 0.0,
) -> int:
    """Minimum perfect-acceptance block length under a fixed block read cost."""
    values = (
        block_target_parameter_reads,
        baseline_parameters,
        allowed_multiplier,
        draft_parameters_per_token,
    )
    if any(not math.isfinite(value) for value in values):
        raise WeightStationaryBudgetError("budget values must be finite")
    if block_target_parameter_reads <= 0 or baseline_parameters <= 0:
        raise WeightStationaryBudgetError("parameter counts must be positive")
    if allowed_multiplier <= 0 or draft_parameters_per_token < 0:
        raise WeightStationaryBudgetError("multipliers and draft costs are invalid")
    remaining_per_token = (
        allowed_multiplier * baseline_parameters - draft_parameters_per_token
    )
    if remaining_per_token <= 0:
        raise WeightStationaryBudgetError(
            "draft traffic alone consumes the complete baseline allowance"
        )
    return math.ceil(block_target_parameter_reads / remaining_per_token)


def search_minimum_block(
    model: MoeActiveSet,
    *,
    route_model: str,
    allowed_multiplier: float,
    draft_parameters_per_token: float = 0.0,
    maximum_block_tokens: int = 4096,
) -> int | None:
    """Find the first perfect-acceptance block meeting a route-union model."""
    model.validate()
    if maximum_block_tokens <= 0:
        raise WeightStationaryBudgetError("maximum block size must be positive")
    for block_tokens in range(1, maximum_block_tokens + 1):
        if route_model == "fixed":
            union = float(model.routed_experts_per_token)
        elif route_model == "independent_uniform_expected":
            union = expected_independent_routed_union(
                expert_count=model.expert_count,
                routed_per_token=model.routed_experts_per_token,
                block_tokens=block_tokens,
            )
        elif route_model == "maximally_distinct":
            union = float(
                maximum_distinct_routed_union(
                    expert_count=model.expert_count,
                    routed_per_token=model.routed_experts_per_token,
                    block_tokens=block_tokens,
                )
            )
        else:
            raise WeightStationaryBudgetError(f"unknown route model: {route_model}")
        fraction = normalized_weight_traffic(
            model,
            accepted_tokens=block_tokens,
            routed_expert_union=union,
            draft_parameters_per_token=draft_parameters_per_token,
        )
        if fraction <= allowed_multiplier:
            return block_tokens
    return None


def decimal_gb_to_gib(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise WeightStationaryBudgetError("storage value must be finite and non-negative")
    return value * 1_000_000_000 / (1024**3)
