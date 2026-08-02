from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class PrecisionStageObservation:
    stage: int
    token: int
    margin: float
    cumulative_layer_fraction: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class PrecisionRouteDecision:
    accepted: bool
    selected_stage: int | None
    selected_token: int | None
    cumulative_layer_fraction: float
    reason: str

    def to_dict(self) -> dict[str, int | float | bool | str | None]:
        return asdict(self)


def select_stable_precision_stage(
    observations: Sequence[PrecisionStageObservation],
    *,
    margin_threshold: float = 0.4,
) -> PrecisionRouteDecision:
    """Select the earliest causally stable residual-precision stage.

    The rule is deliberately fixed and label-free: two adjacent precision
    stages must predict the same token and the later stage must exceed the
    pre-registered top-1 margin. If no stage satisfies the rule, the caller must
    use an exact fallback rather than silently accepting an uncertain token.
    """

    if len(observations) < 2:
        raise ValueError("at least two precision observations are required")
    if margin_threshold < 0:
        raise ValueError("margin_threshold must be non-negative")
    expected_stage = observations[0].stage
    previous_fraction = -1.0
    for observation in observations:
        if observation.stage != expected_stage:
            raise ValueError("precision stages must be consecutive")
        if not 0 <= observation.cumulative_layer_fraction <= 1:
            raise ValueError("cumulative layer fraction must be in [0, 1]")
        if observation.cumulative_layer_fraction < previous_fraction:
            raise ValueError("cumulative layer fractions must be monotonic")
        expected_stage += 1
        previous_fraction = observation.cumulative_layer_fraction

    for previous, current in zip(observations, observations[1:], strict=True):
        if previous.token == current.token and current.margin >= margin_threshold:
            return PrecisionRouteDecision(
                accepted=True,
                selected_stage=current.stage,
                selected_token=current.token,
                cumulative_layer_fraction=current.cumulative_layer_fraction,
                reason="adjacent precision stages agree above fixed margin",
            )
    return PrecisionRouteDecision(
        accepted=False,
        selected_stage=None,
        selected_token=None,
        cumulative_layer_fraction=observations[-1].cumulative_layer_fraction,
        reason="no stable stage; exact fallback required",
    )
