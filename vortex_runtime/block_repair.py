from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil


@dataclass(frozen=True)
class BlockRepairBudget:
    """Traffic accounting for one exact tile set shared across a token block."""

    committed_tokens: int
    selected_weight_bytes: int
    full_model_weight_bytes: int

    def __post_init__(self) -> None:
        if self.committed_tokens < 0:
            raise ValueError("committed_tokens must be non-negative")
        if self.selected_weight_bytes < 0:
            raise ValueError("selected_weight_bytes must be non-negative")
        if self.full_model_weight_bytes <= 0:
            raise ValueError("full_model_weight_bytes must be positive")

    @property
    def repair_fraction(self) -> float:
        return self.selected_weight_bytes / self.full_model_weight_bytes

    @property
    def tokens_per_full_repair_equivalent(self) -> float | None:
        if self.selected_weight_bytes == 0:
            return None
        return self.committed_tokens / self.repair_fraction

    def passes(self, minimum_efficiency: float) -> bool:
        if minimum_efficiency <= 0:
            raise ValueError("minimum_efficiency must be positive")
        if self.selected_weight_bytes == 0:
            return True
        assert self.tokens_per_full_repair_equivalent is not None
        return self.tokens_per_full_repair_equivalent >= minimum_efficiency

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return {
            **asdict(self),
            "repair_fraction": self.repair_fraction,
            "tokens_per_full_repair_equivalent": (
                self.tokens_per_full_repair_equivalent
            ),
        }


def maximum_shared_repair_bytes(
    *,
    committed_tokens: int,
    full_model_weight_bytes: int,
    minimum_efficiency: float,
) -> float:
    if committed_tokens < 0:
        raise ValueError("committed_tokens must be non-negative")
    if full_model_weight_bytes <= 0 or minimum_efficiency <= 0:
        raise ValueError("positive denominator required")
    return (
        committed_tokens
        * full_model_weight_bytes
        / minimum_efficiency
    )


def minimum_committed_tokens(
    *,
    selected_weight_bytes: int,
    full_model_weight_bytes: int,
    minimum_efficiency: float,
) -> int:
    if selected_weight_bytes < 0:
        raise ValueError("selected_weight_bytes must be non-negative")
    if full_model_weight_bytes <= 0 or minimum_efficiency <= 0:
        raise ValueError("positive denominator required")
    return ceil(
        selected_weight_bytes
        * minimum_efficiency
        / full_model_weight_bytes
    )
