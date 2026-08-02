from __future__ import annotations

from dataclasses import asdict, dataclass

from vortex_runtime.substitute_draft_budget import select_layer_indices


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
        min(ordered, key=lambda representative: (abs(position - representative), representative))
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
