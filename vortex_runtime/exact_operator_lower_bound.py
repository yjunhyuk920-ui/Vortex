from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

GIB = 1024**3


@dataclass(frozen=True)
class ExactOperatorInformationBudget:
    parameter_count: int
    bits_per_parameter: int
    exact_information_bits: int
    exact_information_gib: float
    resident_gib: float
    resident_information_bits: int
    resident_fraction: float
    minimum_external_information_bits: int
    minimum_external_information_gib: float
    dense_compute_gflop: float
    baseline_parameter_count: int
    baseline_dense_compute_gflop: float
    compute_ratio_to_baseline: float
    exact_information_exceeds_resident: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class SkippedCoordinateAdversary:
    rows: int
    columns: int
    skipped_row: int
    skipped_column: int
    competitor_row: int
    inspected_coordinates: int
    total_coordinates: int
    input_vector: list[float]
    baseline_output: list[float]
    alternate_output: list[float]
    baseline_winner: int
    alternate_winner: int
    observations_equal: bool
    outputs_differ: bool
    winner_flips: bool
    changed_coordinates: int
    changed_coordinate_is_uninspected: bool

    @property
    def passes(self) -> bool:
        return bool(
            self.observations_equal
            and self.outputs_differ
            and self.winner_flips
            and self.changed_coordinates == 1
            and self.changed_coordinate_is_uninspected
        )

    def to_dict(self) -> dict[str, int | float | bool | list[float]]:
        payload = asdict(self)
        payload["passes"] = self.passes
        return payload


@dataclass(frozen=True)
class ExhaustiveAdversarySummary:
    rows: int
    columns: int
    total_coordinates: int
    passing_coordinates: int
    coverage: float
    all_observations_equal: bool
    all_outputs_differ: bool
    all_winners_flip: bool

    @property
    def passes(self) -> bool:
        return bool(
            self.passing_coordinates == self.total_coordinates
            and self.coverage == 1.0
            and self.all_observations_equal
            and self.all_outputs_differ
            and self.all_winners_flip
        )

    def to_dict(self) -> dict[str, int | float | bool]:
        payload = asdict(self)
        payload["passes"] = self.passes
        return payload


def exact_operator_information_budget(
    *,
    parameter_count: int,
    bits_per_parameter: int,
    resident_gib: float = 8.0,
    baseline_parameter_count: int = 4_000_000_000,
) -> ExactOperatorInformationBudget:
    """Cardinality and arithmetic lower-bound accounting for a dense operator.

    The information result is for exact operator output or lossless checkpoint
    recovery over every independently selectable b-bit checkpoint. It is not a
    metadata-aware top-1-only bit lower bound.
    """

    if parameter_count <= 0 or bits_per_parameter <= 0:
        raise ValueError("parameter count and precision must be positive")
    if resident_gib < 0 or baseline_parameter_count <= 0:
        raise ValueError("resident memory and baseline size are invalid")

    exact_bits = parameter_count * bits_per_parameter
    exact_gib = exact_bits / 8 / GIB
    resident_bits = int(resident_gib * GIB * 8)
    external_bits = max(0, exact_bits - resident_bits)
    dense_gflop = 2 * parameter_count / 1e9
    baseline_gflop = 2 * baseline_parameter_count / 1e9
    return ExactOperatorInformationBudget(
        parameter_count=parameter_count,
        bits_per_parameter=bits_per_parameter,
        exact_information_bits=exact_bits,
        exact_information_gib=exact_gib,
        resident_gib=resident_gib,
        resident_information_bits=resident_bits,
        resident_fraction=min(1.0, resident_bits / exact_bits),
        minimum_external_information_bits=external_bits,
        minimum_external_information_gib=external_bits / 8 / GIB,
        dense_compute_gflop=dense_gflop,
        baseline_parameter_count=baseline_parameter_count,
        baseline_dense_compute_gflop=baseline_gflop,
        compute_ratio_to_baseline=dense_gflop / baseline_gflop,
        exact_information_exceeds_resident=exact_bits > resident_bits,
    )


def inspected_observation(
    weight: torch.Tensor,
    inspected_mask: torch.Tensor,
) -> torch.Tensor:
    source = weight.detach().to("cpu", torch.float64)
    mask = inspected_mask.detach().to("cpu", torch.bool)
    if source.ndim != 2 or source.shape != mask.shape:
        raise ValueError("weight and inspected mask must be matching matrices")
    return source[mask].contiguous()


def construct_skipped_coordinate_adversary(
    *,
    rows: int,
    columns: int,
    skipped_row: int,
    skipped_column: int,
) -> SkippedCoordinateAdversary:
    """Construct indistinguishable inspected transcripts with different winners.

    Every coordinate except `(skipped_row, skipped_column)` is inspected. The
    two checkpoints differ only at that uninspected coordinate. A one-hot input
    isolates its column, while a fixed competing row makes the baseline and
    alternate top-1 winners unique and different.
    """

    if rows < 2 or columns <= 0:
        raise ValueError("adversary requires at least two rows and one column")
    if not 0 <= skipped_row < rows or not 0 <= skipped_column < columns:
        raise ValueError("skipped coordinate is out of range")

    competitor = (skipped_row + 1) % rows
    baseline = torch.zeros(rows, columns, dtype=torch.float64)
    alternate = baseline.clone()
    baseline[competitor, skipped_column] = 1.0
    alternate[competitor, skipped_column] = 1.0
    baseline[skipped_row, skipped_column] = 0.0
    alternate[skipped_row, skipped_column] = 2.0

    inspected = torch.ones(rows, columns, dtype=torch.bool)
    inspected[skipped_row, skipped_column] = False
    activation = torch.zeros(columns, dtype=torch.float64)
    activation[skipped_column] = 1.0

    observation_baseline = inspected_observation(baseline, inspected)
    observation_alternate = inspected_observation(alternate, inspected)
    output_baseline = baseline @ activation
    output_alternate = alternate @ activation
    baseline_winner = int(torch.argmax(output_baseline).item())
    alternate_winner = int(torch.argmax(output_alternate).item())
    changed = baseline != alternate

    return SkippedCoordinateAdversary(
        rows=rows,
        columns=columns,
        skipped_row=skipped_row,
        skipped_column=skipped_column,
        competitor_row=competitor,
        inspected_coordinates=int(inspected.sum().item()),
        total_coordinates=rows * columns,
        input_vector=activation.tolist(),
        baseline_output=output_baseline.tolist(),
        alternate_output=output_alternate.tolist(),
        baseline_winner=baseline_winner,
        alternate_winner=alternate_winner,
        observations_equal=bool(
            torch.equal(observation_baseline, observation_alternate)
        ),
        outputs_differ=bool(not torch.equal(output_baseline, output_alternate)),
        winner_flips=baseline_winner != alternate_winner,
        changed_coordinates=int(changed.sum().item()),
        changed_coordinate_is_uninspected=bool(
            changed[skipped_row, skipped_column]
            and not inspected[skipped_row, skipped_column]
        ),
    )


def exhaustive_single_skip_adversaries(
    *,
    rows: int,
    columns: int,
) -> tuple[ExhaustiveAdversarySummary, list[SkippedCoordinateAdversary]]:
    cases = [
        construct_skipped_coordinate_adversary(
            rows=rows,
            columns=columns,
            skipped_row=row,
            skipped_column=column,
        )
        for row in range(rows)
        for column in range(columns)
    ]
    passing = sum(case.passes for case in cases)
    total = rows * columns
    summary = ExhaustiveAdversarySummary(
        rows=rows,
        columns=columns,
        total_coordinates=total,
        passing_coordinates=passing,
        coverage=passing / total,
        all_observations_equal=all(case.observations_equal for case in cases),
        all_outputs_differ=all(case.outputs_differ for case in cases),
        all_winners_flip=all(case.winner_flips for case in cases),
    )
    return summary, cases
