"""E0 source Gates for average-case MatVec error-correction proposals.

``OMEGA-XORLIFT`` asks an average-case finite-field MatVec oracle for noisy
answers and applies an error-correcting reduction.  The reduction is a real
amplifier, but it does not construct the oracle.  This module separates the
paper's average-coordinate premise from a stronger every-row premise, and it
screens a direct exact-row source without pretending that the two coincide.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Iterable, Mapping, Sequence


GIB = 1 << 30
HOT_BITS = 8 * GIB * 8
REGISTERED_BINARY_BITS = 403_747_897_344
REGISTERED_OUTPUT_ROWS = 19_997_952
DECISION = "REJECT_ROW_LOTTERY_AND_UNCHARGED_AMPLIFIER_AS_CORE"


def _decimal_log2(value: Decimal, *, precision: int = 80) -> Decimal:
    if value <= 0:
        raise ValueError("logarithm input must be positive")
    with localcontext() as context:
        context.prec = precision
        return value.ln() / Decimal(2).ln()


def fourier_list_size_upper_bound(advantage: Fraction) -> int:
    """Return ``floor(1 / (4*advantage**2))`` exactly.

    If a Boolean predictor agrees with a parity on at least
    ``1/2 + advantage`` of uniformly random inputs, the corresponding Walsh
    coefficient is at least ``2*advantage``.  Parseval therefore permits at
    most this many such parities for one fixed predictor truth table.
    """

    if not Fraction(0) < advantage <= Fraction(1, 2):
        raise ValueError("advantage must lie in (0, 1/2]")
    numerator = advantage.denominator * advantage.denominator
    denominator = 4 * advantage.numerator * advantage.numerator
    return numerator // denominator


def average_coordinate_accuracy_from_exact_row_fraction(
    exact_row_fraction: Fraction,
    *,
    field_order: int,
) -> Fraction:
    """Accuracy from exact rows and baseline guesses on all other rows.

    This is a counterexample to conflating average-coordinate accuracy with a
    common per-row advantage.  On the uncomputed coordinates the predictor is
    granted the uniform-guess baseline ``1/field_order``.
    """

    if not Fraction(0) <= exact_row_fraction <= Fraction(1):
        raise ValueError("exact_row_fraction must lie in [0, 1]")
    if field_order < 2:
        raise ValueError("field_order must be at least two")
    baseline = Fraction(1, field_order)
    return exact_row_fraction + (1 - exact_row_fraction) * baseline


def minimum_exact_row_fraction_for_average_advantage(
    advantage: Fraction,
    *,
    field_order: int,
) -> Fraction:
    """Return the direct exact-row fraction needed for average advantage."""

    if field_order < 2:
        raise ValueError("field_order must be at least two")
    if not Fraction(0) < advantage <= 1 - Fraction(1, field_order):
        raise ValueError("advantage is outside the attainable range")
    return advantage * field_order / (field_order - 1)


def minimum_rank_covering_row_forms(
    *,
    output_dimension: int,
    input_dimension: int,
) -> tuple[int, int]:
    """Return the direct row-form rank and coefficient-payload floors.

    Recovering every vector in ``F**output_dimension`` from exact linear row
    observations needs at least ``output_dimension`` independent forms.  A
    direct evaluator stores/reads ``input_dimension`` coefficients per form.
    This scoped result does not reject a different compressed nonlinear
    oracle.
    """

    if output_dimension <= 0 or input_dimension <= 0:
        raise ValueError("dimensions must be positive")
    forms = output_dimension
    return forms, forms * input_dimension


def minimum_self_contained_state_bits(
    *,
    binary_coefficient_bits: int,
    output_rows: int,
    advantage: Fraction,
) -> int:
    """Finite state floor for a *common per-row* advantage.

    One state can represent at most ``L**output_rows`` arbitrary row tuples,
    where ``L`` is the per-row parity list bound.  Covering every binary
    checkpoint therefore requires ``2**state_bits * L**output_rows >= 2**D``.
    """

    if binary_coefficient_bits <= 0 or output_rows <= 0:
        raise ValueError("population sizes must be positive")
    list_size = fourier_list_size_upper_bound(advantage)
    if list_size <= 0:
        return binary_coefficient_bits
    with localcontext() as context:
        context.prec = 80
        saved_by_lists = Decimal(output_rows) * _decimal_log2(
            Decimal(list_size)
        )
        lower = Decimal(binary_coefficient_bits) - saved_by_lists
        if lower <= 0:
            return 0
        return int(lower.to_integral_value(rounding="ROUND_CEILING"))


@dataclass(frozen=True)
class AdvantageCeiling:
    missing_bits_per_output_row: Decimal
    log2_advantage_ceiling: Decimal
    log10_advantage_ceiling: Decimal


def self_contained_advantage_ceiling(
    *,
    binary_coefficient_bits: int = REGISTERED_BINARY_BITS,
    output_rows: int = REGISTERED_OUTPUT_ROWS,
    state_bits: int = HOT_BITS,
) -> AdvantageCeiling:
    """Return the optimistic *common per-row* hot-only advantage ceiling."""

    if not 0 <= state_bits < binary_coefficient_bits:
        raise ValueError("state_bits must lie in [0, binary_coefficient_bits)")
    if output_rows <= 0:
        raise ValueError("output_rows must be positive")
    with localcontext() as context:
        context.prec = 80
        missing = (
            Decimal(binary_coefficient_bits - state_bits)
            / Decimal(output_rows)
        )
        log2_ceiling = -(missing + Decimal(2)) / Decimal(2)
        log10_ceiling = log2_ceiling * Decimal(2).log10()
        return AdvantageCeiling(
            missing_bits_per_output_row=+missing,
            log2_advantage_ceiling=+log2_ceiling,
            log10_advantage_ceiling=+log10_ceiling,
        )


def registered_population_from_records(
    records: Iterable[Mapping[str, object]],
    *,
    excluded_tensors: frozenset[str] = frozenset({"embedding"}),
) -> tuple[int, int]:
    """Return coefficient bits and output-row functions from frozen shapes."""

    bits = 0
    rows = 0
    for record in records:
        tensor = str(record["tensor"])
        if tensor in excluded_tensors:
            continue
        count = int(record["count"])
        row_count = int(record["rows"])
        column_count = int(record["columns"])
        if count <= 0 or row_count <= 0 or column_count <= 0:
            raise ValueError("shape records must be positive")
        bits += count * row_count * column_count
        rows += count * row_count
    if bits <= 0 or rows <= 0:
        raise ValueError("no registered rows remain")
    return bits, rows


def parity_agreement_count(
    truth_table: Sequence[int],
    *,
    dimension: int,
    advantage: Fraction,
) -> int:
    """Exhaustively count parities predicted with the requested advantage."""

    if dimension < 0 or len(truth_table) != 1 << dimension:
        raise ValueError("truth table length must equal 2**dimension")
    if any(value not in (0, 1) for value in truth_table):
        raise ValueError("truth table must be Boolean")
    if not Fraction(0) < advantage <= Fraction(1, 2):
        raise ValueError("advantage must lie in (0, 1/2]")
    threshold = Fraction(1, 2) + advantage
    count = 0
    for parity_mask in range(1 << dimension):
        correct = 0
        for query, predicted in enumerate(truth_table):
            expected = (parity_mask & query).bit_count() & 1
            correct += predicted == expected
        if Fraction(correct, 1 << dimension) >= threshold:
            count += 1
    return count


def two_call_self_correction_accuracy(
    truth_table: Sequence[int],
    *,
    parity_mask: int,
    query: int,
) -> Fraction:
    """Exhaustively evaluate ``g(r) xor g(r xor query)`` over uniform ``r``."""

    size = len(truth_table)
    if size == 0 or size & (size - 1):
        raise ValueError("truth table length must be a positive power of two")
    dimension = size.bit_length() - 1
    if not 0 <= parity_mask < size or not 0 <= query < size:
        raise ValueError("mask and query must fit the truth table dimension")
    if any(value not in (0, 1) for value in truth_table):
        raise ValueError("truth table must be Boolean")
    expected = (parity_mask & query).bit_count() & 1
    correct = sum(
        (truth_table[randomizer] ^ truth_table[randomizer ^ query])
        == expected
        for randomizer in range(1 << dimension)
    )
    return Fraction(correct, size)


def derive_audit(
    *,
    binary_coefficient_bits: int = REGISTERED_BINARY_BITS,
    output_rows: int = REGISTERED_OUTPUT_ROWS,
    state_bits: int = HOT_BITS,
) -> dict[str, object]:
    ceiling = self_contained_advantage_ceiling(
        binary_coefficient_bits=binary_coefficient_bits,
        output_rows=output_rows,
        state_bits=state_bits,
    )
    scenarios = {}
    for label, advantage in (
        ("one_percent_advantage", Fraction(1, 100)),
        ("blum_luby_rubinfeld_pair_threshold", Fraction(1, 4)),
    ):
        minimum_bits = minimum_self_contained_state_bits(
            binary_coefficient_bits=binary_coefficient_bits,
            output_rows=output_rows,
            advantage=advantage,
        )
        scenarios[label] = {
            "advantage": str(advantage),
            "accuracy": str(Fraction(1, 2) + advantage),
            "fourier_list_size_upper_bound": fourier_list_size_upper_bound(
                advantage
            ),
            "minimum_state_bits": minimum_bits,
            "minimum_state_gib": minimum_bits / 8 / GIB,
            "multiple_of_8_gib": minimum_bits / state_bits,
        }

    hot_fraction = Fraction(state_bits, binary_coefficient_bits)
    concentrated_accuracy = average_coordinate_accuracy_from_exact_row_fraction(
        hot_fraction,
        field_order=2,
    )
    concentrated_advantage = concentrated_accuracy - Fraction(1, 2)
    return {
        "name": "average_oracle_amplifier_frontier",
        "candidate": "OMEGA-XORLIFT",
        "decision": DECISION,
        "old_2_5_percent_assumption_used": False,
        "registered_population": {
            "binary_coefficient_bits": binary_coefficient_bits,
            "output_row_functions": output_rows,
            "hot_state_bits": state_bits,
            "hot_state_fraction": state_bits / binary_coefficient_bits,
            "average_input_bits_per_row": binary_coefficient_bits / output_rows,
        },
        "published_premise_correction": {
            "metric": "EXPECTED_NORMALIZED_HAMMING_DISTANCE_OVER_OUTPUT_COORDINATES",
            "common_advantage_required_for_every_row": False,
            "hot_fraction_exact_rows_counterexample": str(hot_fraction),
            "gf2_average_coordinate_accuracy": str(concentrated_accuracy),
            "gf2_average_coordinate_advantage": str(concentrated_advantage),
            "counterexample_satisfies_accuracy_part_only": True,
            "counterexample_has_near_linear_query_time": False,
        },
        "uniform_every_row_source_gate": {
            "model": "ONE_GLOBAL_NONLINEAR_HOT_STATE_FREE_QUERY_COMPUTE_NO_COLD_PROBES_COMMON_ROW_ADVANTAGE",
            "missing_bits_per_output_row": str(
                ceiling.missing_bits_per_output_row
            ),
            "common_advantage_ceiling_log2": str(
                ceiling.log2_advantage_ceiling
            ),
            "common_advantage_ceiling_log10": str(
                ceiling.log10_advantage_ceiling
            ),
            "scenarios": scenarios,
            "decision": "REJECT_COMMON_PER_ROW_HOT_ONLY_PARITY_ORACLE",
            "does_not_reject_published_average_distance_premise": True,
        },
        "direct_row_lottery_gate": {
            "model": "EXACT_ENCODED_ROW_FORMS_PLUS_BASELINE_GUESSES",
            "independent_row_forms_needed": output_rows,
            "minimum_direct_coefficient_payload_bits": binary_coefficient_bits,
            "minimum_direct_payload_fraction_of_binary_source": 1.0,
            "hot_exact_state_injection_bits": binary_coefficient_bits,
            "hot_exact_state_gib": binary_coefficient_bits / 8 / GIB,
            "multiple_of_8_gib": binary_coefficient_bits / state_bits,
            "decision": "REJECT_DIRECT_ROW_LOTTERY_AS_A_TRAFFIC_REDUCTION",
        },
        "amplifier_boundary": {
            "published_reduction_supplies_approximate_oracle": False,
            "published_domain": "FINITE_FIELD_RANDOM_MATRIX_AND_VECTOR",
            "native_bf16_fp32_semantics_proved": False,
            "target_derived_causal_predictor_meets_uniform_oracle_premise": False,
            "cold_backed_oracle_covered_by_capacity_gate": False,
            "every_cold_probe_and_repeated_oracle_call_must_be_charged": True,
        },
        "claim_boundary": {
            "general_nonlinear_adaptive_probe_oracle_rejected": False,
            "average_oracle_amplifier_rejected_as_mathematics": False,
            "published_average_distance_oracle_rejected": False,
            "common_per_row_hot_oracle_rejected": True,
            "direct_row_lottery_source_rejected": True,
            "self_contained_exact_8_gib_derived_executor_rejected": True,
            "native_transformer_executor_constructed": False,
            "target_achieved": False,
            "model_or_hardware_execution": False,
        },
    }
