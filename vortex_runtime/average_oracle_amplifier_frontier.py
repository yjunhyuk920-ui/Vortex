"""E0 source Gate for average-case MatVec error-correction proposals.

``OMEGA-XORLIFT`` asks an average-case finite-field MatVec oracle for noisy
answers and applies an error-correcting reduction.  The reduction is a real
amplifier, but it does not construct the oracle.  This module gives a finite
capacity screen for the most favorable self-contained oracle: one global hot
state, no cold probes, free query computation, and one predicted bit per row.
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
DECISION = "REJECT_SELF_CONTAINED_AVERAGE_ORACLE_AMPLIFIER_AS_CORE"


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


def minimum_self_contained_state_bits(
    *,
    binary_coefficient_bits: int,
    output_rows: int,
    advantage: Fraction,
) -> int:
    """Finite state floor obtained from the Fourier list bound.

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
    """Return the optimistic common-advantage ceiling for a hot-only oracle."""

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
        "self_contained_source_gate": {
            "model": "ONE_GLOBAL_NONLINEAR_HOT_STATE_FREE_QUERY_COMPUTE_NO_COLD_PROBES",
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
            "decision": "REJECT_HOT_ONLY_AVERAGE_CASE_PARITY_ORACLE",
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
            "self_contained_8_gib_oracle_rejected": True,
            "native_transformer_executor_constructed": False,
            "target_achieved": False,
            "model_or_hardware_execution": False,
        },
    }
