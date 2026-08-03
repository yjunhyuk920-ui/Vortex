from __future__ import annotations

import random

import pytest

from vortex_runtime.bit_circuit import (
    AIGBuilder,
    AIGCircuit,
    BinaryLinearTop1Spec,
    BitCircuitError,
    FALSE_LITERAL,
    TRUE_LITERAL,
    compile_binary_linear_top1,
    invert,
)


def exhaustive(specification: BinaryLinearTop1Spec) -> None:
    compiled = compile_binary_linear_top1(specification)
    for value in range(1 << specification.input_count):
        assert compiled.evaluate(value) == specification.reference_top1(value)


def test_structural_hashing_and_boolean_simplification() -> None:
    builder = AIGBuilder(2)
    left, right = builder.input_literals
    first = builder.and_(left, right)
    second = builder.and_(right, left)
    assert first == second
    assert builder.and_(left, TRUE_LITERAL) == left
    assert builder.and_(left, FALSE_LITERAL) == FALSE_LITERAL
    assert builder.and_(left, invert(left)) == FALSE_LITERAL
    circuit = builder.finalize((first,), source_parameter_count=1)
    assert circuit.and_node_count == 1
    assert circuit.requested_and_count == 2
    assert circuit.reachable_and_node_count == 1


def test_aig_binary_round_trip_preserves_evaluation() -> None:
    builder = AIGBuilder(3)
    a, b, c = builder.input_literals
    output = builder.xor(builder.and_(a, b), c)
    circuit = builder.finalize((output,), source_parameter_count=3)
    restored = AIGCircuit.from_bytes(circuit.to_bytes())
    assert restored == circuit
    assert [restored.evaluate_scalar(value) for value in range(8)] == [
        circuit.evaluate_scalar(value) for value in range(8)
    ]


def test_packed_evaluation_matches_scalar() -> None:
    builder = AIGBuilder(3)
    a, b, c = builder.input_literals
    circuit = builder.finalize(
        (builder.xor(a, builder.and_(b, c)),), source_parameter_count=3
    )
    patterns = []
    for bit in range(3):
        pattern = 0
        for value in range(8):
            pattern |= ((value >> bit) & 1) << value
        patterns.append(pattern)
    (packed_output,) = circuit.evaluate_packed(patterns, assignment_count=8)
    scalar_output = sum(
        circuit.evaluate_scalar(value) << value for value in range(8)
    )
    assert packed_output == scalar_output


def test_signed_modular_linear_top1_is_bit_exact() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((7, 7, -8, 3), (-6, 2, 7, -5), (1, 1, 1, 1)),
        biases=(120, -120, 0),
        accumulator_bits=8,
        family="signed_wrap",
    )
    exhaustive(specification)


def test_lower_class_wins_exact_tie() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        biases=(4, 4, 4),
        accumulator_bits=8,
        family="tie",
    )
    compiled = compile_binary_linear_top1(specification)
    assert all(compiled.evaluate(value) == 0 for value in range(8))


def test_late_bit_operator_simplifies_to_input_literal() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((0,) * 7 + (0,), (0,) * 7 + (1,)),
        biases=(0, 0),
        accumulator_bits=8,
        family="late_bit",
    )
    compiled = compile_binary_linear_top1(specification)
    assert compiled.circuit.requested_and_count == 0
    assert compiled.circuit.and_node_count == 0
    assert compiled.circuit.query_node_fraction == 0.0
    for value in range(256):
        assert compiled.evaluate(value) == ((value >> 7) & 1)


def test_compiler_representation_is_not_truth_table() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, -1, 2, -2), (-2, 2, -1, 1)),
        biases=(0, 1),
        accumulator_bits=8,
        family="representation",
    )
    compiled = compile_binary_linear_top1(specification)
    assert compiled.circuit.representation_kind == (
        "weight_derived_structurally_hashed_aig"
    )
    assert compiled.circuit.contains_truth_table is False
    assert not hasattr(compiled.circuit, "truth_table")


def test_random_weight_derived_circuits_match_reference() -> None:
    for seed in range(5):
        randomizer = random.Random(seed)
        specification = BinaryLinearTop1Spec(
            weights=tuple(
                tuple(randomizer.randint(-3, 3) for _ in range(6))
                for _ in range(4)
            ),
            biases=tuple(randomizer.randint(-5, 5) for _ in range(4)),
            accumulator_bits=8,
            family=f"random_{seed}",
        )
        exhaustive(specification)


def test_invalid_specs_and_circuits_fail_closed() -> None:
    with pytest.raises(BitCircuitError):
        BinaryLinearTop1Spec(
            weights=((1, 2), (1,)),
            biases=(0, 0),
            accumulator_bits=8,
            family="bad",
        ).validate()
    with pytest.raises(BitCircuitError):
        BinaryLinearTop1Spec(
            weights=((1,),),
            biases=(0,),
            accumulator_bits=8,
            family="bad",
        ).validate()
    with pytest.raises(BitCircuitError):
        AIGCircuit.from_bytes(b"bad")


def test_query_fraction_is_bounded_and_serialized() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((3, 0, 0, 0), (0, 3, 0, 0), (0, 0, 3, 0)),
        biases=(0, 0, 0),
        accumulator_bits=8,
        family="sparse",
    )
    circuit = compile_binary_linear_top1(specification).circuit
    assert 0.0 <= circuit.query_node_fraction <= 1.0
    assert 0.0 < circuit.query_byte_fraction <= 1.0
    assert len(circuit.to_bytes()) > 0
