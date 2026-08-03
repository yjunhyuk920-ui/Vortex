from __future__ import annotations

import math
import random

import pytest

from vortex_runtime.bit_circuit import BinaryLinearTop1Spec
from vortex_runtime.column_signature import (
    ColumnSignatureError,
    ColumnSignaturePlan,
    compile_column_signature_plan,
)


def assert_exhaustive(
    specification: BinaryLinearTop1Spec,
    plan: ColumnSignaturePlan,
) -> None:
    for value in range(1 << specification.input_count):
        assert plan.wrapped_scores(value) == specification.wrapped_scores(value)
        assert plan.evaluate_scalar(value) == specification.reference_top1(value)


def test_identical_columns_collapse_to_one_group() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((2,) * 8, (-3,) * 8),
        biases=(1, -1),
        accumulator_bits=8,
        family="repeated",
    )
    plan = compile_column_signature_plan(specification, sign_canonical=False)
    assert plan.group_count == 1
    assert plan.groups[0].positive_indices == tuple(range(8))
    assert plan.groups[0].negative_indices == ()
    assert_exhaustive(specification, plan)


def test_exact_negated_columns_share_canonical_group() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, 1, -1, -1), (2, 2, -2, -2)),
        biases=(0, 1),
        accumulator_bits=8,
        family="sign_related",
    )
    exact = compile_column_signature_plan(specification, sign_canonical=False)
    canonical = compile_column_signature_plan(specification, sign_canonical=True)
    assert exact.group_count == 2
    assert canonical.group_count == 1
    assert canonical.groups[0].member_count == 4
    assert canonical.groups[0].positive_indices
    assert canonical.groups[0].negative_indices
    assert_exhaustive(specification, canonical)


def test_signed_modular_overflow_matches_reference() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((120, 120, -120, -120), (-125, 125, -125, 125)),
        biases=(120, -120),
        accumulator_bits=8,
        family="overflow",
    )
    plan = compile_column_signature_plan(specification)
    assert_exhaustive(specification, plan)


def test_random_dense_exactness() -> None:
    for seed in range(6):
        randomizer = random.Random(seed)
        specification = BinaryLinearTop1Spec(
            weights=tuple(
                tuple(randomizer.randint(-7, 7) for _ in range(8))
                for _ in range(4)
            ),
            biases=tuple(randomizer.randint(-8, 8) for _ in range(4)),
            accumulator_bits=8,
            family=f"dense_{seed}",
        )
        for sign_canonical in (False, True):
            assert_exhaustive(
                specification,
                compile_column_signature_plan(
                    specification,
                    sign_canonical=sign_canonical,
                ),
            )


def test_forced_unique_columns_do_not_claim_compression() -> None:
    input_count = 8
    specification = BinaryLinearTop1Spec(
        weights=(
            tuple(range(1, input_count + 1)),
            tuple(100 + value for value in range(1, input_count + 1)),
        ),
        biases=(0, 0),
        accumulator_bits=16,
        family="forced_unique",
    )
    plan = compile_column_signature_plan(specification)
    assert plan.group_count == input_count
    assert plan.operation_fraction > 1.0
    assert_exhaustive(specification, plan)


def test_packed_evaluator_matches_scalar_classes() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, 1, -1, -1), (2, -2, 2, -2), (0, 1, 0, -1)),
        biases=(0, 0, 1),
        accumulator_bits=8,
        family="packed",
    )
    plan = compile_column_signature_plan(specification)
    assignments = tuple(range(16))
    input_patterns = tuple(
        sum(((value >> bit) & 1) << assignment for assignment, value in enumerate(assignments))
        for bit in range(specification.input_count)
    )
    packed = plan.evaluate_packed(input_patterns, assignment_count=len(assignments))
    output_width = max(1, math.ceil(math.log2(specification.class_count)))
    assert len(packed) == output_width
    for assignment, value in enumerate(assignments):
        reconstructed = sum(
            ((packed[bit] >> assignment) & 1) << bit
            for bit in range(output_width)
        )
        assert reconstructed == specification.reference_top1(value)


def test_binary_round_trip_preserves_exact_operator() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((3, 3, -3, 1), (-2, -2, 2, 4), (1, 1, -1, -5)),
        biases=(1, -1, 2),
        accumulator_bits=12,
        family="round_trip",
    )
    plan = compile_column_signature_plan(specification)
    restored = ColumnSignaturePlan.from_bytes(plan.to_bytes())
    assert restored.groups == plan.groups
    assert restored.sign_canonical == plan.sign_canonical
    assert restored.to_bytes() == plan.to_bytes()
    assert_exhaustive(specification, restored)


def test_repeated_structure_has_lower_accounted_cost_than_unique() -> None:
    repeated = BinaryLinearTop1Spec(
        weights=((1,) * 64, (2,) * 64, (-3,) * 64, (4,) * 64),
        biases=(0, 0, 0, 0),
        accumulator_bits=16,
        family="repeated",
    )
    unique = BinaryLinearTop1Spec(
        weights=tuple(
            tuple((class_index + 1) * 1000 + index for index in range(64))
            for class_index in range(4)
        ),
        biases=(0, 0, 0, 0),
        accumulator_bits=16,
        family="unique",
    )
    repeated_plan = compile_column_signature_plan(repeated)
    unique_plan = compile_column_signature_plan(unique)
    assert repeated_plan.operation_fraction < unique_plan.operation_fraction
    assert repeated_plan.query_byte_fraction < unique_plan.query_byte_fraction
    assert repeated_plan.group_count == 1
    assert unique_plan.group_count == 64


def test_representation_contains_no_runtime_state_table() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, 1, 2, 2), (-1, -1, -2, -2)),
        biases=(0, 0),
        accumulator_bits=8,
        family="representation",
    )
    plan = compile_column_signature_plan(specification)
    assert plan.contains_truth_table is False
    assert not hasattr(plan, "truth_table")
    assert not hasattr(plan, "runtime_state_table")
    assert plan.representation_kind == "weight_derived_exact_column_signature_popcount"


def test_invalid_bytes_and_packed_inputs_fail_closed() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, 2), (-1, -2)),
        biases=(0, 0),
        accumulator_bits=8,
        family="invalid",
    )
    plan = compile_column_signature_plan(specification)
    with pytest.raises(ColumnSignatureError):
        ColumnSignaturePlan.from_bytes(b"bad")
    with pytest.raises(ColumnSignatureError):
        ColumnSignaturePlan.from_bytes(plan.to_bytes() + b"trailing")
    with pytest.raises(ColumnSignatureError):
        plan.evaluate_packed((1,), assignment_count=1)
    with pytest.raises(ColumnSignatureError):
        plan.evaluate_packed((2, 0), assignment_count=1)
