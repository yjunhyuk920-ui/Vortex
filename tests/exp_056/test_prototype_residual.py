from __future__ import annotations

import math
import random

import pytest

from vortex_runtime.bit_circuit import BinaryLinearTop1Spec
from vortex_runtime.prototype_residual import (
    PrototypeResidualError,
    PrototypeResidualPlan,
    compile_prototype_residual_plan,
)


def assert_exhaustive(
    specification: BinaryLinearTop1Spec,
    plan: PrototypeResidualPlan,
) -> None:
    for value in range(1 << specification.input_count):
        assert plan.wrapped_scores(value) == specification.wrapped_scores(value)
        assert plan.evaluate_scalar(value) == specification.reference_top1(value)


def test_exact_repeated_columns_need_no_residuals() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((2,) * 8, (-3,) * 8, (1,) * 8),
        biases=(1, -1, 0),
        accumulator_bits=8,
        family="repeated",
    )
    plan = compile_prototype_residual_plan(
        specification,
        requested_prototype_count=1,
        strategy="greedy",
    )
    assert plan.prototype_count == 1
    assert plan.residual_scalar_count == 0
    assert_exhaustive(specification, plan)


def test_sparse_exact_perturbations_are_reconstructed() -> None:
    columns = [(2, -3, 1)] * 8
    columns[2] = (2, -2, 1)
    columns[7] = (4, -3, 1)
    specification = BinaryLinearTop1Spec(
        weights=tuple(
            tuple(column[class_index] for column in columns)
            for class_index in range(3)
        ),
        biases=(0, 1, -1),
        accumulator_bits=8,
        family="prototype_sparse_residual",
    )
    plan = compile_prototype_residual_plan(
        specification,
        requested_prototype_count=1,
        strategy="frequency",
    )
    assert plan.residual_column_count == 2
    assert plan.residual_scalar_count == 2
    assert_exhaustive(specification, plan)


def test_multiple_prototypes_reduce_residuals() -> None:
    columns = [(1, 2)] * 4 + [(-3, 5)] * 4
    specification = BinaryLinearTop1Spec(
        weights=tuple(
            tuple(column[class_index] for column in columns)
            for class_index in range(2)
        ),
        biases=(0, 0),
        accumulator_bits=8,
        family="two_prototypes",
    )
    one = compile_prototype_residual_plan(
        specification,
        requested_prototype_count=1,
        strategy="frequency",
    )
    two = compile_prototype_residual_plan(
        specification,
        requested_prototype_count=2,
        strategy="frequency",
    )
    assert two.residual_scalar_count < one.residual_scalar_count
    assert two.residual_scalar_count == 0
    assert_exhaustive(specification, two)


def test_signed_modular_overflow_matches_reference() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((120, 120, -120, -120), (-125, 125, -125, 125)),
        biases=(120, -120),
        accumulator_bits=8,
        family="overflow",
    )
    for strategy in ("frequency", "greedy"):
        plan = compile_prototype_residual_plan(
            specification,
            requested_prototype_count=2,
            strategy=strategy,
        )
        assert_exhaustive(specification, plan)


def test_random_dense_all_registered_variants_are_exact() -> None:
    for seed in range(4):
        rng = random.Random(seed)
        specification = BinaryLinearTop1Spec(
            weights=tuple(
                tuple(rng.randint(-7, 7) for _ in range(8))
                for _ in range(4)
            ),
            biases=tuple(rng.randint(-5, 5) for _ in range(4)),
            accumulator_bits=8,
            family=f"dense_{seed}",
        )
        for strategy in ("frequency", "greedy"):
            for count in (1, 2, 4, 8):
                assert_exhaustive(
                    specification,
                    compile_prototype_residual_plan(
                        specification,
                        requested_prototype_count=count,
                        strategy=strategy,
                    ),
                )


def test_greedy_is_deterministic() -> None:
    specification = BinaryLinearTop1Spec(
        weights=(
            (1, 1, 2, 2, 3, 3),
            (4, 4, 5, 5, 6, 6),
            (-1, -1, -2, -2, -3, -3),
        ),
        biases=(0, 0, 0),
        accumulator_bits=12,
        family="deterministic",
    )
    left = compile_prototype_residual_plan(
        specification, requested_prototype_count=2, strategy="greedy"
    )
    right = compile_prototype_residual_plan(
        specification, requested_prototype_count=2, strategy="greedy"
    )
    assert left == right
    assert left.to_bytes() == right.to_bytes()


def test_binary_round_trip_preserves_operator() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((3, 3, 4, 3), (-2, -2, -2, 1), (1, 1, 1, -5)),
        biases=(1, -1, 2),
        accumulator_bits=12,
        family="round_trip",
    )
    plan = compile_prototype_residual_plan(
        specification, requested_prototype_count=2, strategy="greedy"
    )
    restored = PrototypeResidualPlan.from_bytes(plan.to_bytes())
    assert restored.groups == plan.groups
    assert restored.residual_columns == plan.residual_columns
    assert restored.strategy == plan.strategy
    assert restored.compile_operation_count == plan.compile_operation_count
    assert restored.to_bytes() == plan.to_bytes()
    assert_exhaustive(specification, restored)


def test_packed_evaluator_matches_scalar() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, 1, 2, 2), (2, 2, -1, -1), (0, 1, 0, -1)),
        biases=(0, 0, 1),
        accumulator_bits=8,
        family="packed",
    )
    plan = compile_prototype_residual_plan(
        specification, requested_prototype_count=2, strategy="greedy"
    )
    assignments = tuple(range(16))
    patterns = tuple(
        sum(((value >> bit) & 1) << assignment for assignment, value in enumerate(assignments))
        for bit in range(specification.input_count)
    )
    packed = plan.evaluate_packed(patterns, assignment_count=len(assignments))
    width = max(1, math.ceil(math.log2(specification.class_count)))
    assert len(packed) == width
    for assignment, value in enumerate(assignments):
        actual = sum(
            ((packed[bit] >> assignment) & 1) << bit
            for bit in range(width)
        )
        assert actual == specification.reference_top1(value)


def test_representation_has_no_runtime_state_table() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, 1, 2, 2), (-1, -1, -2, -2)),
        biases=(0, 0),
        accumulator_bits=8,
        family="representation",
    )
    plan = compile_prototype_residual_plan(
        specification, requested_prototype_count=2, strategy="frequency"
    )
    assert plan.contains_truth_table is False
    assert not hasattr(plan, "truth_table")
    assert not hasattr(plan, "runtime_state_table")
    assert plan.representation_kind == "weight_derived_exact_prototype_sparse_residual"


def test_invalid_inputs_fail_closed() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, 2), (-1, -2)),
        biases=(0, 0),
        accumulator_bits=8,
        family="invalid",
    )
    with pytest.raises(PrototypeResidualError):
        compile_prototype_residual_plan(
            specification, requested_prototype_count=0, strategy="frequency"
        )
    with pytest.raises(PrototypeResidualError):
        compile_prototype_residual_plan(
            specification, requested_prototype_count=1, strategy="unknown"
        )
    plan = compile_prototype_residual_plan(
        specification, requested_prototype_count=1, strategy="frequency"
    )
    with pytest.raises(PrototypeResidualError):
        PrototypeResidualPlan.from_bytes(b"bad")
    with pytest.raises(PrototypeResidualError):
        PrototypeResidualPlan.from_bytes(plan.to_bytes() + b"trailing")
    with pytest.raises(PrototypeResidualError):
        plan.evaluate_packed((1,), assignment_count=1)
