from __future__ import annotations

import random

import pytest

from vortex_runtime.bit_circuit import BinaryLinearTop1Spec
from vortex_runtime.decision_diagram import (
    DecisionDiagramError,
    ReducedDecisionDiagram,
    compile_reduced_decision_diagram,
    natural_variable_order,
    select_smaller_completed_compilation,
    weight_magnitude_variable_order,
)


def compile_complete(
    specification: BinaryLinearTop1Spec,
    order: tuple[int, ...] | None = None,
) -> ReducedDecisionDiagram:
    compilation = compile_reduced_decision_diagram(
        specification,
        variable_order=order or natural_variable_order(specification.input_count),
        compile_state_ceiling=100000,
    )
    assert not compilation.ceiling_hit
    assert compilation.diagram is not None
    return compilation.diagram


def assert_exhaustive(specification: BinaryLinearTop1Spec, diagram: ReducedDecisionDiagram) -> None:
    for value in range(1 << specification.input_count):
        actual, probes = diagram.evaluate_scalar(value)
        assert actual == specification.reference_top1(value)
        assert 0 <= probes <= specification.input_count


def test_late_bit_reduces_to_one_decision_node() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((0,) * 8, (0,) * 7 + (1,)),
        biases=(0, 0),
        accumulator_bits=8,
        family="late_bit",
    )
    diagram = compile_complete(specification)
    assert diagram.node_count == 1
    assert diagram.nodes[0].variable == 7
    assert {diagram.evaluate_scalar(value)[1] for value in range(256)} == {1}
    assert_exhaustive(specification, diagram)


def test_low_equals_high_eliminates_irrelevant_variables() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((2, 0, 0, 0), (0, 0, 0, 0)),
        biases=(-1, 0),
        accumulator_bits=8,
        family="irrelevant",
    )
    diagram = compile_complete(specification)
    assert diagram.node_count == 1
    assert diagram.nodes[0].variable == 0
    assert_exhaustive(specification, diagram)


def test_weight_magnitude_order_is_deterministic() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, 5, 0, -2), (1, -4, 0, 3)),
        biases=(0, 0),
        accumulator_bits=8,
        family="order",
    )
    assert weight_magnitude_variable_order(specification) == (1, 3, 0, 2)


def test_random_dense_diagram_matches_reference() -> None:
    for seed in range(4):
        randomizer = random.Random(seed)
        specification = BinaryLinearTop1Spec(
            weights=tuple(
                tuple(randomizer.choice((-3, -2, -1, 1, 2, 3)) for _ in range(8))
                for _ in range(4)
            ),
            biases=tuple(randomizer.randint(-4, 4) for _ in range(4)),
            accumulator_bits=8,
            family=f"dense_{seed}",
        )
        for order in (
            natural_variable_order(8),
            weight_magnitude_variable_order(specification),
        ):
            diagram = compile_complete(specification, order)
            assert_exhaustive(specification, diagram)


def test_binary_round_trip_preserves_diagram() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, -1, 2, -2), (-2, 2, -1, 1)),
        biases=(0, 1),
        accumulator_bits=8,
        family="round_trip",
    )
    diagram = compile_complete(specification)
    restored = ReducedDecisionDiagram.from_bytes(diagram.to_bytes())
    assert restored == diagram
    assert_exhaustive(specification, restored)


def test_compile_ceiling_fails_closed_without_partial_diagram() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, 1, 1, 1), (-1, -1, -1, -1)),
        biases=(0, 0),
        accumulator_bits=8,
        family="ceiling",
    )
    compilation = compile_reduced_decision_diagram(
        specification,
        variable_order=natural_variable_order(4),
        compile_state_ceiling=2,
    )
    assert compilation.ceiling_hit
    assert compilation.diagram is None
    assert compilation.compile_state_visits == 3


def test_selector_chooses_smaller_completed_diagram() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, 0, 0, 0), (0, 0, 0, 2)),
        biases=(0, 0),
        accumulator_bits=8,
        family="selector",
    )
    natural = compile_reduced_decision_diagram(
        specification,
        variable_order=natural_variable_order(4),
        compile_state_ceiling=1000,
    )
    magnitude = compile_reduced_decision_diagram(
        specification,
        variable_order=weight_magnitude_variable_order(specification),
        compile_state_ceiling=1000,
    )
    selected = select_smaller_completed_compilation((natural, magnitude))
    assert selected is not None
    assert selected.diagram is not None
    assert selected.diagram.serialized_bytes == min(
        natural.diagram.serialized_bytes,
        magnitude.diagram.serialized_bytes,
    )


def test_representation_contains_no_truth_table() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, 2, 3), (-1, -2, -3)),
        biases=(0, 0),
        accumulator_bits=8,
        family="representation",
    )
    diagram = compile_complete(specification)
    assert diagram.contains_truth_table is False
    assert not hasattr(diagram, "truth_table")
    assert diagram.representation_kind == (
        "weight_derived_reduced_ordered_multi_terminal_diagram"
    )


def test_invalid_orders_and_bytes_fail_closed() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((1, 2), (-1, -2)),
        biases=(0, 0),
        accumulator_bits=8,
        family="invalid",
    )
    with pytest.raises(DecisionDiagramError):
        compile_reduced_decision_diagram(
            specification,
            variable_order=(0, 0),
            compile_state_ceiling=100,
        )
    with pytest.raises(DecisionDiagramError):
        compile_reduced_decision_diagram(
            specification,
            variable_order=(0, 1),
            compile_state_ceiling=0,
        )
    with pytest.raises(DecisionDiagramError):
        ReducedDecisionDiagram.from_bytes(b"bad")


def test_path_probe_count_never_exceeds_input_width() -> None:
    specification = BinaryLinearTop1Spec(
        weights=((3, -2, 1, -1, 2), (-1, 3, -2, 2, -3)),
        biases=(1, -1),
        accumulator_bits=8,
        family="probe",
    )
    diagram = compile_complete(specification)
    probes = [diagram.evaluate_scalar(value)[1] for value in range(32)]
    assert max(probes) <= 5
    assert min(probes) >= 0
