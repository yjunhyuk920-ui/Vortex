"""E0 audit helpers for exact-field nonlinear bilinear query programs.

The scientific result is a containment statement, not a universal lower
bound.  A bounded algebraic query program that returns ``r.T @ W @ u`` exactly
on an open field domain has a fixed branch path that computes the same
rational function.  Baur--Strassen reverse differentiation then turns that
scalar path into a static arithmetic circuit for ``W @ u`` with constant
factor arithmetic overhead.

The exact tape below is an independent finite reference for the derivative
identity, including deliberately nonlinear terms that cancel.  It is not a
proof of the path/identity theorem and it does not model floating-point
rounding, bit operations, or data-dependent word addresses.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import random
from typing import Iterable, Sequence

REJECT_DECISION = (
    "REJECT_EXACT_FIELD_NONLINEAR_BILINEAR_ARITHMETIC_AS_DISTINCT_CORE_CLASS"
)
BAUR_STRASSEN_UNIT_COST_FACTOR = 4
REFERENCE_SEED = 0xB11E_A2
P50_TARGET_FRACTION = Fraction(8, 675)


@dataclass(frozen=True)
class TapeNode:
    """One exact scalar node in a division-free arithmetic tape."""

    operation: str
    value: Fraction
    left: int | None = None
    right: int | None = None


class ExactArithmeticTape:
    """Tiny exact tape with reverse-mode derivatives over ``Fraction``."""

    def __init__(self) -> None:
        self.nodes: list[TapeNode] = []

    def input(self, value: Fraction | int) -> int:
        self.nodes.append(TapeNode("input", Fraction(value)))
        return len(self.nodes) - 1

    def constant(self, value: Fraction | int) -> int:
        self.nodes.append(TapeNode("constant", Fraction(value)))
        return len(self.nodes) - 1

    def _binary(self, operation: str, left: int, right: int) -> int:
        lhs = self.nodes[left].value
        rhs = self.nodes[right].value
        if operation == "add":
            value = lhs + rhs
        elif operation == "sub":
            value = lhs - rhs
        elif operation == "mul":
            value = lhs * rhs
        else:  # pragma: no cover - private caller guard
            raise ValueError(f"unsupported operation: {operation}")
        self.nodes.append(TapeNode(operation, value, left, right))
        return len(self.nodes) - 1

    def add(self, left: int, right: int) -> int:
        return self._binary("add", left, right)

    def sub(self, left: int, right: int) -> int:
        return self._binary("sub", left, right)

    def mul(self, left: int, right: int) -> int:
        return self._binary("mul", left, right)

    def gradient(self, output: int, inputs: Iterable[int]) -> tuple[Fraction, ...]:
        """Return exact reverse derivatives for the requested input nodes."""

        adjoints = [Fraction(0) for _ in self.nodes]
        adjoints[output] = Fraction(1)
        for index in range(output, -1, -1):
            node = self.nodes[index]
            adjoint = adjoints[index]
            if not adjoint or node.operation in {"input", "constant"}:
                continue
            if node.left is None or node.right is None:
                raise AssertionError("binary tape node is missing an operand")
            if node.operation == "add":
                adjoints[node.left] += adjoint
                adjoints[node.right] += adjoint
            elif node.operation == "sub":
                adjoints[node.left] += adjoint
                adjoints[node.right] -= adjoint
            elif node.operation == "mul":
                adjoints[node.left] += adjoint * self.nodes[node.right].value
                adjoints[node.right] += adjoint * self.nodes[node.left].value
            else:  # pragma: no cover - node construction prevents this
                raise AssertionError(f"unknown tape operation: {node.operation}")
        return tuple(adjoints[index] for index in inputs)

    @property
    def arithmetic_operations(self) -> int:
        return sum(
            node.operation in {"add", "sub", "mul"} for node in self.nodes
        )


def exact_matvec(
    matrix: Sequence[Sequence[Fraction | int]],
    vector: Sequence[Fraction | int],
) -> tuple[Fraction, ...]:
    if not matrix:
        raise ValueError("matrix must not be empty")
    width = len(matrix[0])
    if width == 0 or len(vector) != width:
        raise ValueError("matrix/vector shape mismatch")
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have equal width")
    return tuple(
        sum(
            (Fraction(coefficient) * Fraction(value) for coefficient, value in zip(row, vector, strict=True)),
            Fraction(0),
        )
        for row in matrix
    )


def exact_bilinear(
    left: Sequence[Fraction | int],
    matrix: Sequence[Sequence[Fraction | int]],
    right: Sequence[Fraction | int],
) -> Fraction:
    product = exact_matvec(matrix, right)
    if len(left) != len(product):
        raise ValueError("left/matrix shape mismatch")
    return sum(
        (Fraction(value) * output for value, output in zip(left, product, strict=True)),
        Fraction(0),
    )


def build_obfuscated_bilinear_tape(
    left: Sequence[Fraction | int],
    matrix: Sequence[Sequence[Fraction | int]],
    right: Sequence[Fraction | int],
    *,
    branch: int,
) -> tuple[ExactArithmeticTape, int, tuple[int, ...]]:
    """Build one of two exact paths with different nonlinear zero terms.

    Both paths compute the same bilinear polynomial.  The extra terms are
    intentionally nonlinear so a numerical equality check alone is not being
    mistaken for a syntactically linear program.
    """

    if branch not in {0, 1}:
        raise ValueError("branch must be zero or one")
    if not matrix or len(left) != len(matrix):
        raise ValueError("left/matrix shape mismatch")
    width = len(matrix[0])
    if len(right) != width or any(len(row) != width for row in matrix):
        raise ValueError("matrix/right shape mismatch")

    tape = ExactArithmeticTape()
    left_nodes = tuple(tape.input(Fraction(value)) for value in left)
    right_nodes = tuple(tape.input(Fraction(value)) for value in right)
    zero = tape.constant(0)
    output = zero
    for row_index, row in enumerate(matrix):
        for column_index, coefficient in enumerate(row):
            coefficient_node = tape.constant(Fraction(coefficient))
            query_product = tape.mul(
                left_nodes[row_index], right_nodes[column_index]
            )
            term = tape.mul(coefficient_node, query_product)
            output = tape.add(output, term)

    seed = tape.mul(left_nodes[0], right_nodes[0])
    if branch == 0:
        first = tape.mul(seed, seed)
        second = tape.mul(seed, seed)
    else:
        one = tape.constant(1)
        shifted_first = tape.add(seed, one)
        shifted_second = tape.add(seed, one)
        square_first = tape.mul(shifted_first, shifted_first)
        square_second = tape.mul(shifted_second, shifted_second)
        first = tape.mul(square_first, shifted_first)
        second = tape.mul(square_second, shifted_second)
    nonlinear_zero = tape.sub(first, second)
    output = tape.add(output, nonlinear_zero)
    return tape, output, left_nodes


def run_reference_controls(
    *, seed: int = REFERENCE_SEED, cases: int = 64
) -> dict[str, int | bool]:
    """Exercise exact value and ``d/dr = W u`` identities deterministically."""

    if cases <= 0:
        raise ValueError("cases must be positive")
    generator = random.Random(seed)
    value_matches = 0
    gradient_matches = 0
    branch_counts = {0: 0, 1: 0}
    maximum_tape_operations = 0
    for case in range(cases):
        rows = 2 + case % 3
        columns = 2 + (case // 3) % 4
        matrix = tuple(
            tuple(generator.randint(-7, 7) for _ in range(columns))
            for _ in range(rows)
        )
        left = tuple(
            Fraction(generator.randint(-5, 5), generator.randint(1, 5))
            for _ in range(rows)
        )
        right = tuple(
            Fraction(generator.randint(-5, 5), generator.randint(1, 5))
            for _ in range(columns)
        )
        branch = case % 2
        branch_counts[branch] += 1
        tape, output, left_nodes = build_obfuscated_bilinear_tape(
            left, matrix, right, branch=branch
        )
        maximum_tape_operations = max(
            maximum_tape_operations, tape.arithmetic_operations
        )
        if tape.nodes[output].value == exact_bilinear(left, matrix, right):
            value_matches += 1
        if tape.gradient(output, left_nodes) == exact_matvec(matrix, right):
            gradient_matches += 1
    return {
        "seed": seed,
        "cases": cases,
        "branch_zero_cases": branch_counts[0],
        "branch_one_cases": branch_counts[1],
        "exact_value_matches": value_matches,
        "exact_gradient_matches": gradient_matches,
        "maximum_forward_tape_operations": maximum_tape_operations,
        "all_controls_pass": value_matches == cases and gradient_matches == cases,
    }


def derive_audit() -> dict[str, object]:
    """Return the deterministic E0 containment result."""

    controls = run_reference_controls()
    if not controls["all_controls_pass"]:
        raise AssertionError("exact nonlinear bilinear reference controls failed")
    static_fraction = BAUR_STRASSEN_UNIT_COST_FACTOR * P50_TARGET_FRACTION
    return {
        "classification": "E0_EXACT_FIELD_NONLINEAR_BILINEAR_PATH_COLLAPSE",
        "decision": REJECT_DECISION,
        "registered_target": {
            "p50_fraction": str(P50_TARGET_FRACTION),
            "p50_fraction_decimal": float(P50_TARGET_FRACTION),
            "dense_reduction_required": float(1 / P50_TARGET_FRACTION),
        },
        "containment": {
            "bounded_exact_algebraic_branching": (
                "one full-dimensional fixed path computes the same rational function"
            ),
            "bilinear_function": "f_W(r,u)=r^T W u",
            "left_gradient": "gradient_r f_W(r,u)=W u",
            "baur_strassen_unit_cost_factor": BAUR_STRASSEN_UNIT_COST_FACTOR,
            "target_query_implies_static_matvec_fraction_at_most": str(
                static_fraction
            ),
            "target_query_implies_static_matvec_fraction_decimal": float(
                static_fraction
            ),
            "result_type": "NOVELTY_CONTAINMENT_NOT_FINITE_IMPOSSIBILITY",
        },
        "known_structure_audit": {
            "general_bounded_twin_width": (
                "Hamming-coherence predecessor traversal; already F-049"
            ),
            "twin_ordered_rectangle_decomposition": (
                "static exact linear schedule; already F-050/EXP-072B"
            ),
            "grammar_compressed_matvec": (
                "static grammar evaluation proportional to representation; F-050 class"
            ),
        },
        "reference_controls": controls,
        "claim_boundary": {
            "exact_field_arithmetic_as_distinct_adaptive_source": "REJECTED",
            "continuous_algebraic_branching_and_finite_probes": "CONTAINED",
            "finite_word_bitwise_rounding_or_discontinuous_addressing": "OPEN",
            "native_bf16_q4_rounding_semantics": "NOT MODELED",
            "general_cell_probe_lower_bound": "NOT PROVED",
            "static_circuit_meets_registered_target": "NOT PROVED",
            "core_candidate": "NONE",
            "model_forward_calls": 0,
            "hardware_actions": 0,
        },
    }
