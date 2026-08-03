"""Weight-derived bit-exact AIG decision circuits for EXP-053.

The compiler accepts a bounded binary-activation, signed modular linear top-1
operator and emits a structurally hashed AND-inverter graph (AIG). Runtime input
states are not enumerated or stored. Exhaustive input enumeration is permitted
only by validators that compare the compiled circuit with an independent
arithmetic reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Sequence


class BitCircuitError(ValueError):
    """Raised when a circuit or exact arithmetic contract is malformed."""


FALSE_LITERAL = 0
TRUE_LITERAL = 1
_MAGIC = b"VAIG1\0"


def invert(literal: int) -> int:
    return int(literal) ^ 1


@dataclass(frozen=True)
class AIGNode:
    lhs: int
    rhs: int

    def validate(self, maximum_prior_node_id: int) -> None:
        for literal in (self.lhs, self.rhs):
            if literal < 0:
                raise BitCircuitError("AIG literals must be non-negative")
            if literal >> 1 > maximum_prior_node_id:
                raise BitCircuitError("AIG node references a future node")


@dataclass(frozen=True)
class AIGCircuit:
    input_count: int
    outputs: tuple[int, ...]
    nodes: tuple[AIGNode, ...]
    requested_and_count: int
    source_parameter_count: int
    representation_kind: str = "weight_derived_structurally_hashed_aig"

    def validate(self) -> None:
        if self.input_count <= 0:
            raise BitCircuitError("input_count must be positive")
        if not self.outputs:
            raise BitCircuitError("at least one output literal is required")
        if self.requested_and_count < len(self.nodes):
            raise BitCircuitError("unique nodes exceed requested AND nodes")
        if self.source_parameter_count <= 0:
            raise BitCircuitError("source_parameter_count must be positive")
        maximum = self.input_count
        for node in self.nodes:
            node.validate(maximum)
            maximum += 1
        for literal in self.outputs:
            if literal < 0 or literal >> 1 > maximum:
                raise BitCircuitError("output references an invalid node")
        if self.representation_kind != "weight_derived_structurally_hashed_aig":
            raise BitCircuitError("unregistered circuit representation")

    @property
    def input_literals(self) -> tuple[int, ...]:
        return tuple((index + 1) << 1 for index in range(self.input_count))

    @property
    def and_node_count(self) -> int:
        return len(self.nodes)

    @property
    def maximum_node_id(self) -> int:
        return self.input_count + len(self.nodes)

    @property
    def contains_truth_table(self) -> bool:
        return False

    def reachable_and_node_ids(self) -> frozenset[int]:
        self.validate()
        first_and_id = self.input_count + 1
        node_by_id = {
            first_and_id + offset: node for offset, node in enumerate(self.nodes)
        }
        seen: set[int] = set()
        stack = [
            literal >> 1
            for literal in self.outputs
            if literal >> 1 >= first_and_id
        ]
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = node_by_id[node_id]
            for literal in (node.lhs, node.rhs):
                child = literal >> 1
                if child >= first_and_id:
                    stack.append(child)
        return frozenset(seen)

    @property
    def reachable_and_node_count(self) -> int:
        return len(self.reachable_and_node_ids())

    @property
    def query_node_fraction(self) -> float:
        if self.requested_and_count == 0:
            return 0.0
        return self.reachable_and_node_count / self.requested_and_count

    @property
    def raw_bitblast_baseline_bytes(self) -> int:
        # Two uint32 literals per requested AND plus fixed header/output fields.
        return 32 + 4 * len(self.outputs) + 8 * self.requested_and_count

    @property
    def query_reachable_bytes(self) -> int:
        return 32 + 4 * len(self.outputs) + 8 * self.reachable_and_node_count

    @property
    def query_byte_fraction(self) -> float:
        return self.query_reachable_bytes / self.raw_bitblast_baseline_bytes

    def evaluate_scalar(self, input_value: int) -> int:
        self.validate()
        if input_value < 0 or input_value >= 1 << self.input_count:
            raise BitCircuitError("input value outside declared finite domain")
        values = [False] * (self.maximum_node_id + 1)
        for bit in range(self.input_count):
            values[bit + 1] = bool((input_value >> bit) & 1)

        def literal_value(literal: int) -> bool:
            node_id = literal >> 1
            value = False if node_id == 0 else values[node_id]
            return not value if literal & 1 else value

        first_and_id = self.input_count + 1
        for offset, node in enumerate(self.nodes):
            node_id = first_and_id + offset
            values[node_id] = literal_value(node.lhs) and literal_value(node.rhs)
        result = 0
        for bit, literal in enumerate(self.outputs):
            if literal_value(literal):
                result |= 1 << bit
        return result

    def evaluate_packed(
        self, input_patterns: Sequence[int], *, assignment_count: int
    ) -> tuple[int, ...]:
        """Evaluate many assignments represented as little-endian bit masks."""

        self.validate()
        if len(input_patterns) != self.input_count:
            raise BitCircuitError("packed input count mismatch")
        if assignment_count <= 0:
            raise BitCircuitError("assignment_count must be positive")
        mask = (1 << assignment_count) - 1
        values = [0] * (self.maximum_node_id + 1)
        for index, pattern in enumerate(input_patterns, start=1):
            value = int(pattern)
            if value < 0 or value & ~mask:
                raise BitCircuitError("packed input has bits outside assignment mask")
            values[index] = value

        def literal_value(literal: int) -> int:
            node_id = literal >> 1
            value = 0 if node_id == 0 else values[node_id]
            return value ^ mask if literal & 1 else value

        first_and_id = self.input_count + 1
        for offset, node in enumerate(self.nodes):
            values[first_and_id + offset] = literal_value(node.lhs) & literal_value(
                node.rhs
            )
        return tuple(literal_value(literal) for literal in self.outputs)

    def to_bytes(self) -> bytes:
        self.validate()
        if self.maximum_node_id >= 1 << 31:
            raise BitCircuitError("circuit exceeds uint32 literal format")
        header = struct.pack(
            ">6sIIQQQ",
            _MAGIC,
            self.input_count,
            len(self.outputs),
            self.requested_and_count,
            len(self.nodes),
            self.source_parameter_count,
        )
        output_bytes = b"".join(struct.pack(">I", value) for value in self.outputs)
        node_bytes = b"".join(
            struct.pack(">II", node.lhs, node.rhs) for node in self.nodes
        )
        return header + output_bytes + node_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> "AIGCircuit":
        header_size = struct.calcsize(">6sIIQQQ")
        if len(data) < header_size:
            raise BitCircuitError("truncated AIG header")
        magic, inputs, outputs, requested, node_count, source_parameters = struct.unpack(
            ">6sIIQQQ", data[:header_size]
        )
        if magic != _MAGIC:
            raise BitCircuitError("invalid AIG magic")
        expected = header_size + 4 * outputs + 8 * node_count
        if len(data) != expected:
            raise BitCircuitError("AIG byte length mismatch")
        cursor = header_size
        output_literals = tuple(
            struct.unpack(">I", data[cursor + 4 * index : cursor + 4 * index + 4])[0]
            for index in range(outputs)
        )
        cursor += 4 * outputs
        nodes = tuple(
            AIGNode(
                *struct.unpack(
                    ">II", data[cursor + 8 * index : cursor + 8 * index + 8]
                )
            )
            for index in range(node_count)
        )
        circuit = cls(
            input_count=inputs,
            outputs=output_literals,
            nodes=nodes,
            requested_and_count=requested,
            source_parameter_count=source_parameters,
        )
        circuit.validate()
        return circuit


class AIGBuilder:
    """AIG builder with constants, algebraic simplification, and structural hashing."""

    def __init__(self, input_count: int) -> None:
        if input_count <= 0:
            raise BitCircuitError("input_count must be positive")
        self.input_count = input_count
        self.input_literals = tuple((index + 1) << 1 for index in range(input_count))
        self._next_node_id = input_count + 1
        self._nodes: list[AIGNode] = []
        self._hash: dict[tuple[int, int], int] = {}
        self.requested_and_count = 0

    def and_(self, lhs: int, rhs: int) -> int:
        lhs = int(lhs)
        rhs = int(rhs)
        if lhs == FALSE_LITERAL or rhs == FALSE_LITERAL:
            return FALSE_LITERAL
        if lhs == TRUE_LITERAL:
            return rhs
        if rhs == TRUE_LITERAL:
            return lhs
        if lhs == rhs:
            return lhs
        if lhs == invert(rhs):
            return FALSE_LITERAL
        if rhs < lhs:
            lhs, rhs = rhs, lhs
        self.requested_and_count += 1
        key = (lhs, rhs)
        previous = self._hash.get(key)
        if previous is not None:
            return previous << 1
        node_id = self._next_node_id
        self._next_node_id += 1
        self._hash[key] = node_id
        self._nodes.append(AIGNode(lhs, rhs))
        return node_id << 1

    def or_(self, lhs: int, rhs: int) -> int:
        return invert(self.and_(invert(lhs), invert(rhs)))

    def xor(self, lhs: int, rhs: int) -> int:
        return self.or_(
            self.and_(lhs, invert(rhs)), self.and_(invert(lhs), rhs)
        )

    def mux(self, select: int, when_true: int, when_false: int) -> int:
        return self.or_(
            self.and_(select, when_true),
            self.and_(invert(select), when_false),
        )

    @staticmethod
    def constant_bits(value: int, width: int) -> tuple[int, ...]:
        if width <= 0:
            raise BitCircuitError("bit-vector width must be positive")
        unsigned = int(value) & ((1 << width) - 1)
        return tuple(
            TRUE_LITERAL if (unsigned >> bit) & 1 else FALSE_LITERAL
            for bit in range(width)
        )

    def add_bits(
        self, lhs: Sequence[int], rhs: Sequence[int]
    ) -> tuple[int, ...]:
        if not lhs or len(lhs) != len(rhs):
            raise BitCircuitError("bit-vector add width mismatch")
        carry = FALSE_LITERAL
        output: list[int] = []
        for left_bit, right_bit in zip(lhs, rhs):
            pair_xor = self.xor(left_bit, right_bit)
            output.append(self.xor(pair_xor, carry))
            carry = self.or_(
                self.and_(left_bit, right_bit),
                self.and_(carry, pair_xor),
            )
        return tuple(output)

    def conditional_add_constant(
        self, accumulator: Sequence[int], condition: int, value: int
    ) -> tuple[int, ...]:
        constant = self.constant_bits(value, len(accumulator))
        addend = tuple(
            condition if bit == TRUE_LITERAL else FALSE_LITERAL for bit in constant
        )
        return self.add_bits(accumulator, addend)

    def unsigned_greater_than(
        self, lhs: Sequence[int], rhs: Sequence[int]
    ) -> int:
        if not lhs or len(lhs) != len(rhs):
            raise BitCircuitError("unsigned comparator width mismatch")
        greater = FALSE_LITERAL
        equal = TRUE_LITERAL
        for left_bit, right_bit in reversed(tuple(zip(lhs, rhs))):
            greater_here = self.and_(
                equal, self.and_(left_bit, invert(right_bit))
            )
            greater = self.or_(greater, greater_here)
            equal = self.and_(equal, invert(self.xor(left_bit, right_bit)))
        return greater

    def signed_greater_than(
        self, lhs: Sequence[int], rhs: Sequence[int]
    ) -> int:
        if len(lhs) < 2 or len(lhs) != len(rhs):
            raise BitCircuitError("signed comparator requires equal width >=2")
        lhs_sign = lhs[-1]
        rhs_sign = rhs[-1]
        signs_differ = self.xor(lhs_sign, rhs_sign)
        lhs_positive_rhs_negative = self.and_(invert(lhs_sign), rhs_sign)
        same_sign_greater = self.unsigned_greater_than(lhs[:-1], rhs[:-1])
        return self.mux(
            signs_differ, lhs_positive_rhs_negative, same_sign_greater
        )

    def finalize(
        self, outputs: Sequence[int], *, source_parameter_count: int
    ) -> AIGCircuit:
        circuit = AIGCircuit(
            input_count=self.input_count,
            outputs=tuple(int(value) for value in outputs),
            nodes=tuple(self._nodes),
            requested_and_count=self.requested_and_count,
            source_parameter_count=int(source_parameter_count),
        )
        circuit.validate()
        return circuit


@dataclass(frozen=True)
class BinaryLinearTop1Spec:
    weights: tuple[tuple[int, ...], ...]
    biases: tuple[int, ...]
    accumulator_bits: int
    family: str = "unspecified"

    def validate(self) -> None:
        if not self.weights or not self.weights[0]:
            raise BitCircuitError("weights must be non-empty")
        input_count = len(self.weights[0])
        if any(len(row) != input_count for row in self.weights):
            raise BitCircuitError("weight rows must have equal width")
        if len(self.biases) != len(self.weights):
            raise BitCircuitError("bias count must equal class count")
        if len(self.weights) < 2:
            raise BitCircuitError("at least two output classes are required")
        if not 2 <= self.accumulator_bits <= 32:
            raise BitCircuitError("accumulator_bits must lie in [2,32]")
        if not self.family:
            raise BitCircuitError("operator family label is required")
        for value in (*self.biases, *(value for row in self.weights for value in row)):
            if not isinstance(value, int):
                raise BitCircuitError("weights and biases must be integers")

    @property
    def input_count(self) -> int:
        self.validate()
        return len(self.weights[0])

    @property
    def class_count(self) -> int:
        self.validate()
        return len(self.weights)

    @property
    def source_parameter_count(self) -> int:
        return self.class_count * self.input_count + self.class_count

    @property
    def nonzero_weight_count(self) -> int:
        return sum(value != 0 for row in self.weights for value in row)

    def wrapped_scores(self, input_value: int) -> tuple[int, ...]:
        self.validate()
        if input_value < 0 or input_value >= 1 << self.input_count:
            raise BitCircuitError("input value outside declared finite domain")
        mask = (1 << self.accumulator_bits) - 1
        sign = 1 << (self.accumulator_bits - 1)
        scores: list[int] = []
        for row, bias in zip(self.weights, self.biases):
            exact = int(bias) + sum(
                int(weight) * ((input_value >> bit) & 1)
                for bit, weight in enumerate(row)
            )
            unsigned = exact & mask
            scores.append(
                unsigned - (1 << self.accumulator_bits)
                if unsigned & sign
                else unsigned
            )
        return tuple(scores)

    def reference_top1(self, input_value: int) -> int:
        scores = self.wrapped_scores(input_value)
        # Python max returns the first maximum, preserving lower-class tie break.
        return max(range(len(scores)), key=lambda index: scores[index])


@dataclass(frozen=True)
class CompiledDecisionOperator:
    specification: BinaryLinearTop1Spec
    circuit: AIGCircuit

    def validate(self) -> None:
        self.specification.validate()
        self.circuit.validate()
        if self.circuit.input_count != self.specification.input_count:
            raise BitCircuitError("compiled input width mismatch")
        expected_outputs = max(1, math.ceil(math.log2(self.specification.class_count)))
        if len(self.circuit.outputs) != expected_outputs:
            raise BitCircuitError("compiled class-bit width mismatch")
        if self.circuit.source_parameter_count != self.specification.source_parameter_count:
            raise BitCircuitError("compiled source parameter count mismatch")

    def evaluate(self, input_value: int) -> int:
        self.validate()
        value = self.circuit.evaluate_scalar(input_value)
        if value >= self.specification.class_count:
            raise BitCircuitError("circuit produced an invalid class index")
        return value


def compile_binary_linear_top1(
    specification: BinaryLinearTop1Spec,
) -> CompiledDecisionOperator:
    specification.validate()
    builder = AIGBuilder(specification.input_count)
    scores: list[tuple[int, ...]] = []
    for row, bias in zip(specification.weights, specification.biases):
        accumulator = builder.constant_bits(bias, specification.accumulator_bits)
        for input_literal, weight in zip(builder.input_literals, row):
            if weight != 0:
                accumulator = builder.conditional_add_constant(
                    accumulator, input_literal, weight
                )
        scores.append(accumulator)

    class_bit_width = max(1, math.ceil(math.log2(specification.class_count)))
    best_score = scores[0]
    best_class = builder.constant_bits(0, class_bit_width)
    for class_index in range(1, specification.class_count):
        better = builder.signed_greater_than(scores[class_index], best_score)
        best_score = tuple(
            builder.mux(better, candidate, previous)
            for candidate, previous in zip(scores[class_index], best_score)
        )
        class_bits = builder.constant_bits(class_index, class_bit_width)
        best_class = tuple(
            builder.mux(better, candidate, previous)
            for candidate, previous in zip(class_bits, best_class)
        )

    compiled = CompiledDecisionOperator(
        specification=specification,
        circuit=builder.finalize(
            best_class,
            source_parameter_count=specification.source_parameter_count,
        ),
    )
    compiled.validate()
    return compiled
