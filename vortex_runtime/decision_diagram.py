"""Exact reduced ordered multi-terminal decision diagrams for EXP-054.

The compiler uses immutable signed modular linear-top1 weights and exact partial
score states. Runtime input assignments are not stored as a truth table. Each
query follows one root-to-terminal path in a reduced ordered diagram.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import time
from typing import Sequence

from vortex_runtime.bit_circuit import BinaryLinearTop1Spec, BitCircuitError


class DecisionDiagramError(ValueError):
    """Raised when an exact decision-diagram contract is malformed."""


_MAGIC = b"VMTD1\0"


@dataclass(frozen=True)
class DecisionNode:
    variable: int
    low: int
    high: int


@dataclass(frozen=True)
class ReducedDecisionDiagram:
    input_count: int
    class_count: int
    variable_order: tuple[int, ...]
    root: int
    nodes: tuple[DecisionNode, ...]
    compile_state_visits: int
    memoized_state_count: int
    source_parameter_count: int
    representation_kind: str = "weight_derived_reduced_ordered_multi_terminal_diagram"

    def validate(self) -> None:
        if self.input_count <= 0 or self.class_count < 2:
            raise DecisionDiagramError("invalid input or terminal count")
        if tuple(sorted(self.variable_order)) != tuple(range(self.input_count)):
            raise DecisionDiagramError("variable_order must be a permutation")
        if self.compile_state_visits <= 0 or self.memoized_state_count <= 0:
            raise DecisionDiagramError("compile accounting must be positive")
        if self.source_parameter_count <= 0:
            raise DecisionDiagramError("source_parameter_count must be positive")
        if self.representation_kind != (
            "weight_derived_reduced_ordered_multi_terminal_diagram"
        ):
            raise DecisionDiagramError("unregistered representation")
        maximum_reference = self.class_count + len(self.nodes) - 1
        if self.root < 0 or self.root > maximum_reference:
            raise DecisionDiagramError("invalid root reference")
        order_position = {variable: index for index, variable in enumerate(self.variable_order)}
        for index, node in enumerate(self.nodes):
            node_reference = self.class_count + index
            if node.variable not in order_position:
                raise DecisionDiagramError("node variable outside input domain")
            if node.low == node.high:
                raise DecisionDiagramError("unreduced equal-child node")
            for child in (node.low, node.high):
                if child < 0 or child >= node_reference:
                    raise DecisionDiagramError("node child must be terminal or earlier node")
                if child >= self.class_count:
                    child_node = self.nodes[child - self.class_count]
                    if order_position[child_node.variable] <= order_position[node.variable]:
                        raise DecisionDiagramError("diagram violates variable ordering")

    @property
    def contains_truth_table(self) -> bool:
        return False

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def terminal_count(self) -> int:
        reachable = {self.evaluate_scalar(value)[0] for value in range(1 << min(self.input_count, 12))}
        return len(reachable)

    @property
    def serialized_bytes(self) -> int:
        return len(self.to_bytes())

    def evaluate_scalar(self, input_value: int) -> tuple[int, int]:
        self.validate()
        if input_value < 0 or input_value >= 1 << self.input_count:
            raise DecisionDiagramError("input outside declared domain")
        reference = self.root
        probes = 0
        while reference >= self.class_count:
            node = self.nodes[reference - self.class_count]
            probes += 1
            reference = node.high if (input_value >> node.variable) & 1 else node.low
        return reference, probes

    def to_bytes(self) -> bytes:
        self.validate()
        if self.class_count + len(self.nodes) >= 1 << 32:
            raise DecisionDiagramError("diagram exceeds uint32 reference format")
        header = struct.pack(
            ">6sIIIIQQQ",
            _MAGIC,
            self.input_count,
            self.class_count,
            self.root,
            len(self.nodes),
            self.compile_state_visits,
            self.memoized_state_count,
            self.source_parameter_count,
        )
        order_bytes = b"".join(struct.pack(">I", value) for value in self.variable_order)
        node_bytes = b"".join(
            struct.pack(">III", node.variable, node.low, node.high)
            for node in self.nodes
        )
        return header + order_bytes + node_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> "ReducedDecisionDiagram":
        header_format = ">6sIIIIQQQ"
        header_size = struct.calcsize(header_format)
        if len(data) < header_size:
            raise DecisionDiagramError("truncated diagram header")
        magic, input_count, class_count, root, node_count, visits, memoized, source = struct.unpack(
            header_format, data[:header_size]
        )
        if magic != _MAGIC:
            raise DecisionDiagramError("invalid diagram magic")
        expected = header_size + 4 * input_count + 12 * node_count
        if len(data) != expected:
            raise DecisionDiagramError("diagram byte length mismatch")
        cursor = header_size
        order = tuple(
            struct.unpack(">I", data[cursor + 4 * index : cursor + 4 * index + 4])[0]
            for index in range(input_count)
        )
        cursor += 4 * input_count
        nodes = tuple(
            DecisionNode(
                *struct.unpack(
                    ">III", data[cursor + 12 * index : cursor + 12 * index + 12]
                )
            )
            for index in range(node_count)
        )
        diagram = cls(
            input_count=input_count,
            class_count=class_count,
            variable_order=order,
            root=root,
            nodes=nodes,
            compile_state_visits=visits,
            memoized_state_count=memoized,
            source_parameter_count=source,
        )
        diagram.validate()
        return diagram


@dataclass(frozen=True)
class DiagramCompilation:
    variable_order: tuple[int, ...]
    diagram: ReducedDecisionDiagram | None
    ceiling_hit: bool
    compile_state_visits: int
    memoized_state_count: int
    unique_node_count: int
    compile_elapsed_ns: int
    ceiling: int

    def validate(self) -> None:
        if self.compile_state_visits <= 0 or self.ceiling <= 0:
            raise DecisionDiagramError("invalid compilation accounting")
        if self.memoized_state_count < 0 or self.unique_node_count < 0:
            raise DecisionDiagramError("negative compilation counts")
        if self.ceiling_hit == (self.diagram is not None):
            raise DecisionDiagramError("ceiling/diagram state inconsistent")
        if self.diagram is not None:
            self.diagram.validate()
            if self.diagram.variable_order != self.variable_order:
                raise DecisionDiagramError("compilation order mismatch")


class _CompileCeiling(RuntimeError):
    pass


def natural_variable_order(input_count: int) -> tuple[int, ...]:
    if input_count <= 0:
        raise DecisionDiagramError("input_count must be positive")
    return tuple(range(input_count))


def weight_magnitude_variable_order(
    specification: BinaryLinearTop1Spec,
) -> tuple[int, ...]:
    specification.validate()
    strengths = [
        sum(abs(row[index]) for row in specification.weights)
        for index in range(specification.input_count)
    ]
    return tuple(sorted(range(specification.input_count), key=lambda index: (-strengths[index], index)))


def compile_reduced_decision_diagram(
    specification: BinaryLinearTop1Spec,
    *,
    variable_order: Sequence[int],
    compile_state_ceiling: int,
) -> DiagramCompilation:
    """Compile exact partial-score states into a reduced ordered diagram."""

    try:
        specification.validate()
    except BitCircuitError as exc:
        raise DecisionDiagramError(str(exc)) from exc
    order = tuple(int(value) for value in variable_order)
    if tuple(sorted(order)) != tuple(range(specification.input_count)):
        raise DecisionDiagramError("variable_order must be a permutation")
    if compile_state_ceiling <= 0:
        raise DecisionDiagramError("compile_state_ceiling must be positive")

    mask = (1 << specification.accumulator_bits) - 1
    memo: dict[tuple[int, tuple[int, ...]], int] = {}
    unique: dict[tuple[int, int, int], int] = {}
    nodes: list[DecisionNode] = []
    visits = 0
    started = time.perf_counter_ns()

    def compile_state(position: int, unsigned_scores: tuple[int, ...]) -> int:
        nonlocal visits
        key = (position, unsigned_scores)
        previous = memo.get(key)
        if previous is not None:
            return previous
        visits += 1
        if visits > compile_state_ceiling:
            raise _CompileCeiling
        if position == specification.input_count:
            sign = 1 << (specification.accumulator_bits - 1)
            signed_scores = tuple(
                value - (1 << specification.accumulator_bits)
                if value & sign
                else value
                for value in unsigned_scores
            )
            result = max(
                range(specification.class_count),
                key=lambda index: signed_scores[index],
            )
            memo[key] = result
            return result

        variable = order[position]
        low = compile_state(position + 1, unsigned_scores)
        high_scores = tuple(
            (score + specification.weights[class_index][variable]) & mask
            for class_index, score in enumerate(unsigned_scores)
        )
        high = compile_state(position + 1, high_scores)
        if low == high:
            result = low
        else:
            node_key = (variable, low, high)
            result = unique.get(node_key, -1)
            if result < 0:
                if len(nodes) >= compile_state_ceiling:
                    raise _CompileCeiling
                result = specification.class_count + len(nodes)
                unique[node_key] = result
                nodes.append(DecisionNode(variable, low, high))
        memo[key] = result
        return result

    initial_scores = tuple(int(value) & mask for value in specification.biases)
    try:
        root = compile_state(0, initial_scores)
    except _CompileCeiling:
        compilation = DiagramCompilation(
            variable_order=order,
            diagram=None,
            ceiling_hit=True,
            compile_state_visits=visits,
            memoized_state_count=len(memo),
            unique_node_count=len(nodes),
            compile_elapsed_ns=time.perf_counter_ns() - started,
            ceiling=compile_state_ceiling,
        )
        compilation.validate()
        return compilation

    diagram = ReducedDecisionDiagram(
        input_count=specification.input_count,
        class_count=specification.class_count,
        variable_order=order,
        root=root,
        nodes=tuple(nodes),
        compile_state_visits=visits,
        memoized_state_count=len(memo),
        source_parameter_count=specification.source_parameter_count,
    )
    compilation = DiagramCompilation(
        variable_order=order,
        diagram=diagram,
        ceiling_hit=False,
        compile_state_visits=visits,
        memoized_state_count=len(memo),
        unique_node_count=len(nodes),
        compile_elapsed_ns=time.perf_counter_ns() - started,
        ceiling=compile_state_ceiling,
    )
    compilation.validate()
    return compilation


def select_smaller_completed_compilation(
    compilations: Sequence[DiagramCompilation],
) -> DiagramCompilation | None:
    completed = [item for item in compilations if item.diagram is not None]
    if not completed:
        return None
    return min(
        completed,
        key=lambda item: (
            item.diagram.serialized_bytes,
            item.diagram.node_count,
            item.variable_order,
        ),
    )
