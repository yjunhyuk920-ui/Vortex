"""Exact prototype plus sparse-residual execution plans for EXP-056.

Each binary-activation weight column is reconstructed exactly as one shared
prototype plus a sparse list of residual class scalars.  Plans are derived
only from the immutable source weights; no runtime states are enumerated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import struct
from typing import Iterable, Sequence

from vortex_runtime.bit_circuit import BinaryLinearTop1Spec


class PrototypeResidualError(ValueError):
    """Raised when an exact prototype/residual plan is malformed."""


_MAGIC = b"VPRD1\0"
_REPRESENTATION = "weight_derived_exact_prototype_sparse_residual"
_WORD_BITS = 64


def _signed_wrap(value: int, width: int) -> int:
    unsigned = int(value) & ((1 << width) - 1)
    sign = 1 << (width - 1)
    return unsigned - (1 << width) if unsigned & sign else unsigned


def _column_vectors(specification: BinaryLinearTop1Spec) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(row[index] for row in specification.weights)
        for index in range(specification.input_count)
    )


def _residual_key(column: tuple[int, ...], prototype: tuple[int, ...]) -> tuple[int, int]:
    residual = tuple(value - base for value, base in zip(column, prototype))
    return sum(value != 0 for value in residual), sum(abs(value) for value in residual)


def _candidate_order(
    columns: Sequence[tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    if not columns:
        raise PrototypeResidualError("at least one source column is required")
    zero = (0,) * len(columns[0])
    counts = Counter(columns)
    candidates = set(columns)
    candidates.add(zero)
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                -counts[item],
                sum(value != 0 for value in item),
                sum(abs(value) for value in item),
                item,
            ),
        )
    )


def _total_residual_cost(
    columns: Sequence[tuple[int, ...]],
    prototypes: Sequence[tuple[int, ...]],
) -> tuple[int, int]:
    nonzero = 0
    magnitude = 0
    for column in columns:
        best = min(_residual_key(column, prototype) for prototype in prototypes)
        nonzero += best[0]
        magnitude += best[1]
    return nonzero, magnitude


@dataclass(frozen=True)
class ResidualEntry:
    class_index: int
    value: int


@dataclass(frozen=True)
class ResidualColumn:
    input_index: int
    entries: tuple[ResidualEntry, ...]

    def validate(self, *, input_count: int, class_count: int) -> None:
        if self.input_index < 0 or self.input_index >= input_count:
            raise PrototypeResidualError("residual input index outside domain")
        if not self.entries:
            raise PrototypeResidualError("empty residual column is not stored")
        indexes = tuple(entry.class_index for entry in self.entries)
        if indexes != tuple(sorted(indexes)) or len(set(indexes)) != len(indexes):
            raise PrototypeResidualError("residual class indexes must be unique and sorted")
        if any(index < 0 or index >= class_count for index in indexes):
            raise PrototypeResidualError("residual class index outside domain")
        if any(entry.value == 0 for entry in self.entries):
            raise PrototypeResidualError("zero residual scalar must not be stored")


@dataclass(frozen=True)
class PrototypeGroup:
    prototype: tuple[int, ...]
    member_indices: tuple[int, ...]

    def validate(self, *, input_count: int, class_count: int) -> None:
        if len(self.prototype) != class_count:
            raise PrototypeResidualError("prototype class width mismatch")
        if not self.member_indices:
            raise PrototypeResidualError("prototype group must have members")
        if self.member_indices != tuple(sorted(self.member_indices)):
            raise PrototypeResidualError("prototype members must be sorted")
        if len(set(self.member_indices)) != len(self.member_indices):
            raise PrototypeResidualError("duplicate prototype member")
        if any(index < 0 or index >= input_count for index in self.member_indices):
            raise PrototypeResidualError("prototype member outside input domain")

    @property
    def is_zero(self) -> bool:
        return not any(self.prototype)

    @property
    def nonzero_scalar_count(self) -> int:
        return sum(value != 0 for value in self.prototype)

    @property
    def membership_word_count(self) -> int:
        return len({index // _WORD_BITS for index in self.member_indices})

    def mask(self) -> int:
        return sum(1 << index for index in self.member_indices)


@dataclass(frozen=True)
class PrototypeResidualPlan:
    specification: BinaryLinearTop1Spec
    groups: tuple[PrototypeGroup, ...]
    residual_columns: tuple[ResidualColumn, ...]
    strategy: str
    requested_prototype_count: int
    compile_operation_count: int
    representation_kind: str = _REPRESENTATION

    def validate(self) -> None:
        self.specification.validate()
        if self.representation_kind != _REPRESENTATION:
            raise PrototypeResidualError("unregistered representation kind")
        if self.strategy not in {"frequency", "greedy"}:
            raise PrototypeResidualError("unsupported prototype strategy")
        if self.requested_prototype_count <= 0:
            raise PrototypeResidualError("requested prototype count must be positive")
        if self.compile_operation_count < 0:
            raise PrototypeResidualError("compile operation count must be nonnegative")
        if not self.groups:
            raise PrototypeResidualError("at least one prototype group is required")

        prototype_for: dict[int, tuple[int, ...]] = {}
        seen_prototypes: set[tuple[int, ...]] = set()
        for group in self.groups:
            group.validate(
                input_count=self.specification.input_count,
                class_count=self.specification.class_count,
            )
            if group.prototype in seen_prototypes:
                raise PrototypeResidualError("duplicate prototype group")
            seen_prototypes.add(group.prototype)
            for index in group.member_indices:
                if index in prototype_for:
                    raise PrototypeResidualError("prototype groups overlap")
                prototype_for[index] = group.prototype
        if set(prototype_for) != set(range(self.specification.input_count)):
            raise PrototypeResidualError("prototype groups do not partition columns")

        residual_for: dict[int, tuple[ResidualEntry, ...]] = {}
        for residual in self.residual_columns:
            residual.validate(
                input_count=self.specification.input_count,
                class_count=self.specification.class_count,
            )
            if residual.input_index in residual_for:
                raise PrototypeResidualError("duplicate residual column")
            residual_for[residual.input_index] = residual.entries

        source = _column_vectors(self.specification)
        for index, expected in enumerate(source):
            reconstructed = list(prototype_for[index])
            for entry in residual_for.get(index, ()):
                reconstructed[entry.class_index] += entry.value
            if tuple(reconstructed) != expected:
                raise PrototypeResidualError("prototype plus residual does not reconstruct source")

    @property
    def contains_truth_table(self) -> bool:
        return False

    @property
    def prototype_count(self) -> int:
        return len(self.groups)

    @property
    def active_prototype_groups(self) -> tuple[PrototypeGroup, ...]:
        return tuple(group for group in self.groups if not group.is_zero)

    @property
    def residual_scalar_count(self) -> int:
        return sum(len(column.entries) for column in self.residual_columns)

    @property
    def residual_column_count(self) -> int:
        return len(self.residual_columns)

    @property
    def prototype_scalar_count(self) -> int:
        return sum(group.nonzero_scalar_count for group in self.active_prototype_groups)

    @property
    def membership_word_count(self) -> int:
        return sum(group.membership_word_count for group in self.active_prototype_groups)

    def wrapped_scores(self, input_value: int) -> tuple[int, ...]:
        self.validate()
        if input_value < 0 or input_value >= 1 << self.specification.input_count:
            raise PrototypeResidualError("input value outside finite domain")
        scores = [int(value) for value in self.specification.biases]
        for group in self.active_prototype_groups:
            count = (input_value & group.mask()).bit_count()
            if count:
                for class_index, value in enumerate(group.prototype):
                    scores[class_index] += count * value
        for residual in self.residual_columns:
            if (input_value >> residual.input_index) & 1:
                for entry in residual.entries:
                    scores[entry.class_index] += entry.value
        return tuple(
            _signed_wrap(value, self.specification.accumulator_bits)
            for value in scores
        )

    def evaluate_scalar(self, input_value: int) -> int:
        scores = self.wrapped_scores(input_value)
        return max(range(len(scores)), key=lambda index: scores[index])

    def evaluate_packed(
        self, input_patterns: Sequence[int], *, assignment_count: int
    ) -> tuple[int, ...]:
        self.validate()
        if len(input_patterns) != self.specification.input_count:
            raise PrototypeResidualError("packed input width mismatch")
        if assignment_count <= 0:
            raise PrototypeResidualError("assignment_count must be positive")
        mask = (1 << assignment_count) - 1
        patterns = tuple(int(value) for value in input_patterns)
        if any(value < 0 or value & ~mask for value in patterns):
            raise PrototypeResidualError("packed input outside assignment mask")
        width = max(1, math.ceil(math.log2(self.specification.class_count)))
        outputs = [0] * width
        for assignment in range(assignment_count):
            input_value = sum(
                ((pattern >> assignment) & 1) << bit
                for bit, pattern in enumerate(patterns)
            )
            result = self.evaluate_scalar(input_value)
            for bit in range(width):
                if (result >> bit) & 1:
                    outputs[bit] |= 1 << assignment
        return tuple(outputs)

    def to_bytes(self) -> bytes:
        self.validate()
        strategy_code = 0 if self.strategy == "frequency" else 1
        header = struct.pack(
            ">6sIIIIIIQ",
            _MAGIC,
            self.specification.input_count,
            self.specification.class_count,
            self.specification.accumulator_bits,
            len(self.groups),
            len(self.residual_columns),
            self.requested_prototype_count,
            self.compile_operation_count,
        ) + struct.pack(">B", strategy_code)
        payload = bytearray(header)
        for bias in self.specification.biases:
            payload.extend(struct.pack(">q", bias))
        for group in self.groups:
            for value in group.prototype:
                payload.extend(struct.pack(">q", value))
            payload.extend(struct.pack(">I", len(group.member_indices)))
            for index in group.member_indices:
                payload.extend(struct.pack(">I", index))
        for residual in self.residual_columns:
            payload.extend(struct.pack(">II", residual.input_index, len(residual.entries)))
            for entry in residual.entries:
                payload.extend(struct.pack(">Iq", entry.class_index, entry.value))
        return bytes(payload)

    @classmethod
    def from_bytes(cls, data: bytes) -> "PrototypeResidualPlan":
        header_size = struct.calcsize(">6sIIIIIIQ") + 1
        if len(data) < header_size:
            raise PrototypeResidualError("truncated plan header")
        unpacked = struct.unpack(">6sIIIIIIQ", data[: header_size - 1])
        magic, n, classes, width, group_count, residual_count, requested, compile_ops = unpacked
        strategy_code = data[header_size - 1]
        if magic != _MAGIC or strategy_code not in (0, 1):
            raise PrototypeResidualError("invalid plan header")
        cursor = header_size
        needed = cursor + 8 * classes
        if len(data) < needed:
            raise PrototypeResidualError("truncated biases")
        biases = tuple(
            struct.unpack(">q", data[cursor + 8 * i : cursor + 8 * i + 8])[0]
            for i in range(classes)
        )
        cursor = needed
        groups: list[PrototypeGroup] = []
        prototype_for: list[tuple[int, ...] | None] = [None] * n
        for _ in range(group_count):
            needed = cursor + 8 * classes + 4
            if len(data) < needed:
                raise PrototypeResidualError("truncated prototype group")
            prototype = tuple(
                struct.unpack(">q", data[cursor + 8 * i : cursor + 8 * i + 8])[0]
                for i in range(classes)
            )
            cursor += 8 * classes
            member_count = struct.unpack(">I", data[cursor : cursor + 4])[0]
            cursor += 4
            needed = cursor + 4 * member_count
            if len(data) < needed:
                raise PrototypeResidualError("truncated prototype members")
            members = tuple(
                struct.unpack(">I", data[cursor + 4 * i : cursor + 4 * i + 4])[0]
                for i in range(member_count)
            )
            cursor = needed
            group = PrototypeGroup(prototype, members)
            groups.append(group)
            for index in members:
                if index >= n or prototype_for[index] is not None:
                    raise PrototypeResidualError("invalid prototype partition")
                prototype_for[index] = prototype
        residuals: list[ResidualColumn] = []
        residual_map: dict[int, tuple[ResidualEntry, ...]] = {}
        for _ in range(residual_count):
            if len(data) < cursor + 8:
                raise PrototypeResidualError("truncated residual column")
            input_index, entry_count = struct.unpack(">II", data[cursor : cursor + 8])
            cursor += 8
            entries: list[ResidualEntry] = []
            for _ in range(entry_count):
                if len(data) < cursor + 12:
                    raise PrototypeResidualError("truncated residual entry")
                class_index, value = struct.unpack(">Iq", data[cursor : cursor + 12])
                cursor += 12
                entries.append(ResidualEntry(class_index, value))
            residual = ResidualColumn(input_index, tuple(entries))
            residuals.append(residual)
            if input_index in residual_map:
                raise PrototypeResidualError("duplicate residual input")
            residual_map[input_index] = residual.entries
        if cursor != len(data) or any(value is None for value in prototype_for):
            raise PrototypeResidualError("plan length or partition mismatch")
        columns: list[tuple[int, ...]] = []
        for index, prototype in enumerate(prototype_for):
            values = list(prototype or ())
            for entry in residual_map.get(index, ()):
                values[entry.class_index] += entry.value
            columns.append(tuple(values))
        weights = tuple(
            tuple(columns[index][class_index] for index in range(n))
            for class_index in range(classes)
        )
        spec = BinaryLinearTop1Spec(
            weights=weights,
            biases=biases,
            accumulator_bits=width,
            family="deserialized_prototype_residual",
        )
        plan = cls(
            specification=spec,
            groups=tuple(groups),
            residual_columns=tuple(residuals),
            strategy="frequency" if strategy_code == 0 else "greedy",
            requested_prototype_count=requested,
            compile_operation_count=compile_ops,
        )
        plan.validate()
        return plan


def _select_prototypes(
    columns: Sequence[tuple[int, ...]],
    *,
    requested_count: int,
    strategy: str,
) -> tuple[tuple[tuple[int, ...], ...], int]:
    candidates = _candidate_order(columns)
    count = min(requested_count, len(candidates))
    classes = len(columns[0])
    compile_operations = len(columns) * classes
    if strategy == "frequency":
        compile_operations += len(candidates) * max(1, math.ceil(math.log2(len(candidates) + 1)))
        return candidates[:count], compile_operations
    if strategy != "greedy":
        raise PrototypeResidualError("unsupported prototype strategy")

    selected: list[tuple[int, ...]] = []
    remaining = list(candidates)
    while remaining and len(selected) < count:
        best_candidate = None
        best_key = None
        for candidate in remaining:
            trial = tuple(selected + [candidate])
            cost = _total_residual_cost(columns, trial)
            compile_operations += len(columns) * len(trial) * classes * 2
            key = (cost[0], cost[1], candidate)
            if best_key is None or key < best_key:
                best_key = key
                best_candidate = candidate
        assert best_candidate is not None
        selected.append(best_candidate)
        remaining.remove(best_candidate)
    return tuple(selected), compile_operations


def compile_prototype_residual_plan(
    specification: BinaryLinearTop1Spec,
    *,
    requested_prototype_count: int,
    strategy: str,
) -> PrototypeResidualPlan:
    specification.validate()
    if requested_prototype_count <= 0:
        raise PrototypeResidualError("requested prototype count must be positive")
    columns = _column_vectors(specification)
    prototypes, compile_operations = _select_prototypes(
        columns,
        requested_count=requested_prototype_count,
        strategy=strategy,
    )
    assignments: list[int] = []
    compile_operations += len(columns) * len(prototypes) * specification.class_count * 2
    for column in columns:
        assignments.append(
            min(
                range(len(prototypes)),
                key=lambda index: (*_residual_key(column, prototypes[index]), index),
            )
        )

    members: list[list[int]] = [[] for _ in prototypes]
    residuals: list[ResidualColumn] = []
    for input_index, (column, prototype_index) in enumerate(zip(columns, assignments)):
        prototype = prototypes[prototype_index]
        members[prototype_index].append(input_index)
        entries = tuple(
            ResidualEntry(class_index, value - base)
            for class_index, (value, base) in enumerate(zip(column, prototype))
            if value != base
        )
        if entries:
            residuals.append(ResidualColumn(input_index, entries))

    groups = tuple(
        PrototypeGroup(prototype, tuple(group_members))
        for prototype, group_members in zip(prototypes, members)
        if group_members
    )
    plan = PrototypeResidualPlan(
        specification=specification,
        groups=groups,
        residual_columns=tuple(residuals),
        strategy=strategy,
        requested_prototype_count=requested_prototype_count,
        compile_operation_count=compile_operations,
    )
    plan.validate()
    return plan
