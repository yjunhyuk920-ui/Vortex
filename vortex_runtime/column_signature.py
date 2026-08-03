"""Exact column-signature popcount aggregation for EXP-055.

This module compiles a bounded binary-activation signed modular linear top-1
operator into exact groups of identical or exact-negated weight columns.
Runtime states are never enumerated or stored.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Sequence

from vortex_runtime.bit_circuit import BinaryLinearTop1Spec


class ColumnSignatureError(ValueError):
    """Raised when an exact column-signature plan is malformed."""


_MAGIC = b"VCOL1\0"
_REPRESENTATION = "weight_derived_exact_column_signature_popcount"
_WORD_BITS = 64


def _signed_wrap(value: int, width: int) -> int:
    mask = (1 << width) - 1
    unsigned = int(value) & mask
    sign = 1 << (width - 1)
    return unsigned - (1 << width) if unsigned & sign else unsigned


def _canonical_signature(
    signature: tuple[int, ...],
) -> tuple[tuple[int, ...], int]:
    negated = tuple(-value for value in signature)
    if signature == negated:
        return signature, 1
    if signature < negated:
        return signature, 1
    return negated, -1


@dataclass(frozen=True)
class ColumnGroup:
    signature: tuple[int, ...]
    positive_indices: tuple[int, ...]
    negative_indices: tuple[int, ...]

    @property
    def member_count(self) -> int:
        return len(self.positive_indices) + len(self.negative_indices)

    def validate(self, *, input_count: int, class_count: int) -> None:
        if len(self.signature) != class_count:
            raise ColumnSignatureError("signature class width mismatch")
        if self.member_count <= 0:
            raise ColumnSignatureError("column group must contain at least one member")
        members = self.positive_indices + self.negative_indices
        if len(set(members)) != len(members):
            raise ColumnSignatureError("column group member indices overlap")
        if any(index < 0 or index >= input_count for index in members):
            raise ColumnSignatureError("column group member outside input domain")
        if tuple(sorted(self.positive_indices)) != self.positive_indices:
            raise ColumnSignatureError("positive member indices must be sorted")
        if tuple(sorted(self.negative_indices)) != self.negative_indices:
            raise ColumnSignatureError("negative member indices must be sorted")

    def positive_mask(self) -> int:
        return sum(1 << index for index in self.positive_indices)

    def negative_mask(self) -> int:
        return sum(1 << index for index in self.negative_indices)

    def active_signed_count(self, input_value: int) -> int:
        positive = (input_value & self.positive_mask()).bit_count()
        negative = (input_value & self.negative_mask()).bit_count()
        return positive - negative

    def active_word_count(self) -> int:
        chunks: set[tuple[int, int]] = set()
        for polarity, indices in enumerate(
            (self.positive_indices, self.negative_indices)
        ):
            for index in indices:
                chunks.add((polarity, index // _WORD_BITS))
        return len(chunks)


@dataclass(frozen=True)
class ColumnSignaturePlan:
    specification: BinaryLinearTop1Spec
    groups: tuple[ColumnGroup, ...]
    sign_canonical: bool
    representation_kind: str = _REPRESENTATION

    def validate(self) -> None:
        self.specification.validate()
        if self.representation_kind != _REPRESENTATION:
            raise ColumnSignatureError("unregistered representation kind")
        if not self.groups:
            raise ColumnSignatureError("at least one column group is required")
        observed: dict[int, tuple[int, ...]] = {}
        signatures: set[tuple[int, ...]] = set()
        for group in self.groups:
            group.validate(
                input_count=self.specification.input_count,
                class_count=self.specification.class_count,
            )
            if group.signature in signatures:
                raise ColumnSignatureError("duplicate compiled group signature")
            signatures.add(group.signature)
            if self.sign_canonical:
                canonical, polarity = _canonical_signature(group.signature)
                if canonical != group.signature or polarity != 1:
                    raise ColumnSignatureError("non-canonical group signature")
            elif group.negative_indices:
                raise ColumnSignatureError(
                    "negative members require sign-canonical compilation"
                )
            for index in group.positive_indices:
                observed[index] = group.signature
            for index in group.negative_indices:
                observed[index] = tuple(-value for value in group.signature)

        if set(observed) != set(range(self.specification.input_count)):
            raise ColumnSignatureError("compiled groups do not partition all columns")
        for index in range(self.specification.input_count):
            expected = tuple(
                row[index] for row in self.specification.weights
            )
            if observed[index] != expected:
                raise ColumnSignatureError("compiled group does not reconstruct source")

    @property
    def contains_truth_table(self) -> bool:
        return False

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def scalar_bytes(self) -> int:
        return math.ceil(self.specification.accumulator_bits / 8)

    @property
    def source_weight_bytes(self) -> int:
        return self.specification.source_parameter_count * self.scalar_bytes

    @property
    def membership_word_count(self) -> int:
        return sum(group.active_word_count() for group in self.groups)

    @property
    def baseline_operation_count(self) -> int:
        return self.specification.class_count * self.specification.input_count

    @property
    def grouped_operation_count(self) -> int:
        # One bit-mask AND plus one popcount per active membership word, and
        # one exact scalar multiply plus add per class/group contribution.
        return (
            2 * self.membership_word_count
            + 2 * self.specification.class_count * self.group_count
        )

    @property
    def operation_fraction(self) -> float:
        return self.grouped_operation_count / self.baseline_operation_count

    @property
    def baseline_query_bytes(self) -> int:
        input_words = math.ceil(self.specification.input_count / _WORD_BITS)
        return self.source_weight_bytes + 8 * input_words

    @property
    def grouped_query_bytes(self) -> int:
        signature_bytes = (
            self.group_count
            * self.specification.class_count
            * self.scalar_bytes
        )
        bias_bytes = self.specification.class_count * self.scalar_bytes
        membership_bytes = self.membership_word_count * 8
        return signature_bytes + bias_bytes + membership_bytes

    @property
    def query_byte_fraction(self) -> float:
        return self.grouped_query_bytes / self.baseline_query_bytes

    @property
    def serialized_bytes(self) -> int:
        return len(self.to_bytes())

    @property
    def storage_fraction(self) -> float:
        return self.serialized_bytes / self.source_weight_bytes

    def wrapped_scores(self, input_value: int) -> tuple[int, ...]:
        self.validate()
        if input_value < 0 or input_value >= 1 << self.specification.input_count:
            raise ColumnSignatureError("input value outside declared finite domain")
        scores = [int(value) for value in self.specification.biases]
        for group in self.groups:
            count = group.active_signed_count(input_value)
            if count == 0:
                continue
            for class_index, weight in enumerate(group.signature):
                scores[class_index] += count * weight
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
        """Evaluate packed assignments and return one bit-mask per class-index bit."""

        self.validate()
        if len(input_patterns) != self.specification.input_count:
            raise ColumnSignatureError("packed input width mismatch")
        if assignment_count <= 0:
            raise ColumnSignatureError("assignment_count must be positive")
        mask = (1 << assignment_count) - 1
        patterns = tuple(int(value) for value in input_patterns)
        if any(value < 0 or value & ~mask for value in patterns):
            raise ColumnSignatureError("packed input has bits outside assignment mask")
        output_width = max(
            1, math.ceil(math.log2(self.specification.class_count))
        )
        outputs = [0] * output_width
        for assignment in range(assignment_count):
            input_value = sum(
                ((pattern >> assignment) & 1) << bit
                for bit, pattern in enumerate(patterns)
            )
            result = self.evaluate_scalar(input_value)
            for bit in range(output_width):
                if (result >> bit) & 1:
                    outputs[bit] |= 1 << assignment
        return tuple(outputs)

    def to_bytes(self) -> bytes:
        self.validate()
        flags = 1 if self.sign_canonical else 0
        header = struct.pack(
            ">6sIIIII",
            _MAGIC,
            self.specification.input_count,
            self.specification.class_count,
            self.specification.accumulator_bits,
            self.group_count,
            flags,
        )
        biases = b"".join(
            struct.pack(">q", value) for value in self.specification.biases
        )
        payload = bytearray(header + biases)
        for group in self.groups:
            for value in group.signature:
                payload.extend(struct.pack(">q", value))
            payload.extend(
                struct.pack(
                    ">II",
                    len(group.positive_indices),
                    len(group.negative_indices),
                )
            )
            for index in group.positive_indices + group.negative_indices:
                payload.extend(struct.pack(">I", index))
        return bytes(payload)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ColumnSignaturePlan":
        header_size = struct.calcsize(">6sIIIII")
        if len(data) < header_size:
            raise ColumnSignatureError("truncated column-plan header")
        (
            magic,
            input_count,
            class_count,
            accumulator_bits,
            group_count,
            flags,
        ) = struct.unpack(">6sIIIII", data[:header_size])
        if magic != _MAGIC:
            raise ColumnSignatureError("invalid column-plan magic")
        if flags not in (0, 1):
            raise ColumnSignatureError("unsupported column-plan flags")
        cursor = header_size
        needed = cursor + 8 * class_count
        if len(data) < needed:
            raise ColumnSignatureError("truncated column-plan biases")
        biases = tuple(
            struct.unpack(">q", data[cursor + 8 * i : cursor + 8 * i + 8])[0]
            for i in range(class_count)
        )
        cursor = needed
        groups: list[ColumnGroup] = []
        columns: list[tuple[int, ...] | None] = [None] * input_count
        for _ in range(group_count):
            needed = cursor + 8 * class_count + 8
            if len(data) < needed:
                raise ColumnSignatureError("truncated column group")
            signature = tuple(
                struct.unpack(
                    ">q", data[cursor + 8 * i : cursor + 8 * i + 8]
                )[0]
                for i in range(class_count)
            )
            cursor += 8 * class_count
            positive_count, negative_count = struct.unpack(
                ">II", data[cursor : cursor + 8]
            )
            cursor += 8
            member_count = positive_count + negative_count
            needed = cursor + 4 * member_count
            if len(data) < needed:
                raise ColumnSignatureError("truncated column members")
            members = tuple(
                struct.unpack(
                    ">I", data[cursor + 4 * i : cursor + 4 * i + 4]
                )[0]
                for i in range(member_count)
            )
            cursor = needed
            positive = members[:positive_count]
            negative = members[positive_count:]
            group = ColumnGroup(signature, positive, negative)
            groups.append(group)
            for index in positive:
                if index >= input_count or columns[index] is not None:
                    raise ColumnSignatureError("invalid or duplicate positive member")
                columns[index] = signature
            negated = tuple(-value for value in signature)
            for index in negative:
                if index >= input_count or columns[index] is not None:
                    raise ColumnSignatureError("invalid or duplicate negative member")
                columns[index] = negated
        if cursor != len(data):
            raise ColumnSignatureError("column-plan byte length mismatch")
        if any(column is None for column in columns):
            raise ColumnSignatureError("column plan omits source columns")
        weights = tuple(
            tuple(
                columns[input_index][class_index]
                for input_index in range(input_count)
            )
            for class_index in range(class_count)
        )
        specification = BinaryLinearTop1Spec(
            weights=weights,
            biases=biases,
            accumulator_bits=accumulator_bits,
            family="deserialized_column_signature",
        )
        plan = cls(
            specification=specification,
            groups=tuple(groups),
            sign_canonical=bool(flags & 1),
        )
        plan.validate()
        return plan


def compile_column_signature_plan(
    specification: BinaryLinearTop1Spec,
    *,
    sign_canonical: bool = True,
) -> ColumnSignaturePlan:
    specification.validate()
    grouped: dict[
        tuple[int, ...], tuple[list[int], list[int]]
    ] = {}
    for input_index in range(specification.input_count):
        source = tuple(row[input_index] for row in specification.weights)
        if sign_canonical:
            signature, polarity = _canonical_signature(source)
        else:
            signature, polarity = source, 1
        positive, negative = grouped.setdefault(signature, ([], []))
        if polarity > 0:
            positive.append(input_index)
        else:
            negative.append(input_index)

    groups = tuple(
        ColumnGroup(
            signature=signature,
            positive_indices=tuple(positive),
            negative_indices=tuple(negative),
        )
        for signature, (positive, negative) in sorted(grouped.items())
    )
    plan = ColumnSignaturePlan(
        specification=specification,
        groups=groups,
        sign_canonical=bool(sign_canonical),
    )
    plan.validate()
    return plan
