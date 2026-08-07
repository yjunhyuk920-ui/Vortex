"""Information-capacity bounds for self-contained exact linear-map artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence


GIB = 1024 ** 3


class InformationCapacityError(ValueError):
    """Raised when a capacity audit receives an invalid domain."""


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InformationCapacityError(f"{name} must be a positive integer")
    return value


def ceil_log2(value: int) -> int:
    """Return ceil(log2(value)) exactly for a positive integer."""
    _positive_integer(value, "value")
    return (value - 1).bit_length()


def matrix_population_size(rows: int, columns: int, alphabet_size: int) -> int:
    """Count coefficient matrices in the registered finite domain."""
    rows = _positive_integer(rows, "rows")
    columns = _positive_integer(columns, "columns")
    alphabet_size = _positive_integer(alphabet_size, "alphabet_size")
    return alphabet_size ** (rows * columns)


def variable_length_binary_capacity(maximum_bits: int) -> int:
    """Count all binary strings whose length is at most ``maximum_bits``."""
    if isinstance(maximum_bits, bool) or not isinstance(maximum_bits, int) or maximum_bits < 0:
        raise InformationCapacityError("maximum_bits must be a non-negative integer")
    return (1 << (maximum_bits + 1)) - 1


def minimum_universal_maximum_bits(population_size: int) -> int:
    """Smallest B for which all members can receive distinct <=B-bit strings."""
    population_size = _positive_integer(population_size, "population_size")
    return ceil_log2(population_size + 1) - 1


def power_of_two_alphabet_information_bits(coefficient_count: int, alphabet_size: int) -> int:
    """Exact log2 population size when the coefficient alphabet is a power of two."""
    coefficient_count = _positive_integer(coefficient_count, "coefficient_count")
    alphabet_size = _positive_integer(alphabet_size, "alphabet_size")
    if alphabet_size & (alphabet_size - 1):
        raise InformationCapacityError("alphabet_size must be a power of two")
    return coefficient_count * (alphabet_size.bit_length() - 1)


def exact_matvec(matrix: Sequence[Sequence[int]], vector: Sequence[int]) -> tuple[int, ...]:
    """Evaluate a small integer matrix-vector product without floating arithmetic."""
    normalized = normalize_matrix(matrix)
    if len(vector) != len(normalized[0]):
        raise InformationCapacityError("vector length must equal matrix column count")
    return tuple(
        sum(coefficient * int(value) for coefficient, value in zip(row, vector))
        for row in normalized
    )


def normalize_matrix(matrix: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    if not matrix:
        raise InformationCapacityError("matrix must have at least one row")
    normalized = tuple(tuple(int(value) for value in row) for row in matrix)
    columns = len(normalized[0])
    if columns <= 0 or any(len(row) != columns for row in normalized):
        raise InformationCapacityError("matrix must be non-empty and rectangular")
    return normalized


def standard_basis_signature(matrix: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Return exact outputs on basis inputs; this signature uniquely recovers a matrix."""
    normalized = normalize_matrix(matrix)
    columns = len(normalized[0])
    signature = []
    for column in range(columns):
        basis = tuple(int(index == column) for index in range(columns))
        signature.append(exact_matvec(normalized, basis))
    return tuple(signature)


def enumerate_matrices(rows: int, columns: int, alphabet: Iterable[int]) -> Iterable[tuple[tuple[int, ...], ...]]:
    rows = _positive_integer(rows, "rows")
    columns = _positive_integer(columns, "columns")
    symbols = tuple(int(value) for value in alphabet)
    if not symbols or len(set(symbols)) != len(symbols):
        raise InformationCapacityError("alphabet must contain unique values")
    for coefficients in product(symbols, repeat=rows * columns):
        yield tuple(
            tuple(coefficients[row * columns:(row + 1) * columns])
            for row in range(rows)
        )


@dataclass(frozen=True)
class FiniteDomainAudit:
    rows: int
    columns: int
    alphabet_size: int
    matrix_count: int
    unique_basis_signature_count: int
    signature_collision_count: int
    required_universal_maximum_bits: int
    capacity_below_required_bits: int
    capacity_at_required_bits: int

    @property
    def injective(self) -> bool:
        return self.matrix_count == self.unique_basis_signature_count and self.signature_collision_count == 0

    @property
    def boundary_valid(self) -> bool:
        return (
            self.capacity_below_required_bits < self.matrix_count
            and self.capacity_at_required_bits >= self.matrix_count
        )

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "alphabet_size": self.alphabet_size,
            "matrix_count": self.matrix_count,
            "unique_basis_signature_count": self.unique_basis_signature_count,
            "signature_collision_count": self.signature_collision_count,
            "required_universal_maximum_bits": self.required_universal_maximum_bits,
            "capacity_below_required_bits": self.capacity_below_required_bits,
            "capacity_at_required_bits": self.capacity_at_required_bits,
            "injective": self.injective,
            "boundary_valid": self.boundary_valid,
        }


def audit_finite_domain(rows: int, columns: int, alphabet_size: int) -> FiniteDomainAudit:
    """Exhaustively validate basis-signature injectivity and the bit-cap boundary."""
    population = matrix_population_size(rows, columns, alphabet_size)
    required = minimum_universal_maximum_bits(population)
    signatures: set[tuple[tuple[int, ...], ...]] = set()
    collisions = 0
    for matrix in enumerate_matrices(rows, columns, range(alphabet_size)):
        signature = standard_basis_signature(matrix)
        collisions += int(signature in signatures)
        signatures.add(signature)
    below = variable_length_binary_capacity(required - 1) if required > 0 else 0
    return FiniteDomainAudit(
        rows=rows,
        columns=columns,
        alphabet_size=alphabet_size,
        matrix_count=population,
        unique_basis_signature_count=len(signatures),
        signature_collision_count=collisions,
        required_universal_maximum_bits=required,
        capacity_below_required_bits=below,
        capacity_at_required_bits=variable_length_binary_capacity(required),
    )


@dataclass(frozen=True)
class TargetCapacityAudit:
    coefficient_count: int
    alphabet_size: int
    q4_information_bits: int
    q4_information_gib: float
    hot_allowance_bits: int
    hot_allowance_fraction: float
    required_over_hot_factor: float
    former_static_gate_bits: int
    former_static_gate_gib: float
    former_static_gate_fits_hot: bool
    universal_self_contained_hot_gate_pass: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def audit_target_capacity(
    *,
    coefficient_count: int,
    alphabet_size: int,
    hot_allowance_bytes: int,
    former_static_gate_fraction: float,
) -> TargetCapacityAudit:
    information_bits = power_of_two_alphabet_information_bits(coefficient_count, alphabet_size)
    hot_allowance_bytes = _positive_integer(hot_allowance_bytes, "hot_allowance_bytes")
    if not 0.0 < former_static_gate_fraction <= 1.0:
        raise InformationCapacityError("former_static_gate_fraction must be in (0, 1]")
    hot_bits = hot_allowance_bytes * 8
    former_bits = int(information_bits * former_static_gate_fraction)
    return TargetCapacityAudit(
        coefficient_count=coefficient_count,
        alphabet_size=alphabet_size,
        q4_information_bits=information_bits,
        q4_information_gib=information_bits / 8 / GIB,
        hot_allowance_bits=hot_bits,
        hot_allowance_fraction=hot_bits / information_bits,
        required_over_hot_factor=information_bits / hot_bits,
        former_static_gate_bits=former_bits,
        former_static_gate_gib=former_bits / 8 / GIB,
        former_static_gate_fits_hot=former_bits <= hot_bits,
        universal_self_contained_hot_gate_pass=information_bits <= hot_bits,
    )
