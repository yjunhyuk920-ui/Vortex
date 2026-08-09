"""Exact factorized rank and hit certificates for the causal bilinear Gate.

Every query is represented as two float32 factors ``r`` and ``u`` whose outer
product is never materialized.  Odd-prime images can prove rational
independence.  Exact hit credit is deliberately stricter: coefficients are
solved at frozen pivot coordinates with :class:`fractions.Fraction` and the
complete factorized identity is then checked without a tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
from typing import Any, Iterable, Sequence

import numpy as np

from vortex_runtime.temporal_span_replay import (
    TemporalSpanError,
    exact_float32_fraction,
    float32_to_field,
)


PRIMES = (65_521, 65_519, 65_497)
PROMOTE_DECISION = (
    "PROMOTE_ONLY_TO_PAID_PAIR_EXTRACTOR_AND_NATIVE_NUMERICAL_SEMANTICS_GATE"
)
REJECT_DECISION = "REJECT_CAUSAL_BILINEAR_FACTOR_SPAN_LEDGER_AS_CORE"
INVALID_DECISION = "INVALID_CAUSAL_BILINEAR_RANK_GATE_CONTROL_FAILURE"
INFRASTRUCTURE_DECISION = "INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION"


class CausalBilinearGateError(ValueError):
    """Raised when a factor, witness, or registered Gate state is malformed."""


def _factor(value: Any, *, name: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(-1))
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise CausalBilinearGateError(f"{name} must be a nonempty finite float32 vector")
    array = array.copy()
    array[array == 0] = np.float32(0.0)
    return array


def factor_pair_sha256(left: Any, right: Any) -> str:
    left_array = _factor(left, name="left")
    right_array = _factor(right, name="right")
    digest = hashlib.sha256()
    digest.update(np.asarray([left_array.size, right_array.size], dtype="<u8").tobytes())
    digest.update(left_array.astype("<f4", copy=False).tobytes())
    digest.update(right_array.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def rank_one_fingerprints(
    left: Any,
    right: Any,
    *,
    prime: int,
    seeds: Sequence[int],
) -> tuple[int, ...]:
    """Return deterministic independent rank-one field fingerprints."""

    if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
        raise CausalBilinearGateError("fingerprint seeds must be nonempty and unique")
    try:
        left_field = float32_to_field(_factor(left, name="left"), prime)
        right_field = float32_to_field(_factor(right, name="right"), prime)
    except TemporalSpanError as error:
        raise CausalBilinearGateError(str(error)) from error

    def coefficients(width: int, seed: int, stream: int) -> np.ndarray:
        # A fixed 64-bit LCG makes the vectors reproducible without a library RNG.
        state = (int(seed) ^ (0x9E3779B97F4A7C15 * stream)) & ((1 << 64) - 1)
        values = np.empty(width, dtype=np.int64)
        for index in range(width):
            state = (6364136223846793005 * state + 1442695040888963407) & (
                (1 << 64) - 1
            )
            values[index] = state % prime
        return values

    result: list[int] = []
    for seed in seeds:
        left_probe = coefficients(left_field.size, int(seed), 1)
        right_probe = coefficients(right_field.size, int(seed), 2)
        left_value = int(np.mod(left_field @ left_probe, prime))
        right_value = int(np.mod(right_field @ right_probe, prime))
        result.append(left_value * right_value % prime)
    return tuple(result)


def verify_fingerprint_witness(
    basis_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    target_pair: tuple[np.ndarray, np.ndarray],
    coefficients: Sequence[Fraction | int],
    *,
    prime: int,
    seeds: Sequence[int],
) -> bool:
    if len(basis_pairs) != len(coefficients):
        raise CausalBilinearGateError("fingerprint witness coefficient mismatch")
    target = rank_one_fingerprints(*target_pair, prime=prime, seeds=seeds)
    basis = [
        rank_one_fingerprints(*pair, prime=prime, seeds=seeds)
        for pair in basis_pairs
    ]
    mapped: list[int] = []
    for coefficient in coefficients:
        value = Fraction(coefficient)
        denominator = value.denominator % prime
        if denominator == 0:
            raise CausalBilinearGateError(
                "exact coefficient denominator is singular in fingerprint field"
            )
        mapped.append(
            value.numerator % prime * pow(denominator, prime - 2, prime) % prime
        )
    return all(
        target[index]
        == sum(
            coefficient * fingerprints[index]
            for coefficient, fingerprints in zip(mapped, basis, strict=True)
        )
        % prime
        for index in range(len(seeds))
    )


def two_pass_mgs_basis(rows: Any, *, maximum_rank: int) -> np.ndarray:
    """Build the frozen float32, row-order, two-pass MGS side basis."""

    source = np.ascontiguousarray(np.asarray(rows, dtype=np.float32))
    if source.ndim != 2 or not source.shape[0] or not source.shape[1]:
        raise CausalBilinearGateError("MGS source must be a nonempty matrix")
    if not np.all(np.isfinite(source)) or maximum_rank <= 0:
        raise CausalBilinearGateError("invalid MGS source or rank")
    columns: list[np.ndarray] = []
    for row in source:
        candidate = np.ascontiguousarray(row.copy(), dtype=np.float32)
        for _ in range(2):
            for column in columns:
                coefficient = np.float32(np.dot(column, candidate))
                candidate = np.asarray(
                    candidate - np.float32(coefficient) * column,
                    dtype=np.float32,
                )
        squared_norm = np.float32(np.dot(candidate, candidate))
        if not np.isfinite(squared_norm):
            raise CausalBilinearGateError("non-finite MGS residual")
        if squared_norm == np.float32(0.0):
            continue
        norm = np.float32(np.sqrt(squared_norm, dtype=np.float32))
        column = np.asarray(candidate / norm, dtype=np.float32)
        pivot = int(np.flatnonzero(column != 0)[0])
        if np.signbit(column[pivot]):
            column = np.asarray(-column, dtype=np.float32)
        columns.append(np.ascontiguousarray(column))
        if len(columns) == maximum_rank:
            break
    if not columns:
        return np.empty((source.shape[1], 0), dtype=np.float32)
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float32)


def frozen_side_residual(vector: Any, basis: Any) -> np.ndarray:
    """Apply ``x-Q(Q^T x)`` in the frozen float32 multiplication order."""

    value = _factor(vector, name="side vector")
    matrix = np.ascontiguousarray(np.asarray(basis, dtype=np.float32))
    if matrix.ndim != 2 or matrix.shape[0] != value.size:
        raise CausalBilinearGateError("side basis shape mismatch")
    if not np.all(np.isfinite(matrix)):
        raise CausalBilinearGateError("non-finite side basis")
    coordinates = np.asarray(matrix.T @ value, dtype=np.float32)
    projection = np.asarray(matrix @ coordinates, dtype=np.float32)
    residual = np.asarray(value - projection, dtype=np.float32)
    if not np.all(np.isfinite(residual)):
        raise CausalBilinearGateError("non-finite side residual")
    return np.ascontiguousarray(residual)


def _solve_mod(matrix: np.ndarray, target: np.ndarray, prime: int) -> np.ndarray:
    size = int(target.size)
    if matrix.shape != (size, size):
        raise CausalBilinearGateError("modular solve shape mismatch")
    if size == 0:
        return np.empty(0, dtype=np.int64)
    work = np.concatenate(
        (np.mod(matrix, prime), np.mod(target, prime).reshape(size, 1)), axis=1
    ).astype(np.int64, copy=False)
    for column in range(size):
        candidates = np.flatnonzero(work[column:, column] % prime)
        if candidates.size == 0:
            raise CausalBilinearGateError("singular frozen modular pivot matrix")
        pivot = column + int(candidates[0])
        if pivot != column:
            work[[column, pivot]] = work[[pivot, column]]
        inverse = pow(int(work[column, column]), prime - 2, prime)
        work[column] = np.mod(work[column] * inverse, prime)
        for row in range(size):
            if row == column:
                continue
            factor = int(work[row, column])
            if factor:
                work[row] = np.mod(work[row] - factor * work[column], prime)
    return np.ascontiguousarray(work[:, -1], dtype=np.int64)


def _first_nonzero_factorized_mod(
    left: Sequence[np.ndarray],
    weighted_right: Sequence[np.ndarray],
    *,
    prime: int,
) -> tuple[int, int] | None:
    """Find a nonzero coordinate of ``sum left_j right_j^T`` over F_p."""

    left_matrix = np.stack(left, axis=1).astype(np.int64, copy=False)
    right_matrix = np.stack(weighted_right, axis=0).astype(np.int64, copy=False)
    term_count = left_matrix.shape[1]
    row_basis: list[np.ndarray] = []
    pivots: list[int] = []
    for row_index, original in enumerate(left_matrix):
        reduced = np.mod(original.copy(), prime)
        for pivot, basis_row in zip(pivots, row_basis, strict=True):
            factor = int(reduced[pivot])
            if factor:
                reduced = np.mod(reduced - factor * basis_row, prime)
        nonzero = np.flatnonzero(reduced)
        if nonzero.size == 0:
            continue
        image = np.mod(np.mod(original, prime) @ right_matrix, prime)
        image_nonzero = np.flatnonzero(image)
        if image_nonzero.size:
            return row_index, int(image_nonzero[0])
        pivot = int(nonzero[0])
        reduced = np.mod(reduced * pow(int(reduced[pivot]), prime - 2, prime), prime)
        for index, basis_row in enumerate(row_basis):
            factor = int(basis_row[pivot])
            if factor:
                row_basis[index] = np.mod(basis_row - factor * reduced, prime)
        row_basis.append(reduced)
        pivots.append(pivot)
        if len(row_basis) == term_count:
            break
    return None


@dataclass
class FactorizedModularSpan:
    left_width: int
    right_width: int
    prime: int
    left: list[np.ndarray] = field(default_factory=list)
    right: list[np.ndarray] = field(default_factory=list)
    pivots: list[tuple[int, int]] = field(default_factory=list)
    pivot_matrix: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.int64)
    )

    @property
    def rank(self) -> int:
        return len(self.left)

    def _mapped(self, left: Any, right: Any) -> tuple[np.ndarray, np.ndarray]:
        left_array = _factor(left, name="left")
        right_array = _factor(right, name="right")
        if left_array.size != self.left_width or right_array.size != self.right_width:
            raise CausalBilinearGateError("factor width mismatch")
        try:
            return (
                float32_to_field(left_array, self.prime),
                float32_to_field(right_array, self.prime),
            )
        except TemporalSpanError as error:
            raise CausalBilinearGateError(str(error)) from error

    def add(self, left: Any, right: Any) -> tuple[bool, tuple[int, int] | None]:
        left_field, right_field = self._mapped(left, right)
        old_rank = self.rank
        if old_rank:
            target = np.asarray(
                [left_field[a] * right_field[b] % self.prime for a, b in self.pivots],
                dtype=np.int64,
            )
            coefficients = _solve_mod(self.pivot_matrix, target, self.prime)
        else:
            coefficients = np.empty(0, dtype=np.int64)
        weighted_right = [right_field]
        weighted_right.extend(
            np.mod(-int(coefficient) * prior, self.prime)
            for coefficient, prior in zip(coefficients, self.right, strict=True)
        )
        coordinate = _first_nonzero_factorized_mod(
            [left_field, *self.left], weighted_right, prime=self.prime
        )
        if coordinate is None:
            return False, None
        a, b = coordinate
        new_column = np.asarray(
            [left_field[row] * right_field[column] % self.prime for row, column in self.pivots],
            dtype=np.int64,
        )
        new_row = np.asarray(
            [prior_left[a] * prior_right[b] % self.prime for prior_left, prior_right in zip(self.left, self.right, strict=True)],
            dtype=np.int64,
        )
        value = int(left_field[a] * right_field[b] % self.prime)
        expanded = np.empty((old_rank + 1, old_rank + 1), dtype=np.int64)
        if old_rank:
            expanded[:old_rank, :old_rank] = self.pivot_matrix
            expanded[:old_rank, old_rank] = new_column
            expanded[old_rank, :old_rank] = new_row
        expanded[old_rank, old_rank] = value
        self.left.append(left_field)
        self.right.append(right_field)
        self.pivots.append(coordinate)
        self.pivot_matrix = np.mod(expanded, self.prime)
        # A solve is a deterministic invertibility check on the new minor.
        _solve_mod(self.pivot_matrix, np.zeros(old_rank + 1, dtype=np.int64), self.prime)
        return True, coordinate

    def validate(self) -> None:
        if self.pivot_matrix.shape != (self.rank, self.rank):
            raise CausalBilinearGateError("modular pivot matrix shape mismatch")
        if len(self.right) != self.rank or len(self.pivots) != self.rank:
            raise CausalBilinearGateError("modular factor count mismatch")
        _solve_mod(self.pivot_matrix, np.zeros(self.rank, dtype=np.int64), self.prime)


def _fraction_product(left: np.ndarray, right: np.ndarray, coordinate: tuple[int, int]) -> Fraction:
    a, b = coordinate
    return exact_float32_fraction(left[a]) * exact_float32_fraction(right[b])


def _solve_fractions(matrix: Sequence[Sequence[Fraction]], target: Sequence[Fraction]) -> tuple[Fraction, ...]:
    size = len(target)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise CausalBilinearGateError("exact solve shape mismatch")
    if size == 0:
        return ()
    work = [[Fraction(value) for value in row] + [Fraction(target[index])] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = next((row for row in range(column, size) if work[row][column]), None)
        if pivot is None:
            raise CausalBilinearGateError("singular exact pivot matrix")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(work[row], work[column], strict=True)
                ]
    return tuple(row[-1] for row in work)


def exact_coefficients_at_pivots(
    basis_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    pivots: Sequence[tuple[int, int]],
    target_pair: tuple[np.ndarray, np.ndarray],
) -> tuple[Fraction, ...]:
    if len(basis_pairs) != len(pivots):
        raise CausalBilinearGateError("exact pivot/factor count mismatch")
    left = _factor(target_pair[0], name="target left")
    right = _factor(target_pair[1], name="target right")
    matrix = [
        [_fraction_product(pair_left, pair_right, coordinate) for pair_left, pair_right in basis_pairs]
        for coordinate in pivots
    ]
    target = [_fraction_product(left, right, coordinate) for coordinate in pivots]
    return _solve_fractions(matrix, target)


def first_exact_residual_coordinate(
    basis_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    target_pair: tuple[np.ndarray, np.ndarray],
    coefficients: Sequence[Fraction | int],
) -> tuple[int, int] | None:
    """Return the first exact nonzero residual coordinate, or ``None``."""

    if len(basis_pairs) != len(coefficients):
        raise CausalBilinearGateError("exact witness coefficient count mismatch")
    target_left = _factor(target_pair[0], name="target left")
    target_right = _factor(target_pair[1], name="target right")
    normalized_pairs = [
        (_factor(left, name="basis left"), _factor(right, name="basis right"))
        for left, right in basis_pairs
    ]
    if any(left.size != target_left.size or right.size != target_right.size for left, right in normalized_pairs):
        raise CausalBilinearGateError("exact witness factor width mismatch")
    normalized_coefficients = [Fraction(value) for value in coefficients]
    right_terms: list[tuple[Fraction, ...]] = [
        tuple(exact_float32_fraction(value) for value in target_right)
    ]
    for coefficient, (_, right) in zip(normalized_coefficients, normalized_pairs, strict=True):
        right_terms.append(
            tuple(-coefficient * exact_float32_fraction(value) for value in right)
        )
    row_basis: list[list[Fraction]] = []
    pivots: list[int] = []
    left_columns = [target_left, *(left for left, _ in normalized_pairs)]
    for row_index in range(target_left.size):
        original = [exact_float32_fraction(column[row_index]) for column in left_columns]
        reduced = original.copy()
        for pivot, basis_row in zip(pivots, row_basis, strict=True):
            factor = reduced[pivot]
            if factor:
                reduced = [value - factor * source for value, source in zip(reduced, basis_row, strict=True)]
        pivot = next((index for index, value in enumerate(reduced) if value), None)
        if pivot is None:
            continue
        for column in range(target_right.size):
            observed = sum(
                coefficient * term[column]
                for coefficient, term in zip(original, right_terms, strict=True)
            )
            if observed:
                return row_index, column
        scale = reduced[pivot]
        reduced = [value / scale for value in reduced]
        for index, basis_row in enumerate(row_basis):
            factor = basis_row[pivot]
            if factor:
                row_basis[index] = [
                    value - factor * source
                    for value, source in zip(basis_row, reduced, strict=True)
                ]
        row_basis.append(reduced)
        pivots.append(pivot)
        if len(row_basis) == len(left_columns):
            break
    return None


def verify_exact_factorized_identity(
    basis_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    target_pair: tuple[np.ndarray, np.ndarray],
    coefficients: Sequence[Fraction | int],
) -> bool:
    """Verify every outer-product coordinate through exact row-space factors."""

    return first_exact_residual_coordinate(
        basis_pairs, target_pair, coefficients
    ) is None


def exact_hit_witness(
    basis_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    pivots: Sequence[tuple[int, int]],
    target_pair: tuple[np.ndarray, np.ndarray],
) -> tuple[bool, tuple[Fraction, ...]]:
    coefficients = exact_coefficients_at_pivots(basis_pairs, pivots, target_pair)
    return (
        verify_exact_factorized_identity(basis_pairs, target_pair, coefficients),
        coefficients,
    )


def summarize_gate(
    rows: Sequence[dict[str, Any]],
    *,
    control_failures: Sequence[str],
    execution_complete: bool,
    oracle_rank_lower_bound: int,
    expected_rows: int = 36,
    maximum_misses: int = 4,
    minimum_family_hits: int = 5,
) -> dict[str, Any]:
    misses = sum(not bool(row.get("exact_hit")) for row in rows)
    hits = len(rows) - misses
    family_hits: dict[str, int] = {}
    for row in rows:
        if row.get("exact_hit"):
            family = str(row["family"])
            family_hits[family] = family_hits.get(family, 0) + 1
    if control_failures:
        decision = INVALID_DECISION
    elif misses > maximum_misses or oracle_rank_lower_bound >= 28:
        decision = REJECT_DECISION
    elif execution_complete:
        family_pass = len(family_hits) == 6 and all(
            count >= minimum_family_hits for count in family_hits.values()
        )
        decision = PROMOTE_DECISION if (
            len(rows) == expected_rows
            and misses <= maximum_misses
            and oracle_rank_lower_bound <= 27
            and family_pass
        ) else REJECT_DECISION
    else:
        decision = INFRASTRUCTURE_DECISION
    return {
        "decision": decision,
        "evaluated_rows": len(rows),
        "expected_rows": expected_rows,
        "execution_complete": execution_complete,
        "exact_hits": hits,
        "exact_misses": misses,
        "maximum_misses": maximum_misses,
        "family_hits": family_hits,
        "minimum_family_hits": minimum_family_hits,
        "oracle_rank_lower_bound": int(oracle_rank_lower_bound),
        "control_failures": list(control_failures),
    }


def witness_strings(values: Iterable[Fraction]) -> list[str]:
    return [f"{value.numerator}/{value.denominator}" for value in values]
