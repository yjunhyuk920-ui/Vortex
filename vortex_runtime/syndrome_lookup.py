"""Pure reference logic for the throwaway EXP-081A prototype.

Question: can a cheap candidate whose error belongs to a tiny output code be
recovered from a syndrome and independently fingerprinted before exact commit?
This module answers only the algebraic and resource-accounting part.  It is not
a production matrix kernel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Callable, Sequence


Matrix = tuple[tuple[int, ...], ...]
Vector = tuple[int, ...]


def _matrix(value: Sequence[Sequence[int]], prime: int) -> Matrix:
    rows = tuple(tuple(int(x) % prime for x in row) for row in value)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("matrix must be nonempty and rectangular")
    return rows


def transpose(value: Matrix) -> Matrix:
    return tuple(tuple(value[i][j] for i in range(len(value))) for j in range(len(value[0])))


def matvec(value: Matrix, vector: Sequence[int], prime: int) -> Vector:
    if len(value[0]) != len(vector):
        raise ValueError("matrix/vector shape mismatch")
    return tuple(sum(a * int(b) for a, b in zip(row, vector)) % prime for row in value)


def matmul(left: Matrix, right: Matrix, prime: int) -> Matrix:
    if len(left[0]) != len(right):
        raise ValueError("matrix shape mismatch")
    columns = transpose(right)
    return tuple(
        tuple(sum(a * b for a, b in zip(row, column)) % prime for column in columns)
        for row in left
    )


def add(left: Sequence[int], right: Sequence[int], prime: int) -> Vector:
    if len(left) != len(right):
        raise ValueError("vector shape mismatch")
    return tuple((int(a) + int(b)) % prime for a, b in zip(left, right))


def subtract(left: Sequence[int], right: Sequence[int], prime: int) -> Vector:
    if len(left) != len(right):
        raise ValueError("vector shape mismatch")
    return tuple((int(a) - int(b)) % prime for a, b in zip(left, right))


def invert(value: Matrix, prime: int) -> Matrix:
    if len(value) != len(value[0]):
        raise ValueError("inverse requires a square matrix")
    n = len(value)
    rows = [list(row) + [int(i == j) for j in range(n)] for i, row in enumerate(value)]
    for column in range(n):
        pivot = next((i for i in range(column, n) if rows[i][column] % prime), None)
        if pivot is None:
            raise ValueError("singular recovery sketch")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = pow(rows[column][column] % prime, prime - 2, prime)
        rows[column] = [(x * scale) % prime for x in rows[column]]
        for i in range(n):
            if i == column:
                continue
            factor = rows[i][column] % prime
            if factor:
                rows[i] = [
                    (x - factor * y) % prime for x, y in zip(rows[i], rows[column])
                ]
    return tuple(tuple(row[n:]) for row in rows)


def direct_matvec(weight: Sequence[Sequence[int]], x: Sequence[int], prime: int) -> Vector:
    return matvec(_matrix(weight, prime), x, prime)


@dataclass(frozen=True)
class CompiledSyndromeMatVec:
    prime: int
    weight: Matrix
    dictionary: Matrix
    recovery: Matrix
    verification: Matrix
    recovery_weight: Matrix
    recovery_dictionary_inverse: Matrix
    verification_weight: Matrix

    @property
    def input_width(self) -> int:
        return len(self.weight[0])

    @property
    def output_width(self) -> int:
        return len(self.weight)

    @property
    def rank(self) -> int:
        return len(self.dictionary[0])

    @property
    def verification_rows(self) -> int:
        return len(self.verification)


@dataclass(frozen=True)
class QueryResult:
    candidate: Vector
    syndrome: Vector
    coefficients: Vector
    corrected: Vector
    fingerprint_expected: Vector
    fingerprint_actual: Vector
    verified: bool
    used_fallback: bool
    output: Vector

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compile_syndrome_matvec(
    *,
    weight: Sequence[Sequence[int]],
    dictionary: Sequence[Sequence[int]],
    recovery: Sequence[Sequence[int]],
    verification: Sequence[Sequence[int]],
    prime: int,
) -> CompiledSyndromeMatVec:
    if prime < 3:
        raise ValueError("prime must be at least three")
    w = _matrix(weight, prime)
    d = _matrix(dictionary, prime)
    r = _matrix(recovery, prime)
    v = _matrix(verification, prime)
    if len(d) != len(w) or len(r[0]) != len(w) or len(v[0]) != len(w):
        raise ValueError("output dimensions do not match")
    if len(r) != len(d[0]):
        raise ValueError("recovery row count must equal dictionary rank")
    rd = matmul(r, d, prime)
    return CompiledSyndromeMatVec(
        prime=prime,
        weight=w,
        dictionary=d,
        recovery=r,
        verification=v,
        recovery_weight=matmul(r, w, prime),
        recovery_dictionary_inverse=invert(rd, prime),
        verification_weight=matmul(v, w, prime),
    )


def execute_query(
    compiled: CompiledSyndromeMatVec,
    x: Sequence[int],
    candidate: Sequence[int],
    *,
    fallback: Callable[[Sequence[int]], Sequence[int]] | None = None,
) -> QueryResult:
    p = compiled.prime
    g = tuple(int(value) % p for value in candidate)
    if len(x) != compiled.input_width or len(g) != compiled.output_width:
        raise ValueError("query shape mismatch")
    syndrome = subtract(
        matvec(compiled.recovery_weight, x, p),
        matvec(compiled.recovery, g, p),
        p,
    )
    coefficients = matvec(compiled.recovery_dictionary_inverse, syndrome, p)
    corrected = add(g, matvec(compiled.dictionary, coefficients, p), p)
    expected = matvec(compiled.verification_weight, x, p)
    actual = matvec(compiled.verification, corrected, p)
    verified = expected == actual
    used_fallback = not verified
    if used_fallback:
        if fallback is None:
            output = matvec(compiled.weight, x, p)
        else:
            output = tuple(int(value) % p for value in fallback(x))
    else:
        output = corrected
    return QueryResult(
        candidate=g,
        syndrome=syndrome,
        coefficients=coefficients,
        corrected=corrected,
        fingerprint_expected=expected,
        fingerprint_actual=actual,
        verified=verified,
        used_fallback=used_fallback,
        output=output,
    )


def random_matrix(rows: int, columns: int, prime: int, rng: random.Random) -> Matrix:
    return tuple(tuple(rng.randrange(prime) for _ in range(columns)) for _ in range(rows))


def compile_random(
    *,
    weight: Sequence[Sequence[int]],
    dictionary: Sequence[Sequence[int]],
    verification_rows: int,
    prime: int,
    seed: int,
) -> CompiledSyndromeMatVec:
    d = _matrix(dictionary, prime)
    rng = random.Random(seed)
    for _ in range(256):
        recovery = random_matrix(len(d[0]), len(d), prime, rng)
        verification = random_matrix(verification_rows, len(d), prime, rng)
        try:
            return compile_syndrome_matvec(
                weight=weight,
                dictionary=d,
                recovery=recovery,
                verification=verification,
                prime=prime,
            )
        except ValueError:
            continue
    raise ValueError("could not construct an invertible recovery sketch")


@dataclass(frozen=True)
class ResourceRow:
    rank: int
    verification_rows: int
    stages: int
    leaves: int
    operation_fraction: float
    traffic_fraction: float
    sidecar_bytes: int
    metadata_traffic_bytes: int
    metadata_storage_bytes: int
    required_fast_coverage: float
    operation_pass: bool
    traffic_pass: bool
    storage_pass: bool


def aggregate_resource_row(
    shape_rows: Sequence[dict[str, int | str]],
    *,
    rank: int,
    verification_rows: int,
    stages: int,
    leaves: int,
    field_bytes: int,
    page_bytes: int,
    target_fraction: float,
    storage_limit_bytes: int,
) -> ResourceRow:
    baseline = operations = traffic = storage = 0
    metadata_traffic = metadata_storage = 0
    if leaves <= 0 or leaves & (leaves - 1):
        raise ValueError("leaves must be a positive power of two")
    path_depth = leaves.bit_length() - 1
    for row in shape_rows:
        if row["tensor"] == "embedding":
            continue
        m, n, count = int(row["rows"]), int(row["columns"]), int(row["count"])
        baseline += count * m * n
        recovery_elements = rank * (n + m) + verification_rows * n + rank * rank
        # Per tree: full binary internal nodes store (feature, threshold), leaves
        # store one int32 page scale. Two uint64 seeds generate procedural R,V.
        metadata_storage_per_matrix = stages * ((leaves - 1) * 8 + leaves * 4) + 16
        # A query reads one path per stage, the selected scale, and both seeds.
        metadata_traffic_per_query = stages * (path_depth * 8 + 4) + 16
        operations += count * (
            rank * n
            + 2 * rank * m
            + rank * rank
            + verification_rows * (n + m)
            + stages * m
            + path_depth * stages
        )
        traffic += count * (
            field_bytes * recovery_elements
            + page_bytes * stages * m
            + metadata_traffic_per_query
        )
        storage += count * (
            field_bytes * recovery_elements
            + page_bytes * stages * leaves * m
            + metadata_storage_per_matrix
        )
        metadata_traffic += count * metadata_traffic_per_query
        metadata_storage += count * metadata_storage_per_matrix
    dense_q4_bytes = baseline / 2
    operation_fraction = operations / baseline
    traffic_fraction = traffic / dense_q4_bytes
    remaining = max(0.0, target_fraction - traffic_fraction)
    required_fast_coverage = 1.0 - remaining
    return ResourceRow(
        rank=rank,
        verification_rows=verification_rows,
        stages=stages,
        leaves=leaves,
        operation_fraction=operation_fraction,
        traffic_fraction=traffic_fraction,
        sidecar_bytes=storage,
        metadata_traffic_bytes=metadata_traffic,
        metadata_storage_bytes=metadata_storage,
        required_fast_coverage=required_fast_coverage,
        operation_pass=operation_fraction <= target_fraction,
        traffic_pass=traffic_fraction <= target_fraction,
        storage_pass=storage <= storage_limit_bytes,
    )
