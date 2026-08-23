"""Exact rectangular bilinear-factorization and resource accounting helpers.

EXP-100A is deliberately an E0/E1 upper-bound Gate, not an executor.  It asks
whether explicit public small-coefficient matrix-multiplication tensors can be
composed for the registered Llama-405B projection shapes while charging every
leaf operation, factor transform, cold byte, repair row, and a favorable
workspace lower bound.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class ExplicitFMMError(ValueError):
    """Raised when a factorization or frozen cost contract is malformed."""


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ExplicitFMMError("invalid ceil_div input")
    return (value + divisor - 1) // divisor


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FactorStats:
    rows: int
    rank: int
    nnz: int
    nonempty_rows: int
    nonempty_columns: int
    row_additions: int
    column_additions: int
    nonunit_nnz: int
    negative_unit_nnz: int
    max_abs_coefficient: int
    max_row_support: int
    max_column_support: int
    max_row_l1: int
    max_column_l1: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExactFactorization:
    key: str
    a: int
    b: int
    c: int
    rank: int
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    u_stats: FactorStats
    v_stats: FactorStats
    w_stats: FactorStats
    tensor_sha256: str

    @property
    def classical_rank(self) -> int:
        return self.a * self.b * self.c

    @property
    def rank_ratio(self) -> float:
        return self.rank / self.classical_rank

    def as_manifest(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "shape": [self.a, self.b, self.c],
            "rank": self.rank,
            "classical_rank": self.classical_rank,
            "rank_ratio": self.rank_ratio,
            "u": self.u_stats.as_dict(),
            "v": self.v_stats.as_dict(),
            "w": self.w_stats.as_dict(),
            "tensor_sha256": self.tensor_sha256,
        }


@dataclass(frozen=True)
class SchemeOrientation:
    scheme_id: str
    source_key: str
    permutation: tuple[int, int, int]
    a: int
    b: int
    c: int
    rank: int
    left_factor_name: str
    weight_factor_name: str
    output_factor_name: str
    left: FactorStats
    weight: FactorStats
    output: FactorStats

    @property
    def classical_rank(self) -> int:
        return self.a * self.b * self.c

    @property
    def rank_ratio(self) -> float:
        return self.rank / self.classical_rank

    @property
    def left_additions(self) -> int:
        return self.left.column_additions

    @property
    def weight_additions(self) -> int:
        return self.weight.column_additions

    @property
    def output_additions(self) -> int:
        return self.output.row_additions

    @property
    def left_scales(self) -> int:
        return self.left.nonunit_nnz

    @property
    def weight_scales(self) -> int:
        return self.weight.nonunit_nnz

    @property
    def output_scales(self) -> int:
        return self.output.nonunit_nnz

    @property
    def max_abs_coefficient(self) -> int:
        return max(
            self.left.max_abs_coefficient,
            self.weight.max_abs_coefficient,
            self.output.max_abs_coefficient,
        )

    @property
    def transform_proxy(self) -> int:
        return (
            self.left_additions
            + self.weight_additions
            + self.output_additions
            + self.left_scales
            + self.weight_scales
            + self.output_scales
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scheme_id": self.scheme_id,
            "source_key": self.source_key,
            "permutation": list(self.permutation),
            "shape": [self.a, self.b, self.c],
            "rank": self.rank,
            "classical_rank": self.classical_rank,
            "rank_ratio": self.rank_ratio,
            "factor_assignment": {
                "left": self.left_factor_name,
                "weight": self.weight_factor_name,
                "output": self.output_factor_name,
            },
            "left_additions": self.left_additions,
            "weight_additions": self.weight_additions,
            "output_additions": self.output_additions,
            "left_scales": self.left_scales,
            "weight_scales": self.weight_scales,
            "output_scales": self.output_scales,
            "max_abs_coefficient": self.max_abs_coefficient,
            "transform_proxy": self.transform_proxy,
        }


@dataclass(frozen=True)
class SequenceState:
    scheme_ids: tuple[str, ...]
    a_product: int
    b_product: int
    c_product: int
    rank_product: int
    transform_proxy: int

    @property
    def depth(self) -> int:
        return len(self.scheme_ids)


@dataclass(frozen=True)
class StructuralPlan:
    sequence_id: str
    scheme_ids: tuple[str, ...]
    depth: int
    cut_depth: int
    static_scalar_bytes: int
    a_product: int
    b_product: int
    c_product: int
    rank_product: int
    padded_m: int
    padded_k: int
    padded_n: int
    leaf_m: int
    leaf_k: int
    leaf_n: int
    baseline_multiplications: int
    baseline_additions: int
    leaf_multiplications: int
    leaf_additions: int
    left_transform_additions: int
    left_transform_scales: int
    left_transform_moves: int
    online_weight_transform_additions: int
    online_weight_transform_scales: int
    online_weight_transform_moves: int
    output_transform_additions: int
    output_transform_scales: int
    output_transform_moves: int
    online_operations_without_repair: int
    free_transform_operations_without_repair: int
    transformed_weight_scalar_ratio: float
    transformed_weight_byte_ratio: float
    local_weight_transform_workspace_bytes: int
    favorable_workspace_bytes: int
    factor_metadata_bytes: int
    padding_volume_ratio: float

    @property
    def baseline_operations(self) -> int:
        return self.baseline_multiplications + self.baseline_additions

    @property
    def direct_operation_ratio_without_repair(self) -> float:
        return self.online_operations_without_repair / self.baseline_operations

    @property
    def free_transform_ratio_without_repair(self) -> float:
        return self.free_transform_operations_without_repair / self.baseline_operations

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["scheme_ids"] = list(self.scheme_ids)
        row["baseline_operations"] = self.baseline_operations
        row["direct_operation_ratio_without_repair"] = (
            self.direct_operation_ratio_without_repair
        )
        row["free_transform_ratio_without_repair"] = (
            self.free_transform_ratio_without_repair
        )
        return row


@dataclass(frozen=True)
class FamilyCandidate:
    family: str
    rows: int
    columns: int
    count: int
    block_length: int
    repair_fraction: float
    plan: StructuralPlan
    arithmetic_operations: float
    arithmetic_ratio_within_family: float
    raw_weight_bytes: int
    primary_streamed_bytes_per_block: float
    repair_streamed_bytes_per_block: float
    streamed_bytes_per_block: float
    streamed_bytes_per_token: float
    compiled_static_bytes: float
    workspace_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "rows": self.rows,
            "columns": self.columns,
            "count": self.count,
            "block_length": self.block_length,
            "repair_fraction": self.repair_fraction,
            "arithmetic_operations": self.arithmetic_operations,
            "arithmetic_ratio_within_family": self.arithmetic_ratio_within_family,
            "raw_weight_bytes": self.raw_weight_bytes,
            "primary_streamed_bytes_per_block": self.primary_streamed_bytes_per_block,
            "repair_streamed_bytes_per_block": self.repair_streamed_bytes_per_block,
            "streamed_bytes_per_block": self.streamed_bytes_per_block,
            "streamed_bytes_per_token": self.streamed_bytes_per_token,
            "compiled_static_bytes": self.compiled_static_bytes,
            "workspace_bytes": self.workspace_bytes,
            "plan": self.plan.as_dict(),
        }


@dataclass(frozen=True)
class JointPlan:
    block_length: int
    plan_kind: str
    family_candidates: tuple[FamilyCandidate, ...]
    total_arithmetic_operations: float
    total_baseline_operations: int
    arithmetic_ratio: float
    total_streamed_bytes_per_token: float
    total_raw_weight_bytes: int
    io_fraction: float
    maximum_workspace_bytes: int
    total_compiled_static_bytes: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_length": self.block_length,
            "plan_kind": self.plan_kind,
            "total_arithmetic_operations": self.total_arithmetic_operations,
            "total_baseline_operations": self.total_baseline_operations,
            "arithmetic_ratio": self.arithmetic_ratio,
            "total_streamed_bytes_per_token": self.total_streamed_bytes_per_token,
            "total_raw_weight_bytes": self.total_raw_weight_bytes,
            "io_fraction": self.io_fraction,
            "maximum_workspace_bytes": self.maximum_workspace_bytes,
            "total_compiled_static_bytes": self.total_compiled_static_bytes,
            "family_candidates": [row.as_dict() for row in self.family_candidates],
        }


def _as_integral_matrix(value: Any) -> np.ndarray | None:
    matrix = np.asarray(value)
    if matrix.ndim != 2:
        return None
    if np.issubdtype(matrix.dtype, np.integer):
        return matrix.astype(np.int64, copy=False)
    if not np.issubdtype(matrix.dtype, np.number):
        return None
    if not np.all(np.isfinite(matrix)):
        return None
    rounded = np.rint(matrix)
    if not np.array_equal(matrix, rounded):
        return None
    return rounded.astype(np.int64)


def factor_stats(matrix: np.ndarray) -> FactorStats:
    matrix = _as_integral_matrix(matrix)
    if matrix is None:
        raise ExplicitFMMError("factor matrix is not a finite integral rank-2 array")
    if min(matrix.shape) <= 0:
        raise ExplicitFMMError("empty factor matrix")
    nonzero = matrix != 0
    row_support = np.count_nonzero(nonzero, axis=1)
    column_support = np.count_nonzero(nonzero, axis=0)
    abs_matrix = np.abs(matrix)
    nnz = int(np.count_nonzero(nonzero))
    nonempty_rows = int(np.count_nonzero(row_support))
    nonempty_columns = int(np.count_nonzero(column_support))
    return FactorStats(
        rows=int(matrix.shape[0]),
        rank=int(matrix.shape[1]),
        nnz=nnz,
        nonempty_rows=nonempty_rows,
        nonempty_columns=nonempty_columns,
        row_additions=int(np.maximum(row_support - 1, 0).sum()),
        column_additions=int(np.maximum(column_support - 1, 0).sum()),
        nonunit_nnz=int(np.count_nonzero(nonzero & (abs_matrix != 1))),
        negative_unit_nnz=int(np.count_nonzero(matrix == -1)),
        max_abs_coefficient=int(abs_matrix.max(initial=0)),
        max_row_support=int(row_support.max(initial=0)),
        max_column_support=int(column_support.max(initial=0)),
        max_row_l1=int(abs_matrix.sum(axis=1).max(initial=0)),
        max_column_l1=int(abs_matrix.sum(axis=0).max(initial=0)),
    )


def matrix_multiplication_tensor(a: int, b: int, c: int) -> np.ndarray:
    if min(a, b, c) <= 0:
        raise ExplicitFMMError("tensor dimensions must be positive")
    result = np.zeros((a * b, b * c, c * a), dtype=np.int64)
    for i in range(a):
        for j in range(b):
            for k in range(c):
                result[i * b + j, j * c + k, k * a + i] = 1
    return result


def reconstruct_tensor(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    u_i = _as_integral_matrix(u)
    v_i = _as_integral_matrix(v)
    w_i = _as_integral_matrix(w)
    if u_i is None or v_i is None or w_i is None:
        raise ExplicitFMMError("only integral factorizations are supported")
    rank = u_i.shape[1]
    if v_i.shape[1] != rank or w_i.shape[1] != rank:
        raise ExplicitFMMError("factor ranks differ")
    return np.einsum("ir,jr,kr->ijk", u_i, v_i, w_i, optimize=True)


def exact_factorization(
    key: str,
    a: int,
    b: int,
    c: int,
    u: Any,
    v: Any,
    w: Any,
) -> ExactFactorization:
    u_i = _as_integral_matrix(u)
    v_i = _as_integral_matrix(v)
    w_i = _as_integral_matrix(w)
    if u_i is None or v_i is None or w_i is None:
        raise ExplicitFMMError("catalog factorization is not integral")
    if u_i.shape[0] != a * b or v_i.shape[0] != b * c or w_i.shape[0] != c * a:
        raise ExplicitFMMError(
            f"factor shape mismatch for {key}: {u_i.shape}, {v_i.shape}, {w_i.shape}"
        )
    rank = int(u_i.shape[1])
    if v_i.shape[1] != rank or w_i.shape[1] != rank:
        raise ExplicitFMMError("factor ranks differ")
    expected = matrix_multiplication_tensor(a, b, c)
    actual = reconstruct_tensor(u_i, v_i, w_i)
    if not np.array_equal(actual, expected):
        differences = int(np.count_nonzero(actual != expected))
        raise ExplicitFMMError(
            f"factorization {key} does not reconstruct the integer tensor: {differences} differences"
        )
    tensor_hash = hashlib.sha256(actual.tobytes(order="C")).hexdigest()
    return ExactFactorization(
        key=key,
        a=a,
        b=b,
        c=c,
        rank=rank,
        u=u_i,
        v=v_i,
        w=w_i,
        u_stats=factor_stats(u_i),
        v_stats=factor_stats(v_i),
        w_stats=factor_stats(w_i),
        tensor_sha256=tensor_hash,
    )


def execute_factorization(
    factorization: ExactFactorization, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left_i = np.asarray(left, dtype=np.int64)
    right_i = np.asarray(right, dtype=np.int64)
    if left_i.shape != (factorization.a, factorization.b):
        raise ExplicitFMMError("left matrix shape mismatch")
    if right_i.shape != (factorization.b, factorization.c):
        raise ExplicitFMMError("right matrix shape mismatch")
    left_forms = left_i.reshape(-1) @ factorization.u
    right_forms = right_i.reshape(-1) @ factorization.v
    products = left_forms * right_forms
    output_transposed = factorization.w @ products
    return output_transposed.reshape(factorization.c, factorization.a).T


def _factor_edge_map(
    factorization: ExactFactorization,
) -> dict[frozenset[int], tuple[str, FactorStats]]:
    return {
        frozenset((0, 1)): ("u", factorization.u_stats),
        frozenset((1, 2)): ("v", factorization.v_stats),
        frozenset((0, 2)): ("w", factorization.w_stats),
    }


def orientations(factorization: ExactFactorization) -> tuple[SchemeOrientation, ...]:
    """Return all cyclic/reflection cost orientations of the tensor triangle.

    The runner uses only exact rank and factor support counts.  A future kernel
    must explicitly lower the associated transposes for reflected orientations.
    """

    dimensions = (factorization.a, factorization.b, factorization.c)
    edges = _factor_edge_map(factorization)
    rows: list[SchemeOrientation] = []
    seen: set[tuple[Any, ...]] = set()
    for permutation in itertools.permutations((0, 1, 2)):
        p0, p1, p2 = permutation
        left_name, left = edges[frozenset((p0, p1))]
        weight_name, weight = edges[frozenset((p1, p2))]
        output_name, output = edges[frozenset((p2, p0))]
        a, b, c = dimensions[p0], dimensions[p1], dimensions[p2]
        identity = (
            a,
            b,
            c,
            factorization.rank,
            left_name,
            weight_name,
            output_name,
        )
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(
            SchemeOrientation(
                scheme_id=f"{factorization.key}:p{p0}{p1}{p2}",
                source_key=factorization.key,
                permutation=(p0, p1, p2),
                a=a,
                b=b,
                c=c,
                rank=factorization.rank,
                left_factor_name=left_name,
                weight_factor_name=weight_name,
                output_factor_name=output_name,
                left=left,
                weight=weight,
                output=output,
            )
        )
    return tuple(rows)


def pareto_filter_orientations(
    schemes: Sequence[SchemeOrientation], *, maximum_count: int
) -> tuple[SchemeOrientation, ...]:
    if maximum_count <= 0:
        raise ExplicitFMMError("maximum_count must be positive")
    grouped: dict[tuple[int, int, int], list[SchemeOrientation]] = {}
    for row in schemes:
        grouped.setdefault((row.a, row.b, row.c), []).append(row)

    survivors: list[SchemeOrientation] = []
    for group in grouped.values():
        for candidate in group:
            candidate_cost = (
                candidate.rank,
                candidate.left_additions,
                candidate.weight_additions,
                candidate.output_additions,
                candidate.left_scales,
                candidate.weight_scales,
                candidate.output_scales,
                candidate.max_abs_coefficient,
            )
            dominated = False
            for other in group:
                if other is candidate:
                    continue
                other_cost = (
                    other.rank,
                    other.left_additions,
                    other.weight_additions,
                    other.output_additions,
                    other.left_scales,
                    other.weight_scales,
                    other.output_scales,
                    other.max_abs_coefficient,
                )
                if all(a <= b for a, b in zip(other_cost, candidate_cost)) and any(
                    a < b for a, b in zip(other_cost, candidate_cost)
                ):
                    dominated = True
                    break
            if not dominated:
                survivors.append(candidate)

    survivors.sort(
        key=lambda row: (
            row.rank_ratio,
            row.transform_proxy / max(1, row.classical_rank),
            row.max_abs_coefficient,
            row.a,
            row.b,
            row.c,
            row.scheme_id,
        )
    )
    return tuple(survivors[:maximum_count])


def sequence_id(scheme_ids: Sequence[str]) -> str:
    return canonical_sha256({"schemes": list(scheme_ids)})[:24]


def _prefix_products(
    steps: Sequence[SchemeOrientation],
) -> tuple[list[int], list[int], list[int], list[int]]:
    a = [1]
    b = [1]
    c = [1]
    r = [1]
    for step in steps:
        a.append(a[-1] * step.a)
        b.append(b[-1] * step.b)
        c.append(c[-1] * step.c)
        r.append(r[-1] * step.rank)
    return a, b, c, r


def _factor_metadata_bytes(steps: Sequence[SchemeOrientation]) -> int:
    # Favorable compact sparse encoding: uint16 index plus int8 coefficient.
    total = 0
    for step in steps:
        total += 3 * 32
        total += 3 * (step.left.nnz + step.weight.nnz + step.output.nnz)
    return total


def evaluate_sequence(
    *,
    m: int,
    k: int,
    n: int,
    steps: Sequence[SchemeOrientation],
    cut_depth: int,
    static_scalar_bytes: int,
    activation_bytes: int,
    output_bytes: int,
    accumulator_bytes: int,
) -> StructuralPlan:
    if min(m, k, n) <= 0 or not steps:
        raise ExplicitFMMError("invalid sequence target")
    depth = len(steps)
    if not 0 <= cut_depth <= depth:
        raise ExplicitFMMError("cut depth outside sequence")
    if static_scalar_bytes <= 0 or min(
        activation_bytes, output_bytes, accumulator_bytes
    ) <= 0:
        raise ExplicitFMMError("invalid finite-word byte width")

    prefix_a, prefix_b, prefix_c, prefix_r = _prefix_products(steps)
    a_product = prefix_a[-1]
    b_product = prefix_b[-1]
    c_product = prefix_c[-1]
    rank_product = prefix_r[-1]
    padded_m = ceil_div(m, a_product) * a_product
    padded_k = ceil_div(k, b_product) * b_product
    padded_n = ceil_div(n, c_product) * c_product
    leaf_m = padded_m // a_product
    leaf_k = padded_k // b_product
    leaf_n = padded_n // c_product

    baseline_multiplications = m * k * n
    baseline_additions = m * n * max(0, k - 1)
    leaf_multiplications = rank_product * leaf_m * leaf_k * leaf_n
    leaf_additions = rank_product * leaf_m * leaf_n * max(0, leaf_k - 1)

    left_adds = 0
    left_scales = 0
    left_moves = 0
    weight_adds = 0
    weight_scales = 0
    weight_moves = 0
    output_adds = 0
    output_scales = 0
    output_moves = 0

    for level, step in enumerate(steps):
        parent_products = prefix_r[level]
        m_after = padded_m // prefix_a[level + 1]
        k_after = padded_k // prefix_b[level + 1]
        n_after = padded_n // prefix_c[level + 1]
        left_elements = parent_products * m_after * k_after
        weight_elements = parent_products * k_after * n_after
        output_elements = parent_products * m_after * n_after

        left_adds += step.left_additions * left_elements
        left_scales += step.left_scales * left_elements
        left_moves += step.rank * left_elements

        if level >= cut_depth:
            weight_adds += step.weight_additions * weight_elements
            weight_scales += step.weight_scales * weight_elements
            weight_moves += step.rank * weight_elements

        output_adds += step.output_additions * output_elements
        output_scales += step.output_scales * output_elements
        output_moves += step.a * step.c * output_elements

    online_without_repair = (
        leaf_multiplications
        + leaf_additions
        + left_adds
        + left_scales
        + left_moves
        + weight_adds
        + weight_scales
        + weight_moves
        + output_adds
        + output_scales
        + output_moves
    )
    free_without_repair = leaf_multiplications + leaf_additions

    if cut_depth == 0:
        transformed_scalar_ratio = 1.0
        transformed_byte_ratio = 1.0
        stream_scalar_bytes = 2
    else:
        weight_padding_ratio = (padded_k * padded_n) / (k * n)
        transformed_scalar_ratio = (
            weight_padding_ratio
            * prefix_r[cut_depth]
            / (prefix_b[cut_depth] * prefix_c[cut_depth])
        )
        transformed_byte_ratio = transformed_scalar_ratio * static_scalar_bytes / 2.0
        stream_scalar_bytes = static_scalar_bytes

    remaining_b = b_product // prefix_b[cut_depth]
    remaining_c = c_product // prefix_c[cut_depth]
    remaining_r = rank_product // prefix_r[cut_depth]
    weight_leaf_elements = leaf_k * leaf_n
    store_remaining_inputs = (
        remaining_b * remaining_c * weight_leaf_elements * stream_scalar_bytes
    )
    store_remaining_outputs = (
        remaining_r * weight_leaf_elements * accumulator_bytes
    )
    local_weight_workspace = min(store_remaining_inputs, store_remaining_outputs)

    activation_buffer = m * k * activation_bytes
    output_buffer = m * n * output_bytes
    one_left_form = leaf_m * leaf_k * accumulator_bytes
    output_product_workspace = rank_product * leaf_m * leaf_n * accumulator_bytes
    output_block_workspace = (
        a_product * c_product * leaf_m * leaf_n * accumulator_bytes
    )
    local_output_workspace = min(output_product_workspace, output_block_workspace)
    metadata = _factor_metadata_bytes(steps)
    workspace = (
        activation_buffer
        + output_buffer
        + one_left_form
        + local_weight_workspace
        + local_output_workspace
        + metadata
    )

    return StructuralPlan(
        sequence_id=sequence_id([step.scheme_id for step in steps]),
        scheme_ids=tuple(step.scheme_id for step in steps),
        depth=depth,
        cut_depth=cut_depth,
        static_scalar_bytes=static_scalar_bytes,
        a_product=a_product,
        b_product=b_product,
        c_product=c_product,
        rank_product=rank_product,
        padded_m=padded_m,
        padded_k=padded_k,
        padded_n=padded_n,
        leaf_m=leaf_m,
        leaf_k=leaf_k,
        leaf_n=leaf_n,
        baseline_multiplications=baseline_multiplications,
        baseline_additions=baseline_additions,
        leaf_multiplications=leaf_multiplications,
        leaf_additions=leaf_additions,
        left_transform_additions=left_adds,
        left_transform_scales=left_scales,
        left_transform_moves=left_moves,
        online_weight_transform_additions=weight_adds,
        online_weight_transform_scales=weight_scales,
        online_weight_transform_moves=weight_moves,
        output_transform_additions=output_adds,
        output_transform_scales=output_scales,
        output_transform_moves=output_moves,
        online_operations_without_repair=online_without_repair,
        free_transform_operations_without_repair=free_without_repair,
        transformed_weight_scalar_ratio=transformed_scalar_ratio,
        transformed_weight_byte_ratio=transformed_byte_ratio,
        local_weight_transform_workspace_bytes=local_weight_workspace,
        favorable_workspace_bytes=workspace,
        factor_metadata_bytes=metadata,
        padding_volume_ratio=(padded_m * padded_k * padded_n) / (m * k * n),
    )


def instantiate_family_candidate(
    *,
    family: str,
    rows: int,
    columns: int,
    count: int,
    block_length: int,
    repair_fraction: float,
    plan: StructuralPlan,
    lossless_ratio: float,
    free_transforms: bool = False,
) -> FamilyCandidate:
    if min(rows, columns, count, block_length) <= 0:
        raise ExplicitFMMError("invalid family")
    if not 0.0 <= repair_fraction <= 1.0 or lossless_ratio <= 0:
        raise ExplicitFMMError("invalid repair/compression ratio")
    if plan.padded_m < block_length or plan.baseline_operations <= 0:
        raise ExplicitFMMError("plan target mismatch")
    base = plan.baseline_operations
    core = (
        plan.free_transform_operations_without_repair
        if free_transforms
        else plan.online_operations_without_repair
    )
    one_operations = core + repair_fraction * base
    operations = one_operations * count
    raw_weight_bytes = 2 * rows * columns * count
    byte_ratio = 1.0 if free_transforms else plan.transformed_weight_byte_ratio
    primary_streamed = raw_weight_bytes * byte_ratio / lossless_ratio
    # With the unchanged checkpoint stream (cut 0 / rank oracle), a native row
    # can be repaired while its source weights are already in flight.  A static
    # transformed checkpoint no longer contains those original rows, so the
    # perfect selector must also pay a compressed original-row side stream.
    repair_streamed = (
        0.0
        if free_transforms or plan.cut_depth == 0
        else raw_weight_bytes * repair_fraction / lossless_ratio
    )
    streamed_per_block = primary_streamed + repair_streamed
    workspace_bytes = plan.favorable_workspace_bytes
    if free_transforms:
        workspace_bytes = (
            block_length * columns * 2
            + block_length * rows * 2
            + plan.factor_metadata_bytes
        )
    return FamilyCandidate(
        family=family,
        rows=rows,
        columns=columns,
        count=count,
        block_length=block_length,
        repair_fraction=repair_fraction,
        plan=plan,
        arithmetic_operations=operations,
        arithmetic_ratio_within_family=one_operations / base,
        raw_weight_bytes=raw_weight_bytes,
        primary_streamed_bytes_per_block=primary_streamed,
        repair_streamed_bytes_per_block=repair_streamed,
        streamed_bytes_per_block=streamed_per_block,
        streamed_bytes_per_token=streamed_per_block / block_length,
        compiled_static_bytes=raw_weight_bytes * byte_ratio,
        workspace_bytes=workspace_bytes,
    )


def prune_family_candidates(
    candidates: Iterable[FamilyCandidate],
    *,
    maximum_workspace_bytes: int,
    maximum_candidates: int,
) -> tuple[FamilyCandidate, ...]:
    rows = [row for row in candidates if row.workspace_bytes <= maximum_workspace_bytes]
    rows.sort(
        key=lambda row: (
            row.streamed_bytes_per_token,
            row.arithmetic_operations,
            row.workspace_bytes,
            row.compiled_static_bytes,
            row.plan.sequence_id,
            row.plan.cut_depth,
            row.plan.static_scalar_bytes,
        )
    )
    frontier: list[FamilyCandidate] = []
    best_operations = math.inf
    for row in rows:
        if row.arithmetic_operations < best_operations:
            frontier.append(row)
            best_operations = row.arithmetic_operations
    if len(frontier) <= maximum_candidates:
        return tuple(frontier)
    if maximum_candidates == 1:
        return (frontier[0],)
    indices = {
        round(i * (len(frontier) - 1) / (maximum_candidates - 1))
        for i in range(maximum_candidates)
    }
    return tuple(frontier[index] for index in sorted(indices))


def select_joint_plan(
    *,
    block_length: int,
    plan_kind: str,
    candidates_by_family: Mapping[str, Sequence[FamilyCandidate]],
    io_fraction_limit: float,
    io_bins: int,
) -> JointPlan | None:
    if io_fraction_limit <= 0 or io_bins <= 0 or not candidates_by_family:
        raise ExplicitFMMError("invalid joint-plan contract")
    families = tuple(sorted(candidates_by_family))
    if any(not candidates_by_family[name] for name in families):
        return None
    total_raw = sum(candidates_by_family[name][0].raw_weight_bytes for name in families)
    total_baseline = sum(
        candidates_by_family[name][0].plan.baseline_operations
        * candidates_by_family[name][0].count
        for name in families
    )
    budget_bytes = io_fraction_limit * total_raw
    if budget_bytes <= 0:
        return None

    dp: dict[int, tuple[float, int, float, tuple[FamilyCandidate, ...]]] = {
        0: (0.0, 0, 0.0, ())
    }
    for family in families:
        next_dp: dict[int, tuple[float, int, float, tuple[FamilyCandidate, ...]]] = {}
        for used_bin, state in dp.items():
            operations, max_workspace, static_bytes, chosen = state
            for row in candidates_by_family[family]:
                raw_cost = row.streamed_bytes_per_token / budget_bytes * io_bins
                cost_bin = int(math.ceil(raw_cost - 1e-15))
                new_bin = used_bin + cost_bin
                if new_bin > io_bins:
                    continue
                candidate_state = (
                    operations + row.arithmetic_operations,
                    max(max_workspace, row.workspace_bytes),
                    static_bytes + row.compiled_static_bytes,
                    chosen + (row,),
                )
                incumbent = next_dp.get(new_bin)
                if incumbent is None or (
                    candidate_state[0], candidate_state[1], candidate_state[2]
                ) < (incumbent[0], incumbent[1], incumbent[2]):
                    next_dp[new_bin] = candidate_state
        dp = next_dp
        if not dp:
            return None

    best = min(dp.values(), key=lambda value: (value[0], value[1], value[2]))
    operations, workspace, static_bytes, chosen = best
    actual_io = sum(row.streamed_bytes_per_token for row in chosen)
    if actual_io > budget_bytes * (1.0 + 1e-12):
        raise ExplicitFMMError("quantized DP admitted an over-budget plan")
    return JointPlan(
        block_length=block_length,
        plan_kind=plan_kind,
        family_candidates=chosen,
        total_arithmetic_operations=operations,
        total_baseline_operations=total_baseline,
        arithmetic_ratio=operations / total_baseline,
        total_streamed_bytes_per_token=actual_io,
        total_raw_weight_bytes=total_raw,
        io_fraction=actual_io / total_raw,
        maximum_workspace_bytes=workspace,
        total_compiled_static_bytes=static_bytes,
    )


def state_free_multiplication_ratio(
    state: SequenceState, *, m: int, k: int, n: int
) -> float:
    leaf_m = ceil_div(m, state.a_product)
    leaf_k = ceil_div(k, state.b_product)
    leaf_n = ceil_div(n, state.c_product)
    return state.rank_product * leaf_m * leaf_k * leaf_n / (m * k * n)


def expand_beam(
    *,
    states: Sequence[SequenceState],
    schemes: Sequence[SchemeOrientation],
    m: int,
    k: int,
    n: int,
    maximum_padding_multiplier: float,
    per_split_survivors: int,
    beam_width: int,
) -> tuple[SequenceState, ...]:
    if min(m, k, n, per_split_survivors, beam_width) <= 0:
        raise ExplicitFMMError("invalid beam contract")
    expanded: list[SequenceState] = []
    for state in states:
        for scheme in schemes:
            a_product = state.a_product * scheme.a
            b_product = state.b_product * scheme.b
            c_product = state.c_product * scheme.c
            if (
                a_product > maximum_padding_multiplier * m
                or b_product > maximum_padding_multiplier * k
                or c_product > maximum_padding_multiplier * n
            ):
                continue
            expanded.append(
                SequenceState(
                    scheme_ids=state.scheme_ids + (scheme.scheme_id,),
                    a_product=a_product,
                    b_product=b_product,
                    c_product=c_product,
                    rank_product=state.rank_product * scheme.rank,
                    transform_proxy=state.transform_proxy + scheme.transform_proxy,
                )
            )
    if not expanded:
        return ()

    grouped: dict[tuple[int, int, int], list[SequenceState]] = {}
    for state in expanded:
        grouped.setdefault(
            (state.a_product, state.b_product, state.c_product), []
        ).append(state)
    deduplicated: list[SequenceState] = []
    for group in grouped.values():
        group.sort(
            key=lambda row: (
                state_free_multiplication_ratio(row, m=m, k=k, n=n),
                row.rank_product,
                row.transform_proxy,
                row.scheme_ids,
            )
        )
        deduplicated.extend(group[:per_split_survivors])

    def transform_score(
        row: SequenceState,
    ) -> tuple[float, float, int, tuple[str, ...]]:
        free = state_free_multiplication_ratio(row, m=m, k=k, n=n)
        normalized_proxy = row.transform_proxy / max(1, row.rank_product)
        return free + normalized_proxy / max(1, k), free, row.rank_product, row.scheme_ids

    half = max(1, beam_width // 2)
    free_top = sorted(
        deduplicated,
        key=lambda row: (
            state_free_multiplication_ratio(row, m=m, k=k, n=n),
            row.rank_product,
            row.transform_proxy,
            row.scheme_ids,
        ),
    )[:half]
    transform_top = sorted(deduplicated, key=transform_score)[: beam_width - half]
    selected: dict[tuple[str, ...], SequenceState] = {}
    for row in free_top + transform_top:
        selected[row.scheme_ids] = row
    rows = list(selected.values())
    rows.sort(
        key=lambda row: (
            state_free_multiplication_ratio(row, m=m, k=k, n=n),
            row.transform_proxy,
            row.scheme_ids,
        )
    )
    return tuple(rows[:beam_width])
