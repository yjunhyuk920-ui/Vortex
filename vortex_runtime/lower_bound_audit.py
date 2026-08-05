"""Applicability audit helpers for exact online dense-runtime lower bounds."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


GIB = 1 << 30


@dataclass(frozen=True)
class TensorSpec:
    name: str
    count: int
    rows: int
    columns: int

    @property
    def parameters(self) -> int:
        return self.count * self.rows * self.columns

    @property
    def square_subproblem_dimension(self) -> int:
        return min(self.rows, self.columns)


@dataclass(frozen=True)
class CKLApplicability:
    dimension: int
    side_bits: int
    branch: str | None
    applicable: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "side_bits": self.side_bits,
            "branch": self.branch,
            "applicable": self.applicable,
            "reason": self.reason,
            "minimum_side_bits_for_tradeoff": self.dimension,
            "maximum_side_bits_for_tradeoff": self.dimension * self.dimension // 4,
        }


def ckl2018_applicability(dimension: int, side_bits: int) -> CKLApplicability:
    if dimension <= 0 or side_bits < 0:
        raise ValueError("dimension must be positive and side_bits non-negative")
    if side_bits < dimension:
        return CKLApplicability(
            dimension,
            side_bits,
            "small_redundancy",
            True,
            "r < n: the theorem states a near-quadratic probe lower bound",
        )
    maximum = dimension * dimension // 4
    if side_bits <= maximum:
        return CKLApplicability(
            dimension,
            side_bits,
            "tradeoff",
            True,
            "n <= r <= n^2/4: the theorem states a probe/redundancy tradeoff",
        )
    return CKLApplicability(
        dimension,
        side_bits,
        None,
        False,
        "r > n^2/4: outside the theorem's registered succinct-redundancy range",
    )


def finite_field_symmetric_representatives(prime: int) -> tuple[int, ...]:
    if prime < 2:
        raise ValueError("prime must be >=2")
    if prime == 2:
        return (0, 1)
    half = prime // 2
    return tuple(range(-half, half + 1))


def q4_field_embedding_supported(prime: int, q4_min: int = -8, q4_max: int = 7) -> bool:
    reps = finite_field_symmetric_representatives(prime)
    return min(reps) >= q4_min and max(reps) <= q4_max


def cgl2015_unit_constant_indicator(
    *,
    dimension: int,
    field_size: int,
    side_bits: int,
    word_bits: int,
) -> dict[str, float]:
    if dimension <= 0 or field_size <= 1 or side_bits < 0 or word_bits <= 0:
        raise ValueError("invalid indicator arguments")
    log_field = math.log2(field_size)
    information_bits = dimension * dimension * log_field
    total_matrix_dependent_bits = information_bits + side_bits
    alpha = total_matrix_dependent_bits / information_bits
    denominator = max(1.0, math.log2(max(alpha, 1.0)))
    first = dimension * log_field / denominator
    second = dimension * dimension * log_field / word_bits
    return {
        "field_log2": log_field,
        "information_bits": information_bits,
        "matrix_dependent_bits": total_matrix_dependent_bits,
        "space_overhead_alpha": alpha,
        "first_term_probes_unit_constant": first,
        "second_term_probes_unit_constant": second,
        "minimum_term_probes_unit_constant": min(first, second),
    }


def packed_q4_cells(rows: int, columns: int, word_bits: int) -> int:
    if rows <= 0 or columns <= 0 or word_bits <= 0:
        raise ValueError("invalid packed-cell arguments")
    return math.ceil(rows * columns * 4 / word_bits)


def exhaustive_binary_projection_reduction(max_dimension: int = 4) -> dict[str, int]:
    if max_dimension < 1 or max_dimension > 4:
        raise ValueError("registered exhaustive dimensions are 1..4")
    cases = 0
    float_cases = 0
    mismatches = 0
    for n in range(1, max_dimension + 1):
        vectors = [
            tuple((code >> j) & 1 for j in range(n))
            for code in range(1 << n)
        ]
        for matrix_code in range(1 << (n * n)):
            row_masks = []
            matrix = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                mask = 0
                for j in range(n):
                    bit = (matrix_code >> (i * n + j)) & 1
                    mask |= bit << j
                    matrix[i, j] = bit
                row_masks.append(mask)
            for vector in vectors:
                vector_mask = sum(bit << j for j, bit in enumerate(vector))
                integer = tuple((mask & vector_mask).bit_count() for mask in row_masks)
                boolean = tuple(int(value > 0) for value in integer)
                f2 = tuple(value & 1 for value in integer)
                expected_boolean = tuple(
                    int(any(matrix[i, j] and vector[j] for j in range(n)))
                    for i in range(n)
                )
                expected_f2 = tuple(
                    sum(int(matrix[i, j]) * vector[j] for j in range(n)) % 2
                    for i in range(n)
                )
                mismatches += int(boolean != expected_boolean)
                mismatches += int(f2 != expected_f2)
                if n <= 3:
                    floating = matrix @ np.asarray(vector, dtype=np.float32)
                    mismatches += int(tuple(int(x) for x in floating) != integer)
                    float_cases += 1
                cases += 1
    return {
        "maximum_dimension": max_dimension,
        "cases": cases,
        "float32_replay_cases": float_cases,
        "mismatches": mismatches,
    }


def llama_405b_tensor_plan() -> tuple[TensorSpec, ...]:
    layers = 126
    hidden = 16_384
    kv = 1_024
    intermediate = 53_248
    vocabulary = 128_256
    return (
        TensorSpec("q_proj", layers, hidden, hidden),
        TensorSpec("k_proj", layers, kv, hidden),
        TensorSpec("v_proj", layers, kv, hidden),
        TensorSpec("o_proj", layers, hidden, hidden),
        TensorSpec("gate_proj", layers, intermediate, hidden),
        TensorSpec("up_proj", layers, intermediate, hidden),
        TensorSpec("down_proj", layers, hidden, intermediate),
        TensorSpec("embedding", 1, vocabulary, hidden),
        TensorSpec("lm_head", 1, vocabulary, hidden),
    )


def parameter_total(specs: Iterable[TensorSpec]) -> int:
    return sum(spec.parameters for spec in specs)


def direct_sum_audit() -> dict[str, object]:
    return {
        "certified": False,
        "reason": (
            "Neither registered source theorem supplies a direct-sum result for "
            "many independent matrices sharing one jointly computed redundancy "
            "string and receiving adaptively generated, layer-dependent queries."
        ),
        "pigeonhole_division_allowed": False,
        "fix_other_matrices_rule": (
            "When all other matrices are fixed, the complete 8 GiB side state "
            "may still be an arbitrary function of the selected matrix; it "
            "cannot be divided by layer or tensor count without a theorem."
        ),
    }
