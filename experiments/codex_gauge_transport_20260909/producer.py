"""Finite certificate/witness and native schedule-transport constructors.

Prime-field claims and the explicitly declared scalar FP32 ABI stay separate.
No universal HF compiler, GPU implementation, or target cost acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import struct


def rank(matrix, p):
    a = [[int(v) % p for v in row] for row in matrix]
    nrows, ncols = len(a), len(a[0]) if a else 0
    pivot = 0
    for col in range(ncols):
        found = next((r for r in range(pivot, nrows) if a[r][col]), None)
        if found is None:
            continue
        a[pivot], a[found] = a[found], a[pivot]
        inv = pow(a[pivot][col], -1, p)
        a[pivot] = [v * inv % p for v in a[pivot]]
        for row in range(nrows):
            if row != pivot and a[row][col]:
                multiplier = a[row][col]
                a[row] = [(v - multiplier * w) % p for v, w in zip(a[row], a[pivot])]
        pivot += 1
        if pivot == nrows:
            break
    return pivot


def matvec(matrix, vector, p):
    return [sum(a * b for a, b in zip(row, vector)) % p for row in matrix]


def classify_gauges(a, b, c, p):
    if p < 2 or any(p % q == 0 for q in range(2, math.isqrt(p) + 1)):
        raise ValueError("p must be prime")
    n = len(a)
    if not n or any(len(m) != n or any(len(row) != n for row in m) for m in (a, b, c)):
        raise ValueError("nonempty square matrices of equal size required")
    if any(rank(m, p) != n for m in (a, b, c)):
        return {"status": "OUTSIDE_INVERTIBLE_DOMAIN"}
    for i, j, k in product(range(n), repeat=3):
        expected = c[i][j] % p if j == k else 0
        actual = a[i][j] * b[i][k] % p
        if expected != actual:
            x, y = [0] * n, [0] * n
            x[j], y[k] = 1, 1
            return {"status": "COUNTEREXAMPLE", "x": x, "y": y, "output_row": i}
    permutation = [next(j for j in range(n) if a[i][j] % p) for i in range(n)]
    da = [a[i][j] % p for i, j in enumerate(permutation)]
    db = [b[i][j] % p for i, j in enumerate(permutation)]
    return {"status": "CERTIFIED", "permutation": permutation, "diag_a": da, "diag_b": db}


def all_invertible(n, p):
    for values in product(range(p), repeat=n*n):
        matrix = [list(values[i*n:(i+1)*n]) for i in range(n)]
        if rank(matrix, p) == n:
            yield matrix


def f32(value):
    if math.isnan(value):
        return struct.unpack("<f", struct.pack("<I", 0x7fc00000))[0]
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bits(value):
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def balanced(values):
    values = list(values)
    if not values:
        return f32(0.0)
    size = 1 << (len(values)-1).bit_length()
    values += [f32(0.0)] * (size-len(values))
    while len(values) > 1:
        values = [f32(values[i] + values[i+1]) for i in range(0, len(values), 2)]
    return values[0]


def dense_reference(weights, x):
    if not weights or any(len(row) != len(x) for row in weights):
        raise ValueError("shape mismatch")
    return [balanced(f32(f32(w) * f32(v)) for w, v in zip(row, x)) for row in weights]


def check_permutation(order, length):
    if sorted(order) != list(range(length)):
        raise ValueError("invalid permutation")


@dataclass(frozen=True)
class TransportPlan:
    weights: tuple
    column_order: tuple
    row_order: tuple
    leaf_schedule: tuple


def compile_transport(weights, column_order, row_order):
    m, n = len(weights), len(weights[0])
    if not n or any(len(row) != n for row in weights):
        raise ValueError("nonempty rectangular weights required")
    check_permutation(column_order, n)
    check_permutation(row_order, m)
    inverse = [0] * n
    for new, old in enumerate(column_order):
        inverse[old] = new
    transformed = tuple(tuple(f32(weights[old_row][old_col]) for old_col in column_order) for old_row in row_order)
    return TransportPlan(transformed, tuple(column_order), tuple(row_order), tuple(inverse))


def execute_transport(plan, encoded_x):
    if len(encoded_x) != len(plan.leaf_schedule):
        raise ValueError("shape mismatch")
    return [balanced(f32(row[new] * f32(encoded_x[new])) for new in plan.leaf_schedule) for row in plan.weights]


def decode_rows(plan, encoded_y):
    if len(encoded_y) != len(plan.row_order):
        raise ValueError("shape mismatch")
    original = [None] * len(encoded_y)
    for new, old in enumerate(plan.row_order):
        original[old] = encoded_y[new]
    return original


def operation_ledger(m, n):
    if m < 1 or n < 1:
        raise ValueError("positive shape required")
    padded = 1 << (n-1).bit_length()
    return {
        "m": m, "n": n,
        "coefficient_reads": m*n,
        "original_and_transformed_fp32_payload_bytes": 8*m*n,
        "three_u32_index_vectors_bytes": 4*(2*n+m),
        "runtime_multiplications": m*n,
        "runtime_padded_additions": m*(padded-1),
        "same_reference_multiplications": m*n,
        "same_reference_padded_additions": m*(padded-1),
        "native_arithmetic_fraction_numerator": 1,
        "native_arithmetic_fraction_denominator": 1,
        "additional_costs": [
            "compile read and transformed write of every coefficient",
            "index construction and retention",
            "input/output permutation and gather/scatter",
            "Python object storage and allocator overhead in this reference",
            "CPU RAM/SSD/PCIe/HBM/GPU/KV/state in any deployment",
            "initialization, verification, repair, fallback, synchronization",
        ],
        "target_hardware_measured": False,
    }
