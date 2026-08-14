r"""PROTOTYPE -- wipe me after the functional-span question is settled.

Question: if ``h`` exact linear forms of an arbitrary binary ``m x n``
matrix are resident for free, how many raw matrix bits are still necessary in
the worst case to return the *whole* product ``W x`` for every ``x``?

This deliberately tiny exhaustive program enumerates every resident linear
subspace for ambient dimensions up to six.  It is a logic probe, not a Vortex
runtime component and not a model experiment.

Run:

    .deps\exp076-venv\Scripts\python.exe scripts\prototype_functional_span_cover.py
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math


REGISTERED_Q4_COEFFICIENTS = 403_747_897_344
HOT_ADVICE_BITS = 8 * (1 << 30) * 8


def linear_span(basis: tuple[int, ...]) -> frozenset[int]:
    values = {0}
    for vector in basis:
        values |= {value ^ vector for value in tuple(values)}
    return frozenset(values)


def all_subspaces(width: int) -> dict[int, tuple[frozenset[int], ...]]:
    """Enumerate every binary subspace once, grouped by dimension."""

    levels: dict[int, set[frozenset[int]]] = {0: {frozenset({0})}}
    universe = range(1, 1 << width)
    for dimension in range(width):
        following: set[frozenset[int]] = set()
        for subspace in levels[dimension]:
            for vector in universe:
                if vector in subspace:
                    continue
                following.add(
                    frozenset(subspace | {value ^ vector for value in subspace})
                )
        levels[dimension + 1] = following
    return {
        dimension: tuple(sorted(spaces, key=lambda space: tuple(sorted(space))))
        for dimension, spaces in levels.items()
    }


def row_query_basis(rows: int, columns: int, x: int) -> tuple[int, ...]:
    mask = (1 << columns) - 1
    return tuple((x & mask) << (row * columns) for row in range(rows))


def minimum_raw_coordinates(
    *,
    width: int,
    hot_subspace: frozenset[int],
    required_vectors: tuple[int, ...],
) -> int:
    """Minimum coordinate set P with U contained in H + span(P)."""

    coordinates = tuple(1 << index for index in range(width))
    for probe_count in range(width + 1):
        for selected in combinations(coordinates, probe_count):
            reachable = linear_span(tuple(hot_subspace) + selected)
            if all(vector in reachable for vector in required_vectors):
                return probe_count
    raise AssertionError("the full coordinate basis must recover every query")


def worst_case_product_reads(
    *, rows: int, columns: int, hot_subspace: frozenset[int]
) -> int:
    width = rows * columns
    return max(
        minimum_raw_coordinates(
            width=width,
            hot_subspace=hot_subspace,
            required_vectors=row_query_basis(rows, columns, x),
        )
        for x in range(1 << columns)
    )


def canonical_basis(subspace: frozenset[int], width: int) -> tuple[int, ...]:
    candidates = set(subspace)
    basis: list[int] = []
    for pivot in reversed(range(width)):
        vector = next(
            (value for value in sorted(candidates) if value & (1 << pivot)),
            None,
        )
        if vector is None:
            continue
        basis.append(vector)
        candidates = {
            value ^ vector if value & (1 << pivot) else value
            for value in candidates
        }
    return tuple(basis)


@dataclass(frozen=True)
class Optimum:
    rows: int
    columns: int
    hot_dimension: int
    raw_reads: int
    witness_basis: tuple[int, ...]
    searched_subspaces: int


def solve_case(rows: int, columns: int) -> tuple[Optimum, ...]:
    width = rows * columns
    subspaces = all_subspaces(width)
    answers: list[Optimum] = []
    for hot_dimension, candidates in subspaces.items():
        scored = (
            (
                worst_case_product_reads(
                    rows=rows,
                    columns=columns,
                    hot_subspace=hot_subspace,
                ),
                hot_subspace,
            )
            for hot_subspace in candidates
        )
        raw_reads, witness = min(scored, key=lambda item: item[0])
        answer = Optimum(
            rows=rows,
            columns=columns,
            hot_dimension=hot_dimension,
            raw_reads=raw_reads,
            witness_basis=canonical_basis(witness, width),
            searched_subspaces=len(candidates),
        )
        answers.append(answer)
        print(answer)
    return tuple(answers)


def qary_entropy(radius: float, alphabet_size: int) -> float:
    if radius == 0.0:
        return 0.0
    return (
        radius * math.log(alphabet_size - 1, alphabet_size)
        - radius * math.log(radius, alphabet_size)
        - (1.0 - radius) * math.log(1.0 - radius, alphabet_size)
    )


def registered_gf16_screen() -> dict[str, float | int]:
    alphabet_size = 16
    bits_per_symbol = 4
    hot_symbols = HOT_ADVICE_BITS // bits_per_symbol
    hot_rate = hot_symbols / REGISTERED_Q4_COEFFICIENTS
    target_entropy = 1.0 - hot_rate
    lower = 0.0
    upper = (alphabet_size - 1) / alphabet_size
    for _ in range(200):
        midpoint = (lower + upper) / 2.0
        if qary_entropy(midpoint, alphabet_size) < target_entropy:
            lower = midpoint
        else:
            upper = midpoint
    return {
        "alphabet_size": alphabet_size,
        "hot_symbols": hot_symbols,
        "hot_symbol_rate": hot_rate,
        "covering_radius_root": upper,
        "covering_radius_percent": 100.0 * upper,
        "qary_entropy_at_root": qary_entropy(upper, alphabet_size),
    }


def main() -> None:
    for rows, columns in ((1, 4), (2, 2), (2, 3), (3, 2)):
        print({"case": [rows, columns], "ambient_dimension": rows * columns})
        solve_case(rows, columns)
    print({"registered_all_linear_gf16_screen": registered_gf16_screen()})


if __name__ == "__main__":
    main()
