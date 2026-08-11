r"""PROTOTYPE -- exhaustive nonlinear-fiber decision-depth screen.

For a set ``C`` of binary matrices, measure the largest deterministic
coordinate-decision-tree depth needed to return ``<q, W>`` for every binary
rank-one query ``q = r tensor u`` while ``W`` ranges over ``C``.

The 2x2 case exhausts every nonempty subset of the sixteen matrices.  The
2x3 control exhausts affine fibers, which isolates whether a nonlinear fiber
already beats every linear/coset witness at the smallest useful dimensions.

Run:

    .deps\exp076-venv\Scripts\python.exe scripts\prototype_rank_one_fiber_depth.py
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache


def parity(value: int) -> int:
    return value.bit_count() & 1


def rank_one_queries(rows: int, columns: int) -> tuple[int, ...]:
    queries = {0}
    for left in range(1, 1 << rows):
        for right in range(1, 1 << columns):
            query = 0
            for row in range(rows):
                if left & (1 << row):
                    query |= right << (row * columns)
            queries.add(query)
    return tuple(sorted(queries))


def label_mask(width: int, query: int) -> int:
    labels = 0
    for value in range(1 << width):
        if parity(value & query):
            labels |= 1 << value
    return labels


def coordinate_masks(width: int) -> tuple[int, ...]:
    masks = []
    for coordinate in range(width):
        mask = 0
        for value in range(1 << width):
            if value & (1 << coordinate):
                mask |= 1 << value
        masks.append(mask)
    return tuple(masks)


def make_depth_solver(width: int):
    universe = (1 << (1 << width)) - 1
    coordinates = coordinate_masks(width)

    @lru_cache(maxsize=None)
    def depth(fiber: int, labels: int) -> int:
        ones = fiber & labels
        if ones == 0 or ones == fiber:
            return 0
        best = width + 1
        for coordinate in coordinates:
            left = fiber & coordinate
            right = fiber & (universe ^ coordinate)
            if left == 0 or right == 0:
                continue
            candidate = 1 + max(depth(left, labels), depth(right, labels))
            best = min(best, candidate)
        if best > width:
            raise AssertionError("coordinates must distinguish binary matrices")
        return best

    return depth


def worst_depth(
    *, width: int, fiber: int, queries: tuple[int, ...], depth_solver
) -> int:
    return max(
        depth_solver(fiber, label_mask(width, query)) for query in queries
    )


def exhaustive_2x2() -> None:
    rows, columns = 2, 2
    width = rows * columns
    rank_one = rank_one_queries(rows, columns)
    all_linear = tuple(range(1 << width))
    depth_solver = make_depth_solver(width)
    optimum: dict[int, dict[str, tuple[int, int]]] = defaultdict(dict)
    for fiber in range(1, 1 << (1 << width)):
        cardinality = fiber.bit_count()
        for name, queries in (("rank_one", rank_one), ("all_linear", all_linear)):
            score = worst_depth(
                width=width,
                fiber=fiber,
                queries=queries,
                depth_solver=depth_solver,
            )
            previous = optimum[cardinality].get(name)
            if previous is None or score < previous[0]:
                optimum[cardinality][name] = (score, fiber)
    for cardinality in sorted(optimum):
        print(
            {
                "case": "2x2_all_nonlinear_fibers",
                "fiber_cardinality": cardinality,
                "rank_one_query_count": len(rank_one),
                "rank_one_minimax_depth": optimum[cardinality]["rank_one"][0],
                "rank_one_witness": hex(optimum[cardinality]["rank_one"][1]),
                "all_linear_minimax_depth": optimum[cardinality]["all_linear"][0],
                "all_linear_witness": hex(optimum[cardinality]["all_linear"][1]),
            }
        )


def linear_span(basis: tuple[int, ...]) -> frozenset[int]:
    values = {0}
    for vector in basis:
        values |= {value ^ vector for value in tuple(values)}
    return frozenset(values)


def all_subspaces(width: int) -> tuple[frozenset[int], ...]:
    levels: dict[int, set[frozenset[int]]] = {0: {frozenset({0})}}
    for dimension in range(width):
        following: set[frozenset[int]] = set()
        for subspace in levels[dimension]:
            for vector in range(1, 1 << width):
                if vector not in subspace:
                    following.add(
                        frozenset(
                            subspace
                            | {value ^ vector for value in tuple(subspace)}
                        )
                    )
        levels[dimension + 1] = following
    return tuple(
        subspace
        for dimension in sorted(levels)
        for subspace in sorted(levels[dimension], key=lambda item: tuple(sorted(item)))
    )


def affine_2x3() -> None:
    rows, columns = 2, 3
    width = rows * columns
    rank_one = rank_one_queries(rows, columns)
    depth_solver = make_depth_solver(width)
    optimum: dict[int, tuple[int, int]] = {}
    seen: set[int] = set()
    for subspace in all_subspaces(width):
        for offset in range(1 << width):
            fiber_values = {value ^ offset for value in subspace}
            fiber = sum(1 << value for value in fiber_values)
            if fiber in seen:
                continue
            seen.add(fiber)
            cardinality = len(fiber_values)
            score = worst_depth(
                width=width,
                fiber=fiber,
                queries=rank_one,
                depth_solver=depth_solver,
            )
            previous = optimum.get(cardinality)
            if previous is None or score < previous[0]:
                optimum[cardinality] = (score, fiber)
    for cardinality in sorted(optimum):
        print(
            {
                "case": "2x3_all_affine_fibers",
                "fiber_cardinality": cardinality,
                "rank_one_query_count": len(rank_one),
                "rank_one_minimax_depth": optimum[cardinality][0],
                "rank_one_witness": hex(optimum[cardinality][1]),
                "unique_affine_fibers": len(seen),
            }
        )


def main() -> None:
    exhaustive_2x2()
    affine_2x3()


if __name__ == "__main__":
    main()
