from functools import lru_cache
from itertools import permutations

from experiments.exp_101a.run_experiment import build_oracles
from vortex_runtime.structured_direct_sum import (
    MultiplicationOracle,
    UniformScheme,
    bilinear_rank_lower_bound,
    canonical_shape,
    catalog_schemes,
    ceil_div,
    triple_cyclic_rank,
    weighted_ratio,
)


def test_published_triple_cyclic_rank_identity() -> None:
    assert triple_cyclic_rank(137, 8, 7) == 3_581_065


def test_bilinear_flattening_lower_bound() -> None:
    for shape in ((1, 7, 11), (5, 7, 11), (64, 64, 64)):
        lower = bilinear_rank_lower_bound(*shape)
        classical = shape[0] * shape[1] * shape[2]
        assert 0 < lower <= classical
    assert bilinear_rank_lower_bound(1, 7, 11) == 1 * 7 * 11


def test_canonical_shape_covers_all_tensor_permutations() -> None:
    expected = (5, 7, 11)
    for shape in permutations(expected):
        assert canonical_shape(shape) == expected


def test_classical_fallback_is_never_exceeded() -> None:
    oracle = MultiplicationOracle((), maximum_depth=4, structured_enabled=True)
    for shape in ((1, 1, 1), (5, 7, 11), (64, 64, 64)):
        assert oracle.cost(*shape) <= shape[0] * shape[1] * shape[2]


def test_strassen_exact_rank_reduces_power_of_two_cube() -> None:
    strassen = UniformScheme(2, 2, 2, 7, "strassen")
    oracle = MultiplicationOracle(
        (strassen,), maximum_depth=6, structured_enabled=False
    )
    assert oracle.cost(64, 64, 64) == 7**6
    assert oracle.cost(64, 64, 64) < 64**3


def test_composition_cannot_be_worse_than_either_arm() -> None:
    strassen = UniformScheme(2, 2, 2, 7, "strassen")
    plain = MultiplicationOracle(
        (strassen,), maximum_depth=5, structured_enabled=False
    )
    structured = MultiplicationOracle(
        (), maximum_depth=5, structured_enabled=True
    )
    combined = MultiplicationOracle(
        (strassen,), maximum_depth=5, structured_enabled=True
    )
    shape = (96, 216, 144)
    assert combined.cost(*shape) <= plain.cost(*shape)
    assert combined.cost(*shape) <= structured.cost(*shape)


def test_catalog_orientation_and_safe_dominance() -> None:
    rows = [
        {
            "status": "eligible",
            "shape": [2, 2, 2],
            "rank": 7,
            "key": "s",
        },
        {
            "status": "eligible",
            "shape": [2, 2, 2],
            "rank": 8,
            "key": "bad",
        },
        {
            "status": "eligible",
            "shape": [2, 2, 3],
            "rank": 11,
            "key": "r",
        },
        {
            "status": "exact_but_out_of_scope",
            "shape": [9, 9, 9],
            "rank": 1,
            "key": "skip",
        },
    ]
    schemes = catalog_schemes(rows)
    assert all(row.rank < row.classical_rank for row in schemes)
    assert any(
        (row.a, row.b, row.c, row.rank) == (2, 2, 2, 7)
        for row in schemes
    )
    assert not any(row.source == "bad" for row in schemes)
    signatures = {(row.a, row.b, row.c, row.rank) for row in schemes}
    for row in schemes:
        for oriented in set(permutations((row.a, row.b, row.c))):
            assert (*oriented, row.rank) in signatures


def _brute_cost(
    shape: tuple[int, int, int],
    schemes: tuple[UniformScheme, ...],
    *,
    depth: int,
    structured: bool,
) -> int:
    """Unpruned reference recurrence used only by the focused control."""

    @lru_cache(maxsize=None)
    def solve(m: int, k: int, n: int, remaining: int) -> int:
        best = m * k * n
        if remaining <= 0 or best == 1:
            return best
        for scheme in schemes:
            child = (
                ceil_div(m, scheme.a),
                ceil_div(k, scheme.b),
                ceil_div(n, scheme.c),
            )
            if child != (m, k, n):
                best = min(
                    best,
                    scheme.rank * solve(*child, remaining - 1),
                )
        if structured:
            base = (ceil_div(m, 6), ceil_div(k, 6), ceil_div(n, 6))
            if base != (m, k, n):
                for axis in range(3):
                    doubled = list(base)
                    doubled[axis] *= 2
                    best = min(
                        best,
                        137 * solve(*base, remaining - 1)
                        + 8 * solve(*tuple(doubled), remaining - 1),
                    )
        return best

    return solve(*shape, depth)


def test_exact_branch_and_bound_matches_unpruned_reference() -> None:
    schemes = catalog_schemes(
        [
            {
                "status": "eligible",
                "shape": [2, 2, 2],
                "rank": 7,
                "key": "strassen",
            },
            {
                "status": "eligible",
                "shape": [2, 2, 3],
                "rank": 11,
                "key": "rect",
            },
        ]
    )
    oracle = MultiplicationOracle(
        schemes,
        maximum_depth=3,
        structured_enabled=True,
        state_limit=100_000,
    )
    for shape in ((5, 7, 11), (12, 9, 7), (18, 20, 16)):
        assert oracle.cost(*shape) == _brute_cost(
            shape,
            schemes,
            depth=3,
            structured=True,
        )
    assert oracle.cache_info()["pruned_action_count"] > 0


def test_permutation_quotient_matches_unpruned_cost_and_reuses_cache() -> None:
    schemes = catalog_schemes(
        [
            {
                "status": "eligible",
                "shape": [2, 2, 3],
                "rank": 11,
                "key": "rect",
            }
        ]
    )
    oracle = MultiplicationOracle(
        schemes,
        maximum_depth=3,
        structured_enabled=True,
        state_limit=100_000,
    )
    expected = _brute_cost(
        (12, 9, 7), schemes, depth=3, structured=True
    )
    first = oracle.cost(12, 9, 7)
    states = oracle.cache_info()["state_count"]
    assert first == expected
    for shape in permutations((12, 9, 7)):
        assert oracle.cost(*shape) == expected
    assert oracle.cache_info()["state_count"] == states
    assert oracle.cache_info()["permutation_cache_hits"] > 0


def test_weighted_ratio_counts_repetition() -> None:
    oracle = MultiplicationOracle(
        (), maximum_depth=0, structured_enabled=False
    )
    ratio, candidate, baseline = weighted_ratio([(2, 3, 5, 7)], oracle)
    assert ratio == 1.0
    assert candidate == baseline == 2 * 3 * 5 * 7


def test_block_queries_receive_isolated_state_budgets() -> None:
    """A completed K must not consume the next frozen K query's state budget."""

    strassen = UniformScheme(2, 2, 2, 7, "strassen")
    first = build_oracles(
        (strassen,), maximum_depth=4, state_limit=10_000
    )
    first["catalog_plus_structured"].cost(64, 64, 64)
    assert first["catalog_plus_structured"].cache_info()["state_count"] > 0

    second = build_oracles(
        (strassen,), maximum_depth=4, state_limit=10_000
    )
    assert second["catalog_plus_structured"].cache_info()["state_count"] == 0
    assert (
        second["catalog_plus_structured"].cost(64, 64, 64)
        == first["catalog_plus_structured"].cost(64, 64, 64)
    )
