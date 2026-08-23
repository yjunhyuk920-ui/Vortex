from experiments.exp_101a.run_experiment import build_oracles
from vortex_runtime.structured_direct_sum import (
    MultiplicationOracle,
    UniformScheme,
    catalog_schemes,
    triple_cyclic_rank,
    weighted_ratio,
)


def test_published_triple_cyclic_rank_identity() -> None:
    assert triple_cyclic_rank(137, 8, 7) == 3_581_065


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
