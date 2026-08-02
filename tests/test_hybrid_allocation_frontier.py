import pytest

from scripts.run_hybrid_allocation_frontier import parse_points


def test_default_hybrid_points_cover_precision_boundaries() -> None:
    assert parse_points(None) == [
        (56, 96, 6),
        (64, 136, 4),
        (88, 136, 4),
        (96, 136, 4),
    ]


def test_explicit_hybrid_points_are_deduplicated() -> None:
    assert parse_points([
        "88:136:4",
        "56:96:6",
        "88:136:4",
    ]) == [
        (56, 96, 6),
        (88, 136, 4),
    ]


def test_invalid_hybrid_point_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_points(["137:136:4"])
