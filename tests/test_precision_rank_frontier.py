import pytest

from scripts.run_precision_rank_frontier import parse_points


def test_default_precision_rank_points_cover_fixed_boundaries() -> None:
    assert parse_points(None) == [
        (72, 8),
        (88, 6),
        (96, 6),
        (112, 4),
        (128, 4),
        (136, 4),
    ]


def test_explicit_points_are_deduplicated_and_ordered() -> None:
    assert parse_points(["128:4", "72:8", "128:4", "96:6"]) == [
        (72, 8),
        (96, 6),
        (128, 4),
    ]


def test_invalid_precision_rank_point_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_points(["rank-72"])
