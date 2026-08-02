from vortex_runtime.multi_expert_candidates import (
    ExpertCandidateRow,
    best_budget_compatible_fallback,
    margin_fallback_frontier,
    summarize_fixed_union,
)


def rows() -> list[ExpertCandidateRow]:
    return [
        ExpertCandidateRow(
            position=0,
            exact_token=7,
            primary_margin=3.0,
            primary_candidates=(7, 1, 2, 3),
            secondary_candidates=(8, 9, 10, 11),
        ),
        ExpertCandidateRow(
            position=1,
            exact_token=9,
            primary_margin=0.5,
            primary_candidates=(1, 2, 3, 4),
            secondary_candidates=(9, 5, 6, 7),
        ),
        ExpertCandidateRow(
            position=2,
            exact_token=6,
            primary_margin=1.0,
            primary_candidates=(1, 2, 3, 4),
            secondary_candidates=(5, 6, 7, 8),
        ),
    ]


def test_fixed_union_uses_distinct_token_count() -> None:
    summary = summarize_fixed_union(
        rows(),
        primary_k=2,
        secondary_k=2,
    )
    assert summary.coverage == 1.0
    assert summary.maximum_candidate_count == 4


def test_margin_trigger_does_not_use_exact_target() -> None:
    frontier = margin_fallback_frontier(
        rows(),
        primary_k=2,
        secondary_k=2,
    )
    threshold_one = next(item for item in frontier if item.threshold == 1.0)
    assert threshold_one.secondary_invocation_fraction == 2 / 3
    assert threshold_one.coverage == 1.0


def test_best_budget_fallback_prefers_coverage_then_lower_invocation() -> None:
    frontier = margin_fallback_frontier(
        rows(),
        primary_k=2,
        secondary_k=2,
    )
    best = best_budget_compatible_fallback(
        frontier,
        maximum_secondary_fraction=2 / 3,
    )
    assert best is not None
    assert best.coverage == 1.0
    assert best.secondary_invocation_fraction == 2 / 3
