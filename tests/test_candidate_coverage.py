import torch

from vortex_runtime.candidate_coverage import (
    CandidateCoverageRow,
    coverage_at_k,
    token_rank,
    top1_margin,
)


def test_token_rank_uses_one_based_conservative_ties() -> None:
    logits = torch.tensor([0.5, 2.0, 1.0, 1.0, -1.0])
    assert token_rank(logits, 1) == 1
    assert token_rank(logits, 2) == 2
    assert token_rank(logits, 3) == 2
    assert token_rank(logits, 4) == 5


def test_top1_margin_returns_gap_between_best_two() -> None:
    logits = torch.tensor([-1.0, 3.5, 2.0, 0.0])
    assert top1_margin(logits) == 1.5


def test_coverage_at_k_counts_exact_token_rank() -> None:
    rows = [
        CandidateCoverageRow(0, 10, 10, 1, 0.5, 0.0),
        CandidateCoverageRow(1, 11, 12, 2, 0.1, 0.2),
        CandidateCoverageRow(2, 13, 14, 7, 0.2, 1.1),
        CandidateCoverageRow(3, 15, 16, 40, 0.4, 2.4),
    ]
    coverage = coverage_at_k(rows, (1, 2, 8, 32, 64))
    assert coverage == {
        "1": 0.25,
        "2": 0.5,
        "8": 0.75,
        "32": 0.75,
        "64": 1.0,
    }


def test_coverage_rejects_nonpositive_candidate_width() -> None:
    rows = [CandidateCoverageRow(0, 1, 1, 1, 0.0, 0.0)]
    try:
        coverage_at_k(rows, (0,))
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("nonpositive width must be rejected")
