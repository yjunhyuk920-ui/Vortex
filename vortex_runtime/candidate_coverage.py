from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class CandidateCoverageRow:
    position: int
    exact_token: int
    hot_token: int
    exact_token_rank: int
    hot_top1_margin: float
    exact_logit_gap_from_hot_top1: float

    @property
    def exact_match(self) -> bool:
        return self.exact_token == self.hot_token


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    """Return one-based rank of token_id under descending logits.

    Strictly larger logits precede the target. Tied logits share the same rank,
    which is the conservative useful convention for candidate-set coverage.
    """

    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if not 0 <= token_id < logits.shape[0]:
        raise ValueError("token_id is outside the vocabulary")
    target = logits[token_id]
    return int(torch.count_nonzero(logits > target).item()) + 1


def top1_margin(logits: torch.Tensor) -> float:
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if logits.numel() < 2:
        return float("inf")
    values = torch.topk(logits, k=2).values
    return float((values[0] - values[1]).item())


def coverage_at_k(
    rows: Iterable[CandidateCoverageRow],
    candidate_widths: Iterable[int],
) -> dict[str, float]:
    materialized = list(rows)
    if not materialized:
        return {str(width): 0.0 for width in candidate_widths}
    result: dict[str, float] = {}
    for width in candidate_widths:
        if width <= 0:
            raise ValueError("candidate widths must be positive")
        covered = sum(row.exact_token_rank <= width for row in materialized)
        result[str(width)] = covered / len(materialized)
    return result
