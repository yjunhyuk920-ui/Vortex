from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from typing import Iterable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class ExpertCandidateRow:
    position: int
    exact_token: int
    primary_margin: float
    primary_candidates: tuple[int, ...]
    secondary_candidates: tuple[int, ...]


@dataclass(frozen=True)
class CandidateUnionSummary:
    primary_k: int
    secondary_k: int
    coverage: float
    mean_candidate_count: float
    maximum_candidate_count: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class MarginFallbackSummary:
    threshold: float
    primary_k: int
    secondary_k: int
    secondary_invocation_fraction: float
    coverage: float
    mean_candidate_count: float
    maximum_candidate_count: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def topk_token_ids(logits: torch.Tensor, maximum_k: int) -> tuple[int, ...]:
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if maximum_k <= 0:
        raise ValueError("maximum_k must be positive")
    k = min(int(maximum_k), int(logits.numel()))
    return tuple(int(token) for token in torch.topk(logits, k=k).indices.tolist())


def candidate_union(
    row: ExpertCandidateRow,
    *,
    primary_k: int,
    secondary_k: int,
    include_secondary: bool = True,
) -> set[int]:
    if primary_k < 0 or secondary_k < 0:
        raise ValueError("candidate widths must be non-negative")
    candidates = set(row.primary_candidates[:primary_k])
    if include_secondary:
        candidates.update(row.secondary_candidates[:secondary_k])
    return candidates


def summarize_fixed_union(
    rows: Sequence[ExpertCandidateRow],
    *,
    primary_k: int,
    secondary_k: int,
) -> CandidateUnionSummary:
    if not rows:
        raise ValueError("at least one candidate row is required")
    sets = [
        candidate_union(
            row,
            primary_k=primary_k,
            secondary_k=secondary_k,
        )
        for row in rows
    ]
    hits = sum(row.exact_token in candidates for row, candidates in zip(rows, sets))
    counts = [len(candidates) for candidates in sets]
    return CandidateUnionSummary(
        primary_k=primary_k,
        secondary_k=secondary_k,
        coverage=hits / len(rows),
        mean_candidate_count=sum(counts) / len(counts),
        maximum_candidate_count=max(counts),
    )


def margin_thresholds(rows: Sequence[ExpertCandidateRow]) -> list[float]:
    if not rows:
        raise ValueError("at least one candidate row is required")
    margins = sorted({float(row.primary_margin) for row in rows})
    return [-inf, *margins, inf]


def summarize_margin_fallback(
    rows: Sequence[ExpertCandidateRow],
    *,
    threshold: float,
    primary_k: int,
    secondary_k: int,
) -> MarginFallbackSummary:
    if not rows:
        raise ValueError("at least one candidate row is required")

    triggered = 0
    hits = 0
    counts: list[int] = []
    for row in rows:
        invoke_secondary = row.primary_margin <= threshold
        triggered += int(invoke_secondary)
        candidates = candidate_union(
            row,
            primary_k=primary_k,
            secondary_k=secondary_k,
            include_secondary=invoke_secondary,
        )
        hits += int(row.exact_token in candidates)
        counts.append(len(candidates))

    return MarginFallbackSummary(
        threshold=float(threshold),
        primary_k=primary_k,
        secondary_k=secondary_k,
        secondary_invocation_fraction=triggered / len(rows),
        coverage=hits / len(rows),
        mean_candidate_count=sum(counts) / len(counts),
        maximum_candidate_count=max(counts),
    )


def margin_fallback_frontier(
    rows: Sequence[ExpertCandidateRow],
    *,
    primary_k: int,
    secondary_k: int,
) -> list[MarginFallbackSummary]:
    return [
        summarize_margin_fallback(
            rows,
            threshold=threshold,
            primary_k=primary_k,
            secondary_k=secondary_k,
        )
        for threshold in margin_thresholds(rows)
    ]


def best_budget_compatible_fallback(
    frontier: Iterable[MarginFallbackSummary],
    *,
    maximum_secondary_fraction: float,
) -> MarginFallbackSummary | None:
    if not 0 <= maximum_secondary_fraction <= 1:
        raise ValueError("maximum secondary fraction must be in [0, 1]")
    compatible = [
        item
        for item in frontier
        if item.secondary_invocation_fraction <= maximum_secondary_fraction
    ]
    if not compatible:
        return None
    return max(
        compatible,
        key=lambda item: (
            item.coverage,
            -item.secondary_invocation_fraction,
            -item.mean_candidate_count,
        ),
    )
