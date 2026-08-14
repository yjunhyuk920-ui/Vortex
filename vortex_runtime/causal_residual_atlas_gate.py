"""Frozen arithmetic and oracle rules for the first real-weight Atlas Gate.

This module does not load a checkpoint or execute a Transformer.  It fixes the
finite population, success count, page enumeration, and favorable page-choice
rule before any real-weight result exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
import math
from typing import Sequence

import numpy as np

from vortex_runtime.causal_residual_atlas import derive_atlas_budget


@dataclass(frozen=True)
class AtlasFirstDecodeGate:
    rank: int
    page_columns: int
    requested_cold_fraction: Decimal
    service_tokens: int
    evaluation_prompts: int
    required_families: int
    prompts_per_family: int
    positions_per_prompt: int
    teacher_position_index: int
    layer_indices: tuple[int, ...]
    projection_input_widths: tuple[tuple[str, int], ...]
    pages_per_projection: tuple[tuple[str, int], ...]
    token_states: int
    projection_branches: int
    oracle_page_candidates: int
    minimum_coverage: Decimal
    required_token_successes: int
    maximum_token_failures: int
    required_family_successes: int
    one_token_failure_coverage: Decimal
    maximum_mean_kl: Decimal
    maximum_p95_kl: Decimal

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        return {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in row.items()
        }


def required_successes(population: int, coverage: Decimal) -> int:
    if population <= 0:
        raise ValueError("population must be positive")
    if not Decimal(0) <= coverage <= Decimal(1):
        raise ValueError("coverage must be in [0, 1]")
    return int(
        (Decimal(population) * coverage).to_integral_value(
            rounding=ROUND_CEILING
        )
    )


def derive_first_decode_gate() -> AtlasFirstDecodeGate:
    """Return the preregistered smallest pinned-checkpoint Gate.

    The two projections are both required by the unchanged checkpoint.  The
    middle full-attention layer avoids a boundary-only result.  Every possible
    contiguous 64-column page is granted to the non-deployable oracle.
    """

    rank = 16
    page_columns = 64
    requested_fraction = Decimal("0.002")
    service_tokens = 64
    evaluation_prompts = 18
    required_families = 6
    prompts_per_family = 3
    positions_per_prompt = 1
    # Index 0 is the unchanged logit emitted by the final prompt position.
    # Index 1 is the first module call after prefill: the first committed
    # generated token is processed to predict the second generated token.
    teacher_position_index = 1
    layer_indices = (11,)
    widths = (("q_proj", 1024), ("down_proj", 3584))
    pages = tuple(
        (name, math.ceil(width / page_columns)) for name, width in widths
    )

    target_budget = derive_atlas_budget(
        rank=rank,
        requested_cold_fraction=requested_fraction,
        page_columns=page_columns,
        service_tokens=service_tokens,
    )
    if target_budget.minimum_traffic_coverage is None:
        raise ValueError("registered Atlas point has no feasible coverage")
    coverage = target_budget.minimum_traffic_coverage
    token_states = evaluation_prompts * positions_per_prompt
    projection_branches = token_states * len(widths) * len(layer_indices)
    oracle_page_candidates = token_states * len(layer_indices) * sum(
        count for _, count in pages
    )
    required_tokens = required_successes(token_states, coverage)
    family_states = prompts_per_family * positions_per_prompt

    return AtlasFirstDecodeGate(
        rank=rank,
        page_columns=page_columns,
        requested_cold_fraction=requested_fraction,
        service_tokens=service_tokens,
        evaluation_prompts=evaluation_prompts,
        required_families=required_families,
        prompts_per_family=prompts_per_family,
        positions_per_prompt=positions_per_prompt,
        teacher_position_index=teacher_position_index,
        layer_indices=layer_indices,
        projection_input_widths=widths,
        pages_per_projection=pages,
        token_states=token_states,
        projection_branches=projection_branches,
        oracle_page_candidates=oracle_page_candidates,
        minimum_coverage=coverage,
        required_token_successes=required_tokens,
        maximum_token_failures=token_states - required_tokens,
        required_family_successes=required_successes(
            family_states, coverage
        ),
        one_token_failure_coverage=(
            Decimal(token_states - 1) / Decimal(token_states)
        ),
        maximum_mean_kl=Decimal("0.02"),
        maximum_p95_kl=Decimal("0.05"),
    )


@dataclass(frozen=True)
class FavorablePageChoice:
    page_index: int
    target_top1: int
    candidate_top1: int
    top1_match: bool
    target_to_candidate_kl: float


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    return shifted - math.log(float(np.exp(shifted).sum()))


def select_favorable_page(
    baseline_logits: Sequence[float] | np.ndarray,
    candidate_logits: Sequence[Sequence[float]] | np.ndarray,
    *,
    page_indices: Sequence[int] | None = None,
) -> FavorablePageChoice:
    """Choose the frozen non-deployable exact-reference page.

    A top-1-preserving page is always preferred.  Within that set the oracle
    minimizes target-to-candidate KL and then the page index.  When no page
    preserves top-1, the minimum-KL page is retained for diagnostics but the
    returned choice is a failure.
    """

    baseline = np.asarray(baseline_logits, dtype=np.float64)
    candidates = np.asarray(candidate_logits, dtype=np.float64)
    if baseline.ndim != 1 or baseline.size < 2:
        raise ValueError("baseline_logits must be a vector of size >= 2")
    if candidates.ndim != 2 or candidates.shape[1] != baseline.size:
        raise ValueError("candidate_logits must be [pages, vocabulary]")
    if candidates.shape[0] == 0:
        raise ValueError("at least one page candidate is required")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(candidates)):
        raise ValueError("logits must be finite")

    indices = (
        tuple(range(candidates.shape[0]))
        if page_indices is None
        else tuple(int(value) for value in page_indices)
    )
    if len(indices) != candidates.shape[0] or len(set(indices)) != len(indices):
        raise ValueError("page_indices must be unique and match candidate rows")
    if any(value < 0 for value in indices):
        raise ValueError("page_indices must be non-negative")

    target_top1 = int(np.argmax(baseline))
    target_log_prob = _log_softmax(baseline)
    target_prob = np.exp(target_log_prob)
    rows: list[tuple[bool, float, int, int]] = []
    for page_index, logits in zip(indices, candidates):
        candidate_top1 = int(np.argmax(logits))
        candidate_log_prob = _log_softmax(logits)
        divergence = float(
            np.sum(target_prob * (target_log_prob - candidate_log_prob))
        )
        divergence = max(0.0, divergence)
        rows.append(
            (
                candidate_top1 == target_top1,
                divergence,
                page_index,
                candidate_top1,
            )
        )

    matching = [row for row in rows if row[0]]
    selected = min(matching or rows, key=lambda row: (row[1], row[2]))
    return FavorablePageChoice(
        page_index=selected[2],
        target_top1=target_top1,
        candidate_top1=selected[3],
        top1_match=selected[0],
        target_to_candidate_kl=selected[1],
    )
