from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from vortex_runtime.feasibility import GIB
from vortex_runtime.residual_proof import rowwise_residual_effect_bounds


@dataclass(frozen=True)
class AdaptiveRowProofBudget:
    columns: int
    refined_rows: int
    source_bits: int
    hot_bits: int
    residual_bytes: float
    residual_gib: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveRowCertificate:
    candidate: int
    certified: bool
    certified_margin: float
    refined_rows: tuple[int, ...]
    iterations: int
    ambiguous_rows_remaining: int
    strongest_unread_competitor: int
    strongest_unread_upper_bound: float
    candidate_exact_logit: float

    @property
    def refined_row_count(self) -> int:
        return len(self.refined_rows)

    def to_dict(self) -> dict[str, int | float | bool | list[int]]:
        payload = asdict(self)
        payload["refined_rows"] = list(self.refined_rows)
        payload["refined_row_count"] = self.refined_row_count
        return payload


def adaptive_row_proof_budget(
    *,
    columns: int,
    refined_rows: int,
    source_bits: int = 16,
    hot_bits: int = 4,
) -> AdaptiveRowProofBudget:
    if min(columns, refined_rows, source_bits, hot_bits) <= 0:
        raise ValueError("columns, row count and precision must be positive")
    if hot_bits >= source_bits:
        raise ValueError("hot_bits must be below source_bits")
    residual_bytes = refined_rows * columns * (source_bits - hot_bits) / 8
    return AdaptiveRowProofBudget(
        columns=columns,
        refined_rows=refined_rows,
        source_bits=source_bits,
        hot_bits=hot_bits,
        residual_bytes=residual_bytes,
        residual_gib=residual_bytes / GIB,
    )


def certify_with_adaptive_exact_rows(
    *,
    hot_logits: torch.Tensor,
    activation: torch.Tensor,
    residual: torch.Tensor,
    residual_norms: torch.Tensor,
    column_block: int,
    initial_top_k: int = 1,
    refinement_batch: int = 16,
    max_refined_rows: int | None = None,
) -> AdaptiveRowCertificate:
    """Adaptively read exact residual rows until the global argmax is proven.

    The algorithm starts from the hot top-K rows. It computes their exact logits,
    then finds unread rows whose conservative upper bounds can still beat the
    current exact winner. Only the strongest ambiguous rows are refined next.
    Certification occurs when every unread upper bound is strictly below the
    best exact refined logit.
    """

    if hot_logits.ndim != 1 or activation.ndim != 1:
        raise ValueError("hot_logits and activation must be one-dimensional")
    if residual.ndim != 2:
        raise ValueError("residual must have shape [rows, columns]")
    rows, columns = residual.shape
    if hot_logits.numel() != rows or activation.numel() != columns:
        raise ValueError("hot/activation dimensions do not match residual")
    if residual_norms.shape[0] != rows:
        raise ValueError("one metadata row is required per output row")
    if not 0 < initial_top_k <= rows:
        raise ValueError("initial_top_k must be in [1, rows]")
    if refinement_batch <= 0:
        raise ValueError("refinement_batch must be positive")
    limit = rows if max_refined_rows is None else int(max_refined_rows)
    if limit < initial_top_k or limit > rows:
        raise ValueError("max_refined_rows must be in [initial_top_k, rows]")

    hot = hot_logits.detach().to("cpu", torch.float32)
    x = activation.detach().to("cpu", torch.float32)
    source_residual = residual.detach().to("cpu", torch.float32)
    effects = rowwise_residual_effect_bounds(
        residual_norms=residual_norms,
        activation=x,
        column_block=column_block,
    )
    unread_upper = hot + effects

    selected_mask = torch.zeros(rows, dtype=torch.bool)
    selected_indices = torch.topk(hot, k=initial_top_k).indices.to(torch.long)
    selected_mask[selected_indices] = True
    exact_values = torch.full((rows,), -torch.inf, dtype=torch.float32)
    exact_values[selected_indices] = (
        hot[selected_indices] + source_residual[selected_indices] @ x
    )
    iterations = 0

    while True:
        iterations += 1
        candidate = int(torch.argmax(exact_values).item())
        candidate_exact = float(exact_values[candidate].item())
        outside = unread_upper.clone()
        outside[selected_mask] = -torch.inf
        strongest = int(torch.argmax(outside).item())
        strongest_upper = float(outside[strongest].item())
        margin = candidate_exact - strongest_upper
        ambiguous_mask = (~selected_mask) & (unread_upper >= candidate_exact)
        ambiguous = torch.nonzero(ambiguous_mask, as_tuple=False).reshape(-1)

        if margin > 0.0 or not bool(torch.any(~selected_mask).item()):
            refined = torch.nonzero(selected_mask, as_tuple=False).reshape(-1)
            return AdaptiveRowCertificate(
                candidate=candidate,
                certified=True,
                certified_margin=margin,
                refined_rows=tuple(int(index) for index in refined.tolist()),
                iterations=iterations,
                ambiguous_rows_remaining=0,
                strongest_unread_competitor=strongest,
                strongest_unread_upper_bound=strongest_upper,
                candidate_exact_logit=candidate_exact,
            )

        current_count = int(selected_mask.sum().item())
        capacity = limit - current_count
        if capacity <= 0 or ambiguous.numel() == 0:
            refined = torch.nonzero(selected_mask, as_tuple=False).reshape(-1)
            return AdaptiveRowCertificate(
                candidate=candidate,
                certified=False,
                certified_margin=margin,
                refined_rows=tuple(int(index) for index in refined.tolist()),
                iterations=iterations,
                ambiguous_rows_remaining=int(ambiguous.numel()),
                strongest_unread_competitor=strongest,
                strongest_unread_upper_bound=strongest_upper,
                candidate_exact_logit=candidate_exact,
            )

        take = min(refinement_batch, capacity, int(ambiguous.numel()))
        ambiguous_upper = unread_upper[ambiguous]
        chosen_offsets = torch.topk(ambiguous_upper, k=take).indices
        chosen = ambiguous[chosen_offsets]
        selected_mask[chosen] = True
        exact_values[chosen] = hot[chosen] + source_residual[chosen] @ x
