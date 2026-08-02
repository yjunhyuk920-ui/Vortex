from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from vortex_runtime.feasibility import GIB
from vortex_runtime.residual_proof import rowwise_residual_effect_bounds


@dataclass(frozen=True)
class TopKRowProofBudget:
    rows: int
    columns: int
    top_k: int
    source_bits: int
    hot_bits: int
    exact_residual_bytes_per_token: float
    exact_residual_gib_per_token: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class TopKRowCertificate:
    candidate: int
    certified: bool
    certified_margin: float
    exact_refined_rows: tuple[int, ...]
    strongest_outside_competitor: int
    strongest_outside_upper_bound: float
    candidate_exact_logit: float

    def to_dict(self) -> dict[str, int | float | bool | list[int]]:
        payload = asdict(self)
        payload["exact_refined_rows"] = list(self.exact_refined_rows)
        return payload


def topk_row_proof_budget(
    *,
    rows: int,
    columns: int,
    top_k: int,
    source_bits: int = 16,
    hot_bits: int = 4,
) -> TopKRowProofBudget:
    if min(rows, columns, top_k, source_bits, hot_bits) <= 0:
        raise ValueError("dimensions, top_k and precision must be positive")
    if top_k > rows:
        raise ValueError("top_k cannot exceed rows")
    if hot_bits >= source_bits:
        raise ValueError("hot_bits must be below source_bits")
    exact_bytes = top_k * columns * (source_bits - hot_bits) / 8
    return TopKRowProofBudget(
        rows=rows,
        columns=columns,
        top_k=top_k,
        source_bits=source_bits,
        hot_bits=hot_bits,
        exact_residual_bytes_per_token=exact_bytes,
        exact_residual_gib_per_token=exact_bytes / GIB,
    )


def certify_with_exact_topk_rows(
    *,
    hot_logits: torch.Tensor,
    activation: torch.Tensor,
    residual: torch.Tensor,
    residual_norms: torch.Tensor,
    column_block: int,
    top_k: int,
) -> TopKRowCertificate:
    """Refine hot top-K rows exactly and prove no outside row can win.

    The selected rows read their exact residual dot products. Every unselected
    row remains unread and is bounded using resident row/block norm metadata.
    The certificate is sound even when the exact winner is not in the hot top-K:
    in that case its outside upper bound prevents an unsafe acceptance.
    """

    if hot_logits.ndim != 1 or activation.ndim != 1:
        raise ValueError("hot_logits and activation must be one-dimensional")
    if residual.ndim != 2:
        raise ValueError("residual must have shape [rows, columns]")
    rows, columns = residual.shape
    if hot_logits.numel() != rows or activation.numel() != columns:
        raise ValueError("logit/activation dimensions do not match residual")
    if residual_norms.shape[0] != rows:
        raise ValueError("one metadata row is required per output row")
    if not 0 < top_k <= rows:
        raise ValueError("top_k must be in [1, rows]")

    hot = hot_logits.detach().to("cpu", torch.float32)
    x = activation.detach().to("cpu", torch.float32)
    selected = torch.topk(hot, k=top_k).indices.to(torch.long)
    exact_corrections = residual.detach().to("cpu", torch.float32)[selected] @ x
    refined_selected = hot[selected] + exact_corrections
    winner_offset = int(torch.argmax(refined_selected).item())
    candidate = int(selected[winner_offset].item())
    candidate_exact = float(refined_selected[winner_offset].item())

    effects = rowwise_residual_effect_bounds(
        residual_norms=residual_norms,
        activation=x,
        column_block=column_block,
    )
    outside_upper = hot + effects
    outside_upper[selected] = -torch.inf
    strongest = int(torch.argmax(outside_upper).item())
    strongest_upper = float(outside_upper[strongest].item())
    margin = candidate_exact - strongest_upper
    return TopKRowCertificate(
        candidate=candidate,
        certified=margin > 0.0,
        certified_margin=margin,
        exact_refined_rows=tuple(int(index) for index in selected.tolist()),
        strongest_outside_competitor=strongest,
        strongest_outside_upper_bound=strongest_upper,
        candidate_exact_logit=candidate_exact,
    )
