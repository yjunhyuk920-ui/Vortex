from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import torch

from vortex_runtime.feasibility import GIB


@dataclass(frozen=True)
class ResidualMetadataBudget:
    rows: int
    columns: int
    column_block: int
    blocks_per_row: int
    metadata_bits: int
    metadata_elements: int
    metadata_gib: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ArgmaxCertificate:
    candidate: int
    certified: bool
    certified_margin: float
    candidate_lower_bound: float
    strongest_competitor_upper_bound: float
    strongest_competitor: int

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def residual_metadata_budget(
    *,
    rows: int,
    columns: int,
    column_block: int,
    metadata_bits: int = 16,
) -> ResidualMetadataBudget:
    """Budget row-by-column-block residual norm metadata.

    One nonnegative norm is stored for each output row and input column block.
    The metadata is independent of token count and can certify output decisions
    without reading the omitted residual values themselves.
    """

    if min(rows, columns, column_block, metadata_bits) <= 0:
        raise ValueError("dimensions, block size and metadata precision must be positive")
    blocks = ceil(columns / column_block)
    elements = rows * blocks
    return ResidualMetadataBudget(
        rows=rows,
        columns=columns,
        column_block=column_block,
        blocks_per_row=blocks,
        metadata_bits=metadata_bits,
        metadata_elements=elements,
        metadata_gib=elements * metadata_bits / 8 / GIB,
    )


def rowwise_residual_block_norms(
    residual: torch.Tensor,
    *,
    column_block: int,
) -> torch.Tensor:
    """Return Frobenius/L2 norms for each row and input-column block."""

    if residual.ndim != 2:
        raise ValueError("residual must have shape [rows, columns]")
    if column_block <= 0:
        raise ValueError("column_block must be positive")
    source = residual.detach().to("cpu", torch.float32)
    rows, columns = source.shape
    blocks = ceil(columns / column_block)
    norms = torch.empty(rows, blocks, dtype=torch.float32)
    for block_index, start in enumerate(range(0, columns, column_block)):
        end = min(start + column_block, columns)
        norms[:, block_index] = torch.linalg.vector_norm(
            source[:, start:end],
            dim=1,
        )
    return norms


def activation_block_norms(
    activation: torch.Tensor,
    *,
    column_block: int,
) -> torch.Tensor:
    """Return L2 norms for one activation vector split into column blocks."""

    if activation.ndim != 1:
        raise ValueError("activation must be one-dimensional")
    if column_block <= 0:
        raise ValueError("column_block must be positive")
    source = activation.detach().to("cpu", torch.float32)
    values: list[torch.Tensor] = []
    for start in range(0, source.numel(), column_block):
        end = min(start + column_block, source.numel())
        values.append(torch.linalg.vector_norm(source[start:end]))
    return torch.stack(values)


def rowwise_residual_effect_bounds(
    *,
    residual_norms: torch.Tensor,
    activation: torch.Tensor,
    column_block: int,
) -> torch.Tensor:
    """Bound ``|(W-W_hot)x|`` independently for every output row.

    For row ``r`` and column blocks ``b``:

    ``|sum_b R[r,b] x[b]| <= sum_b ||R[r,b]||_2 ||x[b]||_2``.

    Only norm metadata is needed at runtime; residual values remain unread.
    """

    if residual_norms.ndim != 2:
        raise ValueError("residual_norms must have shape [rows, blocks]")
    x_norms = activation_block_norms(
        activation,
        column_block=column_block,
    )
    if residual_norms.shape[1] != x_norms.numel():
        raise ValueError("metadata block count does not match activation")
    return residual_norms.to(torch.float32) @ x_norms


def certify_linear_argmax(
    *,
    approximate_logits: torch.Tensor,
    activation: torch.Tensor,
    residual_norms: torch.Tensor,
    column_block: int,
    candidate: int | None = None,
) -> ArgmaxCertificate:
    """Certify that a hot linear projection has the exact argmax.

    Let ``W = W_hot + R``. The candidate lower bound is its approximate logit
    minus the residual-effect bound. Every competitor upper bound is its
    approximate logit plus its residual-effect bound. If the candidate lower
    bound exceeds all competitor upper bounds, the exact argmax is proven.
    """

    if approximate_logits.ndim != 1:
        raise ValueError("approximate_logits must be one-dimensional")
    if residual_norms.shape[0] != approximate_logits.numel():
        raise ValueError("one metadata row is required per output logit")
    logits = approximate_logits.detach().to("cpu", torch.float32)
    selected = int(torch.argmax(logits).item()) if candidate is None else int(candidate)
    if selected < 0 or selected >= logits.numel():
        raise ValueError("candidate is outside the output range")

    effects = rowwise_residual_effect_bounds(
        residual_norms=residual_norms,
        activation=activation,
        column_block=column_block,
    )
    lower = float((logits[selected] - effects[selected]).item())
    competitor_upper = logits + effects
    competitor_upper[selected] = -torch.inf
    strongest = int(torch.argmax(competitor_upper).item())
    upper = float(competitor_upper[strongest].item())
    margin = lower - upper
    return ArgmaxCertificate(
        candidate=selected,
        certified=margin > 0.0,
        certified_margin=margin,
        candidate_lower_bound=lower,
        strongest_competitor_upper_bound=upper,
        strongest_competitor=strongest,
    )


def tiled_bilinear_residual_bound(
    *,
    left: torch.Tensor,
    right: torch.Tensor,
    tile_norms: torch.Tensor,
    row_block: int,
    column_block: int,
) -> float:
    """Bound ``|left.T @ residual @ right|`` from tile Frobenius norms.

    This is the model-wide primitive for a future backward decision proof. A
    dense logit-margin adjoint supplies ``left`` and the layer activation
    supplies ``right``. Each residual tile contributes at most the product of
    its Frobenius norm and the two matching vector-block norms.
    """

    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("left and right must be one-dimensional")
    if row_block <= 0 or column_block <= 0:
        raise ValueError("tile dimensions must be positive")
    expected_rows = ceil(left.numel() / row_block)
    expected_columns = ceil(right.numel() / column_block)
    if tile_norms.shape != (expected_rows, expected_columns):
        raise ValueError("tile_norms shape does not match vector blocks")

    left_norms = activation_block_norms(left, column_block=row_block)
    right_norms = activation_block_norms(right, column_block=column_block)
    weights = left_norms[:, None] * right_norms[None, :]
    return float((tile_norms.to(torch.float32) * weights).sum().item())


def residual_tile_norms(
    residual: torch.Tensor,
    *,
    row_block: int,
    column_block: int,
) -> torch.Tensor:
    """Build a 2-D Frobenius-norm certificate table for a residual matrix."""

    if residual.ndim != 2:
        raise ValueError("residual must be two-dimensional")
    if row_block <= 0 or column_block <= 0:
        raise ValueError("tile dimensions must be positive")
    source = residual.detach().to("cpu", torch.float32)
    rows, columns = source.shape
    result = torch.empty(
        ceil(rows / row_block),
        ceil(columns / column_block),
        dtype=torch.float32,
    )
    for row_index, row_start in enumerate(range(0, rows, row_block)):
        row_end = min(row_start + row_block, rows)
        for column_index, column_start in enumerate(
            range(0, columns, column_block)
        ):
            column_end = min(column_start + column_block, columns)
            result[row_index, column_index] = torch.linalg.vector_norm(
                source[row_start:row_end, column_start:column_end]
            )
    return result
