"""Exact non-mask attention-probability sparsity accounting.

Only returned post-softmax probabilities exactly equal to positive or negative
zero at causally eligible positions may skip Value accumulation.  Causal or
padding-mask zeros are excluded from the eligible population and from savings.
QK scoring and softmax remain fully charged because zero status is known only
after those operations complete.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence


class AttentionProbabilitySparsityError(ValueError):
    """Raised when attention tensors or accounting are malformed."""


def unsigned_width(maximum_value: int) -> int:
    if maximum_value < 0:
        raise AttentionProbabilitySparsityError(
            "unsigned width requires a nonnegative maximum"
        )
    return max(1, math.ceil(max(1, maximum_value).bit_length() / 8))


@dataclass(frozen=True)
class AttentionProbabilityAccounting:
    model_id: str
    prompt_family: str
    phase: str
    decode_step: int
    layer_index: int
    batch_size: int
    head_count: int
    query_length: int
    key_length: int
    past_length: int
    head_dimension: int
    eligible_probability_count: int
    masked_probability_count: int
    exact_nonmask_zero_count: int
    nonzero_probability_count: int
    qk_operation_terms: int
    softmax_operation_terms: int
    dense_value_operation_terms: int
    sparse_value_operation_terms: int
    probability_scan_terms: int
    dense_attention_operation_terms: int
    sparse_attention_operation_terms: int
    dense_attention_bytes: int
    sparse_attention_bytes: int
    metadata_bytes: int
    minimum_probability: float
    maximum_probability: float
    maximum_row_sum_error: float

    @property
    def exact_nonmask_zero_fraction(self) -> float:
        return self.exact_nonmask_zero_count / self.eligible_probability_count

    @property
    def attention_operation_fraction(self) -> float:
        return self.sparse_attention_operation_terms / self.dense_attention_operation_terms

    @property
    def attention_query_byte_fraction(self) -> float:
        return (self.sparse_attention_bytes + self.metadata_bytes) / self.dense_attention_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_family": self.prompt_family,
            "phase": self.phase,
            "decode_step": self.decode_step,
            "layer_index": self.layer_index,
            "batch_size": self.batch_size,
            "head_count": self.head_count,
            "query_length": self.query_length,
            "key_length": self.key_length,
            "past_length": self.past_length,
            "head_dimension": self.head_dimension,
            "eligible_probability_count": self.eligible_probability_count,
            "masked_probability_count": self.masked_probability_count,
            "exact_nonmask_zero_count": self.exact_nonmask_zero_count,
            "nonzero_probability_count": self.nonzero_probability_count,
            "exact_nonmask_zero_fraction": self.exact_nonmask_zero_fraction,
            "qk_operation_terms": self.qk_operation_terms,
            "softmax_operation_terms": self.softmax_operation_terms,
            "dense_value_operation_terms": self.dense_value_operation_terms,
            "sparse_value_operation_terms": self.sparse_value_operation_terms,
            "probability_scan_terms": self.probability_scan_terms,
            "dense_attention_operation_terms": self.dense_attention_operation_terms,
            "sparse_attention_operation_terms": self.sparse_attention_operation_terms,
            "attention_operation_fraction": self.attention_operation_fraction,
            "dense_attention_bytes": self.dense_attention_bytes,
            "sparse_attention_bytes": self.sparse_attention_bytes,
            "metadata_bytes": self.metadata_bytes,
            "attention_query_byte_fraction": self.attention_query_byte_fraction,
            "minimum_probability": self.minimum_probability,
            "maximum_probability": self.maximum_probability,
            "maximum_row_sum_error": self.maximum_row_sum_error,
        }


def causal_eligible_mask(
    *,
    torch: Any,
    batch_size: int,
    head_count: int,
    query_length: int,
    key_length: int,
    past_length: int,
    device: Any,
) -> Any:
    """Return the causal eligibility mask for an unpadded single-sequence batch."""

    if min(batch_size, head_count, query_length, key_length) <= 0:
        raise AttentionProbabilitySparsityError(
            "attention dimensions must be positive"
        )
    if past_length < 0 or key_length != past_length + query_length:
        raise AttentionProbabilitySparsityError(
            "key length must equal past length plus query length"
        )
    query_positions = past_length + torch.arange(query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    base = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    return base.reshape(1, 1, query_length, key_length).expand(
        batch_size, head_count, query_length, key_length
    )


def account_attention_probabilities(
    probabilities: Any,
    *,
    model_id: str,
    prompt_family: str,
    phase: str,
    decode_step: int,
    layer_index: int,
    head_dimension: int,
    past_length: int,
    row_sum_tolerance: float = 1e-5,
) -> AttentionProbabilityAccounting:
    """Validate and account one returned attention-probability tensor."""

    import torch

    if phase not in {"prefill", "first_decode", "warm_decode"}:
        raise AttentionProbabilitySparsityError(f"unsupported phase: {phase}")
    if head_dimension <= 0:
        raise AttentionProbabilitySparsityError("head dimension must be positive")
    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 4:
        raise AttentionProbabilitySparsityError(
            "probabilities must be a rank-four torch tensor"
        )
    batch_size, head_count, query_length, key_length = (
        int(value) for value in probabilities.shape
    )
    eligible = causal_eligible_mask(
        torch=torch,
        batch_size=batch_size,
        head_count=head_count,
        query_length=query_length,
        key_length=key_length,
        past_length=past_length,
        device=probabilities.device,
    )
    if not bool(torch.isfinite(probabilities).all()):
        raise AttentionProbabilitySparsityError("attention contains NaN or infinity")
    if bool((probabilities < 0).any()):
        raise AttentionProbabilitySparsityError("attention contains negative probability")
    masked_values = probabilities[~eligible]
    if masked_values.numel() and bool((masked_values != 0).any()):
        raise AttentionProbabilitySparsityError(
            "causally masked attention entries are not exactly zero"
        )
    eligible_values = probabilities[eligible]
    eligible_count = int(eligible_values.numel())
    if eligible_count <= 0:
        raise AttentionProbabilitySparsityError("eligible attention population is empty")
    exact_zero_count = int((eligible_values == 0).sum().item())
    nonzero_count = eligible_count - exact_zero_count

    masked = int(probabilities.numel()) - eligible_count
    row_sums = (probabilities * eligible.to(probabilities.dtype)).sum(dim=-1)
    row_sum_error = float((row_sums - 1.0).abs().max().item())
    if row_sum_error > row_sum_tolerance:
        raise AttentionProbabilitySparsityError(
            f"attention row sum error {row_sum_error} exceeds {row_sum_tolerance}"
        )

    qk_terms = head_dimension * eligible_count
    softmax_terms = eligible_count
    dense_value_terms = head_dimension * eligible_count
    sparse_value_terms = head_dimension * nonzero_count
    scan_terms = eligible_count
    dense_terms = qk_terms + softmax_terms + dense_value_terms
    sparse_terms = qk_terms + softmax_terms + sparse_value_terms + scan_terms

    # Favorable logical bytes: score/probability plus Value-vector reads only.
    dense_bytes = eligible_count * (4 + head_dimension * 4)
    sparse_bytes = nonzero_count * (4 + head_dimension * 4)
    index_width = unsigned_width(max(0, key_length - 1))
    row_count = batch_size * head_count * query_length
    pointer_width = unsigned_width(nonzero_count)
    metadata = nonzero_count * index_width + (row_count + 1) * pointer_width

    return AttentionProbabilityAccounting(
        model_id=model_id,
        prompt_family=prompt_family,
        phase=phase,
        decode_step=decode_step,
        layer_index=layer_index,
        batch_size=batch_size,
        head_count=head_count,
        query_length=query_length,
        key_length=key_length,
        past_length=past_length,
        head_dimension=head_dimension,
        eligible_probability_count=eligible_count,
        masked_probability_count=masked,
        exact_nonmask_zero_count=exact_zero_count,
        nonzero_probability_count=nonzero_count,
        qk_operation_terms=qk_terms,
        softmax_operation_terms=softmax_terms,
        dense_value_operation_terms=dense_value_terms,
        sparse_value_operation_terms=sparse_value_terms,
        probability_scan_terms=scan_terms,
        dense_attention_operation_terms=dense_terms,
        sparse_attention_operation_terms=sparse_terms,
        dense_attention_bytes=dense_bytes,
        sparse_attention_bytes=sparse_bytes,
        metadata_bytes=metadata,
        minimum_probability=float(eligible_values.min().item()),
        maximum_probability=float(eligible_values.max().item()),
        maximum_row_sum_error=row_sum_error,
    )


def zero_skipped_value_accumulation(
    probabilities: Sequence[float],
    values: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return dense and exact-zero-skipped scalar-loop Value accumulation."""

    if len(probabilities) != len(values) or not values:
        raise AttentionProbabilitySparsityError(
            "probability/value populations must be nonempty and aligned"
        )
    width = len(values[0])
    if width <= 0 or any(len(row) != width for row in values):
        raise AttentionProbabilitySparsityError("Value rows have inconsistent width")
    dense = [0.0] * width
    sparse = [0.0] * width
    for probability, row in zip(probabilities, values):
        scalar = float(probability)
        for index, value in enumerate(row):
            product = scalar * float(value)
            dense[index] += product
            if scalar != 0.0:
                sparse[index] += product
    return tuple(dense), tuple(sparse)


def combine_whole_model_accounting(
    *,
    linear_dense_operations: int,
    linear_dense_q4_bytes: int,
    attention_rows: Sequence[AttentionProbabilityAccounting],
) -> dict[str, int | float]:
    """Combine unchanged linear work with candidate attention sparsity."""

    if linear_dense_operations < 0 or linear_dense_q4_bytes < 0:
        raise AttentionProbabilitySparsityError("linear accounting cannot be negative")
    rows = tuple(attention_rows)
    if not rows:
        raise AttentionProbabilitySparsityError("attention accounting is empty")
    dense_attention_ops = sum(row.dense_attention_operation_terms for row in rows)
    sparse_attention_ops = sum(row.sparse_attention_operation_terms for row in rows)
    dense_attention_bytes = sum(row.dense_attention_bytes for row in rows)
    sparse_attention_bytes = sum(
        row.sparse_attention_bytes + row.metadata_bytes for row in rows
    )
    dense_ops = linear_dense_operations + dense_attention_ops
    sparse_ops = linear_dense_operations + sparse_attention_ops
    dense_bytes = linear_dense_q4_bytes + dense_attention_bytes
    sparse_bytes = linear_dense_q4_bytes + sparse_attention_bytes
    return {
        "linear_dense_operations": linear_dense_operations,
        "linear_dense_q4_bytes": linear_dense_q4_bytes,
        "dense_attention_operations": dense_attention_ops,
        "sparse_attention_operations": sparse_attention_ops,
        "dense_attention_bytes": dense_attention_bytes,
        "sparse_attention_bytes": sparse_attention_bytes,
        "dense_whole_model_operations": dense_ops,
        "sparse_whole_model_operations": sparse_ops,
        "whole_model_operation_fraction": sparse_ops / dense_ops,
        "dense_whole_model_bytes": dense_bytes,
        "sparse_whole_model_bytes": sparse_bytes,
        "whole_model_query_byte_fraction": sparse_bytes / dense_bytes,
    }
