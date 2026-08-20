from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F


class SuffixActionInvariantError(RuntimeError):
    """Fail-closed violation of the frozen suffix-action program contract."""


@dataclass(frozen=True)
class Q4HeadManifest:
    rows: int
    cols: int
    minimum: int
    maximum: int
    packed_weight_bytes: int
    scale_bytes: int
    artifact_bytes: int
    quantized_sha256: str
    scale_sha256: str
    dequantized_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RowwiseSymmetricQ4Head:
    quantized: torch.Tensor
    scales_bf16: torch.Tensor
    dequantized_weight_bf16: torch.Tensor
    manifest: Q4HeadManifest

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.manifest.cols:
            raise SuffixActionInvariantError(
                f"head input width {hidden.shape[-1]} != {self.manifest.cols}"
            )
        return F.linear(
            hidden.to(torch.bfloat16), self.dequantized_weight_bf16
        ).to(torch.float32)


@dataclass(frozen=True)
class PredictorSelection:
    mode: str
    minimum_accepted_length: int
    total_accepted_length: int
    rank_p95: float
    rank_max: int
    mode_order: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HotProjection:
    layer_parameter_count: int
    layer_bf16_bytes: int
    q4_head_weight_bytes: int
    q4_head_scale_bytes: int
    suffix_history_bytes: int
    hidden_history_bytes: int
    first_layer_kv_bytes: int
    final_norm_bytes: int
    total_hot_bytes: int
    total_hot_gib: float
    hot_limit_bytes: int
    margin_bytes: int
    hot_limit_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(tensor)).hexdigest()


def bf16_roundtrip(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(torch.bfloat16).to(torch.float32)


def quantize_rowwise_symmetric_int4(
    weight: torch.Tensor,
    *,
    minimum: int = -8,
    maximum: int = 7,
) -> RowwiseSymmetricQ4Head:
    if weight.ndim != 2 or weight.numel() == 0:
        raise SuffixActionInvariantError("Q4 head source must be a nonempty matrix")
    if int(minimum) != -8 or int(maximum) != 7:
        raise SuffixActionInvariantError("frozen Q4 range must be [-8, 7]")

    source = weight.detach().contiguous().cpu().to(torch.float32)
    max_abs = source.abs().amax(dim=1)
    scales = max_abs / float(maximum)
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    scales_bf16 = scales.to(torch.bfloat16).contiguous()
    effective_scales = scales_bf16.to(torch.float32)
    quantized = torch.round(source / effective_scales[:, None]).clamp(
        int(minimum), int(maximum)
    ).to(torch.int8).contiguous()
    dequantized = (
        quantized.to(torch.float32) * effective_scales[:, None]
    ).to(torch.bfloat16).contiguous()

    rows, cols = map(int, source.shape)
    packed_weight_bytes = (rows * cols + 1) // 2
    scale_bytes = rows * 2
    manifest = Q4HeadManifest(
        rows=rows,
        cols=cols,
        minimum=int(minimum),
        maximum=int(maximum),
        packed_weight_bytes=packed_weight_bytes,
        scale_bytes=scale_bytes,
        artifact_bytes=packed_weight_bytes + scale_bytes,
        quantized_sha256=tensor_sha256(quantized),
        scale_sha256=tensor_sha256(scales_bf16),
        dequantized_sha256=tensor_sha256(dequantized),
    )
    return RowwiseSymmetricQ4Head(
        quantized=quantized,
        scales_bf16=scales_bf16,
        dequantized_weight_bf16=dequantized,
        manifest=manifest,
    )


def stable_argmax(scores: torch.Tensor) -> int:
    values = scores.detach().to(torch.float32).reshape(-1)
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise SuffixActionInvariantError("scores must be finite and nonempty")
    # torch.argmax returns the first maximum, which is the frozen token-ID tie rule.
    return int(torch.argmax(values).item())


def stable_token_rank(scores: torch.Tensor, target_token: int) -> int:
    values = scores.detach().to(torch.float32).reshape(-1)
    token = int(target_token)
    if token < 0 or token >= values.numel():
        raise SuffixActionInvariantError("target token outside score vector")
    if not torch.isfinite(values).all():
        raise SuffixActionInvariantError("scores contain nonfinite values")
    target = values[token]
    token_ids = torch.arange(values.numel(), dtype=torch.int64)
    better = values > target
    tied_before = (values == target) & (token_ids < token)
    return 1 + int(better.sum().item()) + int(tied_before.sum().item())


def longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for lhs, rhs in zip(left, right):
        if int(lhs) != int(rhs):
            break
        length += 1
    return length


def _validate_history(
    history_hidden: torch.Tensor, history_residual: torch.Tensor
) -> None:
    if history_hidden.ndim != 2 or history_residual.ndim != 2:
        raise SuffixActionInvariantError("history tensors must be matrices")
    if history_hidden.shape[0] != history_residual.shape[0]:
        raise SuffixActionInvariantError("history row counts differ")
    if history_hidden.shape[0] == 0:
        raise SuffixActionInvariantError("verified history is empty")
    if not torch.isfinite(history_hidden.float()).all():
        raise SuffixActionInvariantError("history hidden contains nonfinite values")
    if not torch.isfinite(history_residual.float()).all():
        raise SuffixActionInvariantError("history residual contains nonfinite values")


def predict_suffix_action(
    mode: str,
    *,
    step_offset: int,
    query_hidden: torch.Tensor,
    history_hidden: torch.Tensor,
    history_residual: torch.Tensor,
) -> tuple[torch.Tensor, int | None]:
    _validate_history(history_hidden, history_residual)
    if int(step_offset) < 0:
        raise SuffixActionInvariantError("step offset must be nonnegative")
    query = query_hidden.detach().to(torch.float32).reshape(-1)
    if query.numel() != history_hidden.shape[1]:
        raise SuffixActionInvariantError("query/history hidden widths differ")

    vocab = int(history_residual.shape[1])
    if mode == "raw_q4":
        return torch.zeros(vocab, dtype=torch.float32), None
    if mode == "carry_bf16":
        return history_residual[-1].to(torch.float32), int(history_residual.shape[0] - 1)
    if mode == "secant_bf16":
        if history_residual.shape[0] < 2:
            return history_residual[-1].to(torch.float32), int(history_residual.shape[0] - 1)
        last = history_residual[-1].to(torch.float32)
        previous = history_residual[-2].to(torch.float32)
        predicted = last + float(int(step_offset) + 1) * (last - previous)
        return bf16_roundtrip(predicted), int(history_residual.shape[0] - 1)
    if mode == "nearest_history_bf16":
        keys = history_hidden.to(torch.float32)
        query_norm = torch.linalg.vector_norm(query)
        key_norms = torch.linalg.vector_norm(keys, dim=1)
        denominator = key_norms * query_norm
        similarities = torch.where(
            denominator > 0,
            (keys @ query) / denominator,
            torch.full_like(denominator, -torch.inf),
        )
        if not torch.isfinite(similarities).any():
            index = int(keys.shape[0] - 1)
        else:
            index = int(torch.argmax(similarities).item())
        return history_residual[index].to(torch.float32), index
    raise SuffixActionInvariantError(f"unsupported predictor mode: {mode}")


def summarize_ranks(ranks: Sequence[int]) -> dict[str, Any]:
    if not ranks:
        raise SuffixActionInvariantError("rank population is empty")
    values = np.asarray([int(value) for value in ranks], dtype=np.float64)
    return {
        "count": int(values.size),
        "p50": float(np.percentile(values, 50, method="linear")),
        "p95": float(np.percentile(values, 95, method="linear")),
        "max": int(values.max()),
        "top1_fraction": float(np.mean(values == 1)),
        "top4_fraction": float(np.mean(values <= 4)),
        "top16_fraction": float(np.mean(values <= 16)),
    }


def select_predictor_mode(
    rows: Sequence[Mapping[str, Any]], mode_order: Sequence[str]
) -> PredictorSelection:
    if not rows or not mode_order:
        raise SuffixActionInvariantError("selection population/order is empty")
    order = {str(mode): index for index, mode in enumerate(mode_order)}
    seen = {str(row["mode"]) for row in rows}
    missing = [mode for mode in mode_order if mode not in seen]
    if missing:
        raise SuffixActionInvariantError(f"missing build rows for modes: {missing}")

    candidates: list[PredictorSelection] = []
    for mode in mode_order:
        selected = [row for row in rows if str(row["mode"]) == str(mode)]
        accepted = [int(row["accepted_length"]) for row in selected]
        ranks: list[int] = []
        for row in selected:
            ranks.extend(int(value) for value in row["true_path_ranks"])
        rank_summary = summarize_ranks(ranks)
        candidates.append(
            PredictorSelection(
                mode=str(mode),
                minimum_accepted_length=min(accepted),
                total_accepted_length=sum(accepted),
                rank_p95=float(rank_summary["p95"]),
                rank_max=int(rank_summary["max"]),
                mode_order=int(order[str(mode)]),
            )
        )

    # Maximize acceptance, then minimize ranks, then preserve frozen order.
    return sorted(
        candidates,
        key=lambda item: (
            -item.minimum_accepted_length,
            -item.total_accepted_length,
            item.rank_p95,
            item.rank_max,
            item.mode_order,
        ),
    )[0]


def project_target_hot_state(target: Mapping[str, Any]) -> HotProjection:
    hidden = int(target["hidden_size"])
    intermediate = int(target["intermediate_size"])
    heads = int(target["num_attention_heads"])
    kv_heads = int(target["num_key_value_heads"])
    head_dim = int(target["head_dim"])
    vocab = int(target["vocab_size"])
    max_context = int(target["max_context"])
    history = int(target["history_entries"])
    hot_limit = int(target["hot_limit_bytes"])
    if heads * head_dim != hidden:
        raise SuffixActionInvariantError("target attention geometry is inconsistent")
    if min(hidden, intermediate, kv_heads, vocab, max_context, history, hot_limit) <= 0:
        raise SuffixActionInvariantError("target projection contains nonpositive values")

    kv_width = kv_heads * head_dim
    layer_parameters = (
        2 * hidden * hidden
        + 2 * hidden * kv_width
        + 3 * hidden * intermediate
    )
    layer_bytes = layer_parameters * 2
    q4_weight_bytes = (vocab * hidden + 1) // 2
    q4_scale_bytes = vocab * 2
    suffix_history_bytes = history * vocab * 2
    hidden_history_bytes = history * hidden * 2
    kv_bytes = max_context * kv_heads * head_dim * 2 * 2
    norm_bytes = hidden * 2
    total = (
        layer_bytes
        + q4_weight_bytes
        + q4_scale_bytes
        + suffix_history_bytes
        + hidden_history_bytes
        + kv_bytes
        + norm_bytes
    )
    return HotProjection(
        layer_parameter_count=layer_parameters,
        layer_bf16_bytes=layer_bytes,
        q4_head_weight_bytes=q4_weight_bytes,
        q4_head_scale_bytes=q4_scale_bytes,
        suffix_history_bytes=suffix_history_bytes,
        hidden_history_bytes=hidden_history_bytes,
        first_layer_kv_bytes=kv_bytes,
        final_norm_bytes=norm_bytes,
        total_hot_bytes=total,
        total_hot_gib=total / float(1 << 30),
        hot_limit_bytes=hot_limit,
        margin_bytes=hot_limit - total,
        hot_limit_passed=total <= hot_limit,
    )


def verification_weight_fraction(
    accepted_length: int, *, compression_ratio: float = 1.0
) -> float:
    length = int(accepted_length)
    ratio = float(compression_ratio)
    if length <= 0 or not math.isfinite(ratio) or ratio < 1.0:
        return math.inf
    return 1.0 / (length * ratio)


def synthetic_carry_control() -> dict[str, Any]:
    history_hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
    history_residual = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, -1.0, 0.5]], dtype=torch.bfloat16
    )
    correction, index = predict_suffix_action(
        "carry_bf16",
        step_offset=0,
        query_hidden=torch.tensor([1.0, 1.0]),
        history_hidden=history_hidden,
        history_residual=history_residual,
    )
    expected = history_residual[-1].to(torch.float32)
    return {
        "passed": tensor_bytes(correction) == tensor_bytes(expected),
        "history_index": index,
        "correction_sha256": tensor_sha256(correction),
    }


def synthetic_nearest_control() -> dict[str, Any]:
    history_hidden = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.bfloat16
    )
    history_residual = torch.tensor(
        [[3.0, 0.0], [0.0, 4.0], [-2.0, 0.0]], dtype=torch.bfloat16
    )
    correction, index = predict_suffix_action(
        "nearest_history_bf16",
        step_offset=0,
        query_hidden=torch.tensor([0.0, 2.0]),
        history_hidden=history_hidden,
        history_residual=history_residual,
    )
    expected = history_residual[1].to(torch.float32)
    return {
        "passed": index == 1 and tensor_bytes(correction) == tensor_bytes(expected),
        "history_index": index,
        "correction_sha256": tensor_sha256(correction),
    }
