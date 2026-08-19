from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch


class CrossLayerTemplateError(RuntimeError):
    """Fail-closed invariant violation for exact cross-layer generators."""


def _words(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().contiguous().cpu()
    if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
        raise CrossLayerTemplateError("expected a 2-D BF16 matrix")
    if not torch.isfinite(tensor.float()).all():
        raise CrossLayerTemplateError("nonfinite checkpoint weight")
    signed = tensor.view(torch.int16).to(torch.int32)
    return torch.bitwise_and(signed, 0xFFFF).numpy().astype(np.uint16)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().contiguous().cpu()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _entropy_bits_per_symbol(array: np.ndarray) -> float:
    values = np.asarray(array).reshape(-1)
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(np.float64) / float(values.size)
    return float(-(probabilities * np.log2(probabilities)).sum())


def _stream_bits(array: np.ndarray) -> float:
    values = np.asarray(array)
    return _entropy_bits_per_symbol(values) * float(values.size)


def _percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise CrossLayerTemplateError("empty percentile population")
    position = (len(ordered) - 1) * float(q)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _bit_majority(stack: np.ndarray) -> np.ndarray:
    values = np.asarray(stack, dtype=np.uint16)
    if values.ndim != 3 or values.shape[0] < 1:
        raise CrossLayerTemplateError("invalid majority stack")
    threshold = (values.shape[0] + 1) // 2
    template = np.zeros(values.shape[1:], dtype=np.uint16)
    for bit in range(16):
        count = ((values >> np.uint16(bit)) & np.uint16(1)).sum(axis=0)
        template |= ((count >= threshold).astype(np.uint16) << np.uint16(bit))
    return template


def _mod_sub(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return ((left.astype(np.uint32) - right.astype(np.uint32)) & 0xFFFF).astype(np.uint16)


def _mod_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return ((left.astype(np.uint32) + right.astype(np.uint32)) & 0xFFFF).astype(np.uint16)


@dataclass(frozen=True)
class ExactLayerCode:
    name: str
    baseline_bits: float
    total_bits: float
    information_fraction: float
    effective_layer_fraction_p05: float
    effective_layer_fraction_p50: float
    effective_layer_fraction_p95: float
    maximum_effective_layer_fraction: float
    template_or_anchor_bits: float
    residual_bits: float
    reconstruction_exact: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate(
    *,
    name: str,
    original: np.ndarray,
    reconstructed: np.ndarray,
    baseline_layer_bits: Sequence[float],
    template_or_anchor_bits: float,
    residual_streams: Sequence[np.ndarray | None],
    metadata: dict[str, Any],
) -> ExactLayerCode:
    if original.shape != reconstructed.shape:
        raise CrossLayerTemplateError("reconstruction shape mismatch")
    exact = bool(np.array_equal(original, reconstructed))
    layer_count = int(original.shape[0])
    if len(residual_streams) != layer_count or len(baseline_layer_bits) != layer_count:
        raise CrossLayerTemplateError("per-layer accounting mismatch")
    residual_layer_bits = [
        0.0 if residual is None else _stream_bits(residual)
        for residual in residual_streams
    ]
    residual_bits = float(sum(residual_layer_bits))
    overhead_per_layer = float(template_or_anchor_bits) / layer_count
    effective_layer_bits = [overhead_per_layer + value for value in residual_layer_bits]
    effective_fractions = [
        effective_layer_bits[index] / max(float(baseline_layer_bits[index]), 1e-300)
        for index in range(layer_count)
    ]
    baseline_bits = float(sum(baseline_layer_bits))
    total_bits = float(template_or_anchor_bits) + residual_bits
    return ExactLayerCode(
        name=name,
        baseline_bits=baseline_bits,
        total_bits=total_bits,
        information_fraction=total_bits / max(baseline_bits, 1e-300),
        effective_layer_fraction_p05=_percentile(effective_fractions, 0.05),
        effective_layer_fraction_p50=_percentile(effective_fractions, 0.50),
        effective_layer_fraction_p95=_percentile(effective_fractions, 0.95),
        maximum_effective_layer_fraction=max(effective_fractions),
        template_or_anchor_bits=float(template_or_anchor_bits),
        residual_bits=residual_bits,
        reconstruction_exact=exact,
        metadata=dict(metadata),
    )


def _audit_role(role: str, tensors: Sequence[torch.Tensor], template_groups: int) -> dict[str, Any]:
    if len(tensors) < 3:
        raise CrossLayerTemplateError("at least three layers are required")
    arrays = [_words(tensor) for tensor in tensors]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise CrossLayerTemplateError("cross-layer shape mismatch")
    stack = np.stack(arrays, axis=0)
    layer_count, rows, columns = stack.shape
    symbols_per_layer = rows * columns
    baseline_layer_bits = [_stream_bits(stack[index]) for index in range(layer_count)]
    candidates: list[ExactLayerCode] = []

    residual = [None] + [np.bitwise_xor(stack[index], stack[index - 1]) for index in range(1, layer_count)]
    reconstructed = np.empty_like(stack)
    reconstructed[0] = stack[0]
    for index in range(1, layer_count):
        reconstructed[index] = np.bitwise_xor(reconstructed[index - 1], residual[index])
    candidates.append(_candidate(
        name="previous_layer_xor",
        original=stack,
        reconstructed=reconstructed,
        baseline_layer_bits=baseline_layer_bits,
        template_or_anchor_bits=_stream_bits(stack[0]),
        residual_streams=residual,
        metadata={"anchor_layers": 1},
    ))

    residual = [None] + [_mod_sub(stack[index], stack[index - 1]) for index in range(1, layer_count)]
    reconstructed = np.empty_like(stack)
    reconstructed[0] = stack[0]
    for index in range(1, layer_count):
        reconstructed[index] = _mod_add(reconstructed[index - 1], residual[index])
    candidates.append(_candidate(
        name="previous_layer_modular_delta",
        original=stack,
        reconstructed=reconstructed,
        baseline_layer_bits=baseline_layer_bits,
        template_or_anchor_bits=_stream_bits(stack[0]),
        residual_streams=residual,
        metadata={"anchor_layers": 1, "modulus": 65536},
    ))

    residual = [None, None] + [
        np.bitwise_xor(np.bitwise_xor(stack[index], stack[index - 1]), stack[index - 2])
        for index in range(2, layer_count)
    ]
    reconstructed = np.empty_like(stack)
    reconstructed[:2] = stack[:2]
    for index in range(2, layer_count):
        reconstructed[index] = np.bitwise_xor(
            np.bitwise_xor(reconstructed[index - 1], reconstructed[index - 2]),
            residual[index],
        )
    candidates.append(_candidate(
        name="second_order_xor_recurrence",
        original=stack,
        reconstructed=reconstructed,
        baseline_layer_bits=baseline_layer_bits,
        template_or_anchor_bits=_stream_bits(stack[0]) + _stream_bits(stack[1]),
        residual_streams=residual,
        metadata={"anchor_layers": 2},
    ))

    residual = [None, None]
    for index in range(2, layer_count):
        prediction = (
            2 * stack[index - 1].astype(np.uint32)
            - stack[index - 2].astype(np.uint32)
        ) & 0xFFFF
        residual.append(_mod_sub(stack[index], prediction.astype(np.uint16)))
    reconstructed = np.empty_like(stack)
    reconstructed[:2] = stack[:2]
    for index in range(2, layer_count):
        prediction = (
            2 * reconstructed[index - 1].astype(np.uint32)
            - reconstructed[index - 2].astype(np.uint32)
        ) & 0xFFFF
        reconstructed[index] = _mod_add(prediction.astype(np.uint16), residual[index])
    candidates.append(_candidate(
        name="second_order_modular_extrapolation",
        original=stack,
        reconstructed=reconstructed,
        baseline_layer_bits=baseline_layer_bits,
        template_or_anchor_bits=_stream_bits(stack[0]) + _stream_bits(stack[1]),
        residual_streams=residual,
        metadata={"anchor_layers": 2, "modulus": 65536},
    ))

    template = _bit_majority(stack)
    residual = [np.bitwise_xor(stack[index], template) for index in range(layer_count)]
    reconstructed = np.stack(
        [np.bitwise_xor(template, residual[index]) for index in range(layer_count)],
        axis=0,
    )
    candidates.append(_candidate(
        name="global_bit_majority_template",
        original=stack,
        reconstructed=reconstructed,
        baseline_layer_bits=baseline_layer_bits,
        template_or_anchor_bits=_stream_bits(template),
        residual_streams=residual,
        metadata={"template_count": 1},
    ))

    group_count = int(template_groups)
    if not 1 < group_count <= layer_count:
        raise CrossLayerTemplateError("invalid frozen template group count")
    groups = [np.asarray(group, dtype=np.int64) for group in np.array_split(np.arange(layer_count), group_count)]
    templates = [_bit_majority(stack[group]) for group in groups]
    residual_streams: list[np.ndarray | None] = [None] * layer_count
    reconstructed = np.empty_like(stack)
    template_bits = 0.0
    group_sizes = []
    for group, template in zip(groups, templates):
        template_bits += _stream_bits(template)
        group_sizes.append(int(group.size))
        for index in group.tolist():
            residual_streams[index] = np.bitwise_xor(stack[index], template)
            reconstructed[index] = np.bitwise_xor(template, residual_streams[index])
    candidates.append(_candidate(
        name=f"contiguous_{group_count}_template_generator",
        original=stack,
        reconstructed=reconstructed,
        baseline_layer_bits=baseline_layer_bits,
        template_or_anchor_bits=template_bits,
        residual_streams=residual_streams,
        metadata={"template_count": group_count, "group_sizes": group_sizes},
    ))

    if not all(candidate.reconstruction_exact for candidate in candidates):
        raise CrossLayerTemplateError(f"{role} exact reconstruction control failed")
    best = min(candidates, key=lambda candidate: (candidate.information_fraction, candidate.name))
    return {
        "role": role,
        "layer_count": layer_count,
        "shape": [rows, columns],
        "symbols_per_layer": symbols_per_layer,
        "baseline_bits": float(sum(baseline_layer_bits)),
        "baseline_layer_bits": [float(value) for value in baseline_layer_bits],
        "weight_sha256": [_tensor_sha256(tensor) for tensor in tensors],
        "candidates": [candidate.to_dict() for candidate in candidates],
        "best_candidate": best.to_dict(),
    }


def audit_cross_layer_templates(
    *,
    gate_weights: Sequence[torch.Tensor],
    up_weights: Sequence[torch.Tensor],
    down_weights_transposed: Sequence[torch.Tensor],
    template_groups: int,
) -> dict[str, Any]:
    counts = {len(gate_weights), len(up_weights), len(down_weights_transposed)}
    if len(counts) != 1:
        raise CrossLayerTemplateError("role layer-count mismatch")
    role_rows = [
        _audit_role("gate", gate_weights, template_groups),
        _audit_role("up", up_weights, template_groups),
        _audit_role("down_transposed", down_weights_transposed, template_groups),
    ]
    layer_count = role_rows[0]["layer_count"]
    baseline_total = sum(float(row["baseline_bits"]) for row in role_rows)
    total_bits = sum(float(row["best_candidate"]["total_bits"]) for row in role_rows)
    combined_layer_bits = [0.0] * layer_count
    combined_baseline_layer_bits = [0.0] * layer_count
    for row in role_rows:
        best = row["best_candidate"]
        equal_layer_bits = float(best["total_bits"]) / layer_count
        for index in range(layer_count):
            combined_layer_bits[index] += equal_layer_bits
            combined_baseline_layer_bits[index] += float(row["baseline_layer_bits"][index])
    effective_fractions = [
        combined_layer_bits[index] / max(combined_baseline_layer_bits[index], 1e-300)
        for index in range(layer_count)
    ]
    aggregate = {
        "role_count": len(role_rows),
        "layer_count": layer_count,
        "role_wise_best_candidate_names": {
            row["role"]: row["best_candidate"]["name"] for row in role_rows
        },
        "total_information_fraction": total_bits / max(baseline_total, 1e-300),
        "effective_layer_fraction_p05": _percentile(effective_fractions, 0.05),
        "effective_layer_fraction_p50": _percentile(effective_fractions, 0.50),
        "effective_layer_fraction_p95": _percentile(effective_fractions, 0.95),
        "maximum_effective_layer_fraction": max(effective_fractions),
        "all_reconstructions_exact": all(
            candidate["reconstruction_exact"]
            for row in role_rows
            for candidate in row["candidates"]
        ),
    }
    return {"role_rows": role_rows, "aggregate": aggregate}
