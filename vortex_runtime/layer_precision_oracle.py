from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable, Mapping

from torch import nn

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class LayerPrecisionEffect:
    group: str
    module_names: tuple[str, ...]
    unique_elements: int
    residual_bytes: float
    layer_fraction: float
    corrected_base_errors: int
    introduced_errors: int
    exact_top1_tokens: int
    total_tokens: int
    exact_top1_rate: float
    exact_gap_reduction: float
    score_per_residual_byte: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def precision_module_groups(
    model: nn.Module,
    *,
    layers_per_group: int = 2,
) -> dict[str, tuple[str, ...]]:
    """Group Linear/Embedding modules into contiguous decoder-layer spans.

    Non-layer modules, including embeddings and LM heads, are assigned to the
    `io` group. The function is architecture-tolerant for models exposing
    decoder blocks under a conventional `.layers.N.` qualified name.
    """

    if layers_per_group <= 0:
        raise ValueError("layers_per_group must be positive")
    grouped: dict[str, list[str]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        match = _LAYER_PATTERN.search(name)
        if match is None:
            group = "io"
        else:
            layer = int(match.group(1))
            start = (layer // layers_per_group) * layers_per_group
            end = start + layers_per_group - 1
            group = f"layers_{start:03d}_{end:03d}"
        grouped.setdefault(group, []).append(name)
    if not grouped:
        raise RuntimeError("no Linear/Embedding precision modules found")
    return {
        group: tuple(sorted(names))
        for group, names in sorted(grouped.items())
    }


def unique_weight_elements(
    model: nn.Module,
    module_names: Iterable[str] | None = None,
) -> int:
    modules = dict(model.named_modules())
    selected = set(module_names) if module_names is not None else None
    seen: set[tuple[int, int]] = set()
    total = 0
    for name, module in modules.items():
        if selected is not None and name not in selected:
            continue
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        weight = module.weight
        identity = (weight.untyped_storage().data_ptr(), weight.storage_offset())
        if identity in seen:
            continue
        seen.add(identity)
        total += weight.numel()
    return total


def rank_precision_effects(
    effects: Iterable[LayerPrecisionEffect],
) -> list[LayerPrecisionEffect]:
    materialized = list(effects)
    if not materialized:
        raise ValueError("at least one precision effect is required")
    return sorted(
        materialized,
        key=lambda effect: (
            effect.corrected_base_errors - effect.introduced_errors,
            effect.exact_gap_reduction,
            effect.score_per_residual_byte,
            -effect.residual_bytes,
        ),
        reverse=True,
    )


def build_layer_precision_effect(
    *,
    group: str,
    module_names: Iterable[str],
    unique_elements: int,
    total_unique_elements: int,
    residual_bits: int,
    corrected_base_errors: int,
    introduced_errors: int,
    exact_top1_tokens: int,
    total_tokens: int,
    exact_gap_reduction: float,
) -> LayerPrecisionEffect:
    if unique_elements <= 0 or total_unique_elements <= 0:
        raise ValueError("precision element counts must be positive")
    if unique_elements > total_unique_elements:
        raise ValueError("group elements exceed total elements")
    if residual_bits <= 0 or total_tokens <= 0:
        raise ValueError("residual bits and total tokens must be positive")
    residual_bytes = unique_elements * residual_bits / 8
    net_corrections = corrected_base_errors - introduced_errors
    score = (
        max(0.0, float(net_corrections)) + max(0.0, exact_gap_reduction)
    ) / residual_bytes
    return LayerPrecisionEffect(
        group=group,
        module_names=tuple(module_names),
        unique_elements=unique_elements,
        residual_bytes=residual_bytes,
        layer_fraction=unique_elements / total_unique_elements,
        corrected_base_errors=corrected_base_errors,
        introduced_errors=introduced_errors,
        exact_top1_tokens=exact_top1_tokens,
        total_tokens=total_tokens,
        exact_top1_rate=exact_top1_tokens / total_tokens,
        exact_gap_reduction=exact_gap_reduction,
        score_per_residual_byte=score,
    )
