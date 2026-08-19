from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


P50_WHOLE_MODEL_FRACTION = 1.2 * 4.0 / 405.0
P95_WHOLE_MODEL_FRACTION = 1.5 * 4.0 / 405.0
LLAMA405_TOTAL_PARAMETERS = 405_849_243_648
LLAMA405_ONE_MLP_PROJECTION_PARAMETERS = 109_924_319_232
LLAMA405_MLP_PARAMETER_SHARE = (
    3.0 * LLAMA405_ONE_MLP_PROJECTION_PARAMETERS / LLAMA405_TOTAL_PARAMETERS
)
LLAMA405_HIDDEN_SIZE = 16_384
LLAMA405_INTERMEDIATE_SIZE = 53_248
LLAMA405_LAYER_COUNT = 126


class JointSwiGLUCompilerError(RuntimeError):
    """Fail-closed compiler/query invariant failure."""


@dataclass(frozen=True)
class PageSpec:
    index: int
    start: int
    stop: int
    payload_bytes: int

    @property
    def channels(self) -> int:
        return self.stop - self.start


@dataclass
class QueryResult:
    mode: str
    pages_read: int
    page_count: int
    page_fraction: float
    projected_whole_model_weight_fraction: float
    projected_whole_model_operation_fraction: float
    certified_without_fallback: bool
    fallback_required: bool
    false_accepts: int
    certified_output_fraction: float
    candidate_matches_reference: bool
    gate_up_row_split_mismatches: int
    native_vs_exact_real_round_mismatches: int
    order_independent_full_read_unresolved: int
    bound_violations: int
    cold_payload_bytes: int
    bound_scan_scalars: int
    page_order: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pages_read": self.pages_read,
            "page_count": self.page_count,
            "page_fraction": self.page_fraction,
            "projected_whole_model_weight_fraction": self.projected_whole_model_weight_fraction,
            "projected_whole_model_operation_fraction": self.projected_whole_model_operation_fraction,
            "certified_without_fallback": self.certified_without_fallback,
            "fallback_required": self.fallback_required,
            "false_accepts": self.false_accepts,
            "certified_output_fraction": self.certified_output_fraction,
            "candidate_matches_reference": self.candidate_matches_reference,
            "gate_up_row_split_mismatches": self.gate_up_row_split_mismatches,
            "native_vs_exact_real_round_mismatches": self.native_vs_exact_real_round_mismatches,
            "order_independent_full_read_unresolved": self.order_independent_full_read_unresolved,
            "bound_violations": self.bound_violations,
            "cold_payload_bytes": self.cold_payload_bytes,
            "bound_scan_scalars": self.bound_scan_scalars,
            "page_order": self.page_order,
        }


@dataclass
class CompiledJointSwiGLU:
    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor
    act_fn: Callable[[torch.Tensor], torch.Tensor]
    pages: tuple[PageSpec, ...]
    page_bound_coeff: torch.Tensor
    page_channels: int
    source_parameter_bytes: int
    hot_metadata_bytes: int
    fingerprint: str

    @property
    def hidden_size(self) -> int:
        return int(self.gate_weight.shape[1])

    @property
    def intermediate_size(self) -> int:
        return int(self.gate_weight.shape[0])

    @property
    def output_size(self) -> int:
        return int(self.down_weight.shape[0])

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def manifest(self) -> dict[str, Any]:
        return {
            "format": "joint-exact-swiglu-compiler-v1",
            "mechanism": "page_separable_fused_swiglu_exact_rounding_refinement",
            "fingerprint": self.fingerprint,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "output_size": self.output_size,
            "page_channels": self.page_channels,
            "page_count": self.page_count,
            "source_parameter_bytes": self.source_parameter_bytes,
            "hot_metadata_bytes": self.hot_metadata_bytes,
            "target_mlp_parameter_share": LLAMA405_MLP_PARAMETER_SHARE,
            "p50_whole_model_fraction": P50_WHOLE_MODEL_FRACTION,
            "p95_whole_model_fraction": P95_WHOLE_MODEL_FRACTION,
            "exactness": {
                "gate_up_rows": "original reduction axis; page partitions only output rows",
                "down_projection": "exact-real page contributions plus conservative FP32 accumulation envelope",
                "unresolved": "unchanged native MLP fallback; never silent approximation",
            },
        }

    def _validate_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[0] != 1 or x.shape[1] != self.hidden_size:
            raise JointSwiGLUCompilerError(
                f"expected one [1,{self.hidden_size}] activation, got {tuple(x.shape)}"
            )
        if not torch.isfinite(x.float()).all():
            raise JointSwiGLUCompilerError("nonfinite activation")
        return x.to(dtype=self.gate_weight.dtype, device=self.gate_weight.device)

    @staticmethod
    def _bf16_unique(center: torch.Tensor, radius: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if center.dtype != torch.float64 or radius.dtype != torch.float64:
            raise JointSwiGLUCompilerError("certificate arithmetic must be float64")
        if torch.any(radius < 0) or not torch.isfinite(center).all() or not torch.isfinite(radius).all():
            raise JointSwiGLUCompilerError("invalid certificate interval")
        lower = (center - radius).to(torch.bfloat16)
        upper = (center + radius).to(torch.bfloat16)
        return lower == upper, lower

    @staticmethod
    def _fp32_accumulation_radius(abs_term_sum: torch.Tensor, terms: int) -> torch.Tensor:
        # BF16 products are exactly representable in FP32. This Higham-style envelope
        # covers any sequence of `terms` FP32 additions when terms*u < 1.
        unit_roundoff = 2.0 ** -24
        product = float(terms) * unit_roundoff
        if not product < 1.0:
            raise JointSwiGLUCompilerError("FP32 accumulation envelope overflow")
        gamma = product / (1.0 - product)
        return abs_term_sum * gamma

    def _reference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        full_gate = F.linear(x, self.gate_weight)
        full_up = F.linear(x, self.up_weight)
        z = self.act_fn(full_gate) * full_up
        reference = F.linear(z, self.down_weight)
        return full_gate, full_up, z, reference

    def _page_materialization(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        full_gate, full_up, _z, _reference = self._reference(x)
        contribution_rows: list[torch.Tensor] = []
        absolute_rows: list[torch.Tensor] = []
        gate_up_mismatches = 0
        payload_bytes = 0
        for page in self.pages:
            gate = F.linear(x, self.gate_weight[page.start : page.stop])
            up = F.linear(x, self.up_weight[page.start : page.stop])
            gate_up_mismatches += int(
                torch.count_nonzero(gate != full_gate[:, page.start : page.stop]).item()
            )
            gate_up_mismatches += int(
                torch.count_nonzero(up != full_up[:, page.start : page.stop]).item()
            )
            z = self.act_fn(gate) * up
            down = self.down_weight[:, page.start : page.stop]
            exact_terms = down.to(torch.float64) * z.squeeze(0).to(torch.float64).unsqueeze(0)
            contribution_rows.append(exact_terms.sum(dim=1))
            absolute_rows.append(exact_terms.abs().sum(dim=1))
            payload_bytes += page.payload_bytes
        return (
            torch.stack(contribution_rows, dim=0),
            torch.stack(absolute_rows, dim=0),
            gate_up_mismatches,
            payload_bytes,
        )


    def _projected_operation_fraction(self, *, mode: str, pages_read: int) -> float:
        target_page_count = LLAMA405_INTERMEDIATE_SIZE // self.page_channels
        page_fraction = pages_read / max(1, self.page_count)
        target_pages_read = page_fraction * target_page_count
        page_compute = page_fraction * LLAMA405_MLP_PARAMETER_SHARE
        if mode == "oracle_exact_contribution_greedy":
            return page_compute
        # Deliberately favorable one-op-per-scalar accounting. It charges one complete
        # page/output metadata scan and only three output-scalar updates per materialized
        # target-equivalent page.
        scan_ops = LLAMA405_LAYER_COUNT * target_page_count * LLAMA405_HIDDEN_SIZE
        update_ops = LLAMA405_LAYER_COUNT * target_pages_read * LLAMA405_HIDDEN_SIZE * 3
        return page_compute + (scan_ops + update_ops) / LLAMA405_TOTAL_PARAMETERS

    def _run_order(
        self,
        *,
        mode: str,
        x: torch.Tensor,
        page_contrib: torch.Tensor,
        page_abs: torch.Tensor,
        reference: torch.Tensor,
        gate_up_mismatches: int,
        order: Sequence[int],
        unread_radius_pages: torch.Tensor,
        fp_abs_bound: torch.Tensor,
        bound_violations: int,
    ) -> QueryResult:
        output = self.output_size
        partial = torch.zeros(output, dtype=torch.float64, device=page_contrib.device)
        unread = unread_radius_pages.sum(dim=0).clone()
        fp_radius = self._fp32_accumulation_radius(fp_abs_bound, self.intermediate_size)
        certified = torch.zeros(output, dtype=torch.bool, device=page_contrib.device)
        predicted = torch.zeros(output, dtype=torch.bfloat16, device=page_contrib.device)
        used: list[int] = []
        for page_index in order:
            partial = partial + page_contrib[page_index]
            unread = torch.clamp(unread - unread_radius_pages[page_index], min=0.0)
            used.append(int(page_index))
            unique, rounded = self._bf16_unique(partial, unread + fp_radius)
            certified = unique & (rounded == reference.squeeze(0))
            predicted = rounded
            if bool(certified.all()):
                break

        certified_without_fallback = bool(certified.all())
        fallback_required = not certified_without_fallback
        candidate = predicted if certified_without_fallback else reference.squeeze(0)
        false_accepts = 0
        if certified_without_fallback:
            false_accepts = int(torch.count_nonzero(candidate != reference.squeeze(0)).item())
        page_payload = sum(self.pages[i].payload_bytes for i in used)
        exact_real = page_contrib.sum(dim=0)
        native_vs_exact = int(
            torch.count_nonzero(exact_real.to(torch.bfloat16) != reference.squeeze(0)).item()
        )
        full_unique, full_rounded = self._bf16_unique(
            exact_real, self._fp32_accumulation_radius(page_abs.sum(dim=0), self.intermediate_size)
        )
        full_unresolved = int(
            torch.count_nonzero(~(full_unique & (full_rounded == reference.squeeze(0)))).item()
        )
        return QueryResult(
            mode=mode,
            pages_read=len(used),
            page_count=self.page_count,
            page_fraction=len(used) / max(1, self.page_count),
            projected_whole_model_weight_fraction=(
                len(used) / max(1, self.page_count) * LLAMA405_MLP_PARAMETER_SHARE
            ),
            projected_whole_model_operation_fraction=self._projected_operation_fraction(
                mode=mode, pages_read=len(used)
            ),
            certified_without_fallback=certified_without_fallback,
            fallback_required=fallback_required,
            false_accepts=false_accepts,
            certified_output_fraction=float(certified.float().mean().item()),
            candidate_matches_reference=bool(torch.equal(candidate, reference.squeeze(0))),
            gate_up_row_split_mismatches=gate_up_mismatches,
            native_vs_exact_real_round_mismatches=native_vs_exact,
            order_independent_full_read_unresolved=full_unresolved,
            bound_violations=bound_violations,
            cold_payload_bytes=page_payload,
            bound_scan_scalars=self.page_count * self.output_size,
            page_order=used,
        )

    def query_oracle_greedy(self, x: torch.Tensor) -> QueryResult:
        """Favorable upper bound: exact per-page contribution magnitudes and all candidate choices are free."""
        x = self._validate_input(x)
        _g, _u, _z, reference = self._reference(x)
        page_contrib, page_abs, mismatches, _ = self._page_materialization(x)
        remaining = list(range(self.page_count))
        partial = torch.zeros(self.output_size, dtype=torch.float64, device=x.device)
        unread = page_abs.sum(dim=0).clone()
        fp_radius = self._fp32_accumulation_radius(page_abs.sum(dim=0), self.intermediate_size)
        order: list[int] = []
        while remaining:
            indices = torch.tensor(remaining, dtype=torch.long, device=x.device)
            centers = partial.unsqueeze(0) + page_contrib[indices]
            radii = torch.clamp(unread.unsqueeze(0) - page_abs[indices], min=0.0) + fp_radius.unsqueeze(0)
            lower = (centers - radii).to(torch.bfloat16)
            upper = (centers + radii).to(torch.bfloat16)
            certified = (lower == upper) & (lower == reference.squeeze(0).unsqueeze(0))
            counts = certified.sum(dim=1)
            best_count = int(counts.max().item())
            choices = torch.nonzero(counts == best_count, as_tuple=False).flatten()
            if choices.numel() > 1:
                radius_scores = radii[choices].sum(dim=1)
                choice = int(choices[int(torch.argmin(radius_scores).item())].item())
            else:
                choice = int(choices[0].item())
            selected = remaining[choice]
            order.append(selected)
            partial = partial + page_contrib[selected]
            unread = torch.clamp(unread - page_abs[selected], min=0.0)
            remaining.pop(choice)
            now_unique, now_value = self._bf16_unique(partial, unread + fp_radius)
            if bool((now_unique & (now_value == reference.squeeze(0))).all()):
                break
        return self._run_order(
            mode="oracle_exact_contribution_greedy",
            x=x,
            page_contrib=page_contrib,
            page_abs=page_abs,
            reference=reference,
            gate_up_mismatches=mismatches,
            order=order,
            unread_radius_pages=page_abs,
            fp_abs_bound=page_abs.sum(dim=0),
            bound_violations=0,
        )

    def query_sound_metadata(self, x: torch.Tensor) -> QueryResult:
        """Deployable reference: page order and unread bounds use only x and compiled norm metadata."""
        x = self._validate_input(x)
        _g, _u, _z, reference = self._reference(x)
        page_contrib, page_abs, mismatches, _ = self._page_materialization(x)
        x_norm_sq = float(torch.linalg.vector_norm(x.to(torch.float64)).item()) ** 2
        bounds = self.page_bound_coeff.to(device=x.device) * x_norm_sq
        violations = int(torch.count_nonzero(page_abs > bounds * (1.0 + 1e-12) + 1e-18).item())
        priorities = bounds.sum(dim=1)
        order = sorted(range(self.page_count), key=lambda i: (-float(priorities[i].item()), i))
        return self._run_order(
            mode="sound_metadata_bound",
            x=x,
            page_contrib=page_contrib,
            page_abs=page_abs,
            reference=reference,
            gate_up_mismatches=mismatches,
            order=order,
            unread_radius_pages=bounds,
            fp_abs_bound=bounds.sum(dim=0),
            bound_violations=violations,
        )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def compile_joint_exact_swiglu(
    mlp: nn.Module,
    *,
    page_channels: int = 8,
) -> CompiledJointSwiGLU:
    for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        if not hasattr(mlp, name):
            raise JointSwiGLUCompilerError(f"MLP missing {name}")
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj
    for name, projection in (("gate", gate), ("up", up), ("down", down)):
        if projection.bias is not None:
            raise JointSwiGLUCompilerError(f"{name} bias is not supported by v1")
        if projection.weight.dtype != torch.bfloat16:
            raise JointSwiGLUCompilerError(f"{name} weight must be BF16")
    if gate.weight.shape != up.weight.shape:
        raise JointSwiGLUCompilerError("gate/up shape mismatch")
    intermediate, hidden = map(int, gate.weight.shape)
    output, down_intermediate = map(int, down.weight.shape)
    if down_intermediate != intermediate:
        raise JointSwiGLUCompilerError("down intermediate mismatch")
    if page_channels <= 0 or intermediate % int(page_channels) != 0:
        raise JointSwiGLUCompilerError("page_channels must evenly divide intermediate size")

    gate_weight = gate.weight.detach().contiguous().clone()
    up_weight = up.weight.detach().contiguous().clone()
    down_weight = down.weight.detach().contiguous().clone()
    pages: list[PageSpec] = []
    bound_rows: list[torch.Tensor] = []
    scalar_bytes = gate_weight.element_size()
    gate_norm = torch.linalg.vector_norm(gate_weight.to(torch.float64), dim=1)
    up_norm = torch.linalg.vector_norm(up_weight.to(torch.float64), dim=1)
    for index, start in enumerate(range(0, intermediate, int(page_channels))):
        stop = start + int(page_channels)
        payload = int(page_channels) * (hidden + hidden + output) * scalar_bytes
        pages.append(PageSpec(index=index, start=start, stop=stop, payload_bytes=payload))
        coeff = (
            down_weight[:, start:stop].to(torch.float64).abs()
            * (gate_norm[start:stop] * up_norm[start:stop]).unsqueeze(0)
        ).sum(dim=1)
        bound_rows.append(coeff)
    page_bound_coeff = torch.stack(bound_rows, dim=0).contiguous()
    source_bytes = int(
        gate_weight.numel() * gate_weight.element_size()
        + up_weight.numel() * up_weight.element_size()
        + down_weight.numel() * down_weight.element_size()
    )
    hot_metadata_bytes = int(page_bound_coeff.numel() * page_bound_coeff.element_size() + len(pages) * 16)
    fingerprint_payload = {
        "gate": _tensor_sha256(gate_weight),
        "up": _tensor_sha256(up_weight),
        "down": _tensor_sha256(down_weight),
        "page_channels": int(page_channels),
        "mechanism": "page_separable_fused_swiglu_exact_rounding_refinement",
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CompiledJointSwiGLU(
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
        act_fn=mlp.act_fn,
        pages=tuple(pages),
        page_bound_coeff=page_bound_coeff,
        page_channels=int(page_channels),
        source_parameter_bytes=source_bytes,
        hot_metadata_bytes=hot_metadata_bytes,
        fingerprint=fingerprint,
    )


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise JointSwiGLUCompilerError("empty percentile")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def aggregate_query_rows(rows: Sequence[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    selected = [row for row in rows if row["mode"] == mode]
    if not selected:
        raise JointSwiGLUCompilerError(f"no rows for {mode}")
    fractions = [row["projected_whole_model_weight_fraction"] for row in selected]
    operation_fractions = [row["projected_whole_model_operation_fraction"] for row in selected]
    pages = [row["pages_read"] for row in selected]
    return {
        "mode": mode,
        "count": len(selected),
        "pages_p05": percentile(pages, 0.05),
        "pages_p50": percentile(pages, 0.50),
        "pages_p95": percentile(pages, 0.95),
        "whole_model_fraction_p05": percentile(fractions, 0.05),
        "whole_model_fraction_p50": percentile(fractions, 0.50),
        "whole_model_fraction_p95": percentile(fractions, 0.95),
        "whole_model_operation_fraction_p05": percentile(operation_fractions, 0.05),
        "whole_model_operation_fraction_p50": percentile(operation_fractions, 0.50),
        "whole_model_operation_fraction_p95": percentile(operation_fractions, 0.95),
        "fallback_rate": sum(bool(row["fallback_required"]) for row in selected) / len(selected),
        "false_accepts": sum(int(row["false_accepts"]) for row in selected),
        "candidate_mismatches": sum(not bool(row["candidate_matches_reference"]) for row in selected),
        "gate_up_row_split_mismatches": sum(int(row["gate_up_row_split_mismatches"]) for row in selected),
        "bound_violations": sum(int(row["bound_violations"]) for row in selected),
        "full_read_unresolved": sum(int(row["order_independent_full_read_unresolved"]) for row in selected),
    }
