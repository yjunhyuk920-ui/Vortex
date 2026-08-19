from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

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
BF16_SIGN_EXPONENT_BITS = 9
BF16_MANTISSA_BITS = 7
BF16_TOTAL_BITS = 16


class SuccessiveRefinementError(RuntimeError):
    """Fail-closed invariant violation for the BF16 refinement compiler."""


def _require_bf16(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.dtype != torch.bfloat16:
        raise SuccessiveRefinementError(f"{name} must be torch.bfloat16, got {tensor.dtype}")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if not torch.isfinite(tensor.float()).all():
        raise SuccessiveRefinementError(f"{name} contains nonfinite values")
    return tensor


def bf16_prefix(tensor: torch.Tensor, mantissa_bits: int) -> torch.Tensor:
    """Keep sign, exponent and the requested most-significant mantissa bits."""
    tensor = _require_bf16(tensor, name="tensor")
    keep = int(mantissa_bits)
    if not 0 <= keep <= BF16_MANTISSA_BITS:
        raise SuccessiveRefinementError(
            f"mantissa_bits must be in [0,{BF16_MANTISSA_BITS}], got {keep}"
        )
    low_bits = BF16_MANTISSA_BITS - keep
    mask = 0xFFFF ^ ((1 << low_bits) - 1) if low_bits else 0xFFFF
    bit_view = tensor.view(torch.uint16)
    mask_tensor = torch.tensor(mask, dtype=torch.uint16, device=bit_view.device)
    return (bit_view & mask_tensor).view(torch.bfloat16)


def tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().contiguous().cpu()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def symbol_entropy_bits(tensor: torch.Tensor, mantissa_bits: int) -> float:
    """Zero-order empirical entropy of BF16 prefix symbols in bits/symbol."""
    prefix = bf16_prefix(tensor.detach().contiguous().cpu(), mantissa_bits)
    symbols = prefix.view(torch.uint16).reshape(-1).to(torch.int64)
    counts = torch.bincount(symbols, minlength=1 << 16)
    nonzero = counts[counts > 0].to(torch.float64)
    probabilities = nonzero / nonzero.sum()
    return float((-(probabilities * torch.log2(probabilities))).sum().item())


def row_symbol_entropy_bits(tensor: torch.Tensor, mantissa_bits: int) -> torch.Tensor:
    tensor = _require_bf16(tensor, name="row tensor")
    if tensor.ndim != 2:
        raise SuccessiveRefinementError("row entropy requires a 2-D tensor")
    prefix = bf16_prefix(tensor.detach().contiguous().cpu(), mantissa_bits)
    result = torch.empty(prefix.shape[0], dtype=torch.float64)
    for row_index, row in enumerate(prefix):
        symbols = row.contiguous().view(torch.uint16).to(torch.int64)
        _, counts = torch.unique(symbols, return_counts=True)
        probabilities = counts.to(torch.float64) / float(symbols.numel())
        result[row_index] = (-(probabilities * torch.log2(probabilities))).sum()
    return result


@dataclass(frozen=True)
class PrecisionCost:
    gate_up_mantissa_bits: int
    down_mantissa_bits: int
    raw_bits: int
    entropy_bits: float
    raw_fraction_of_bf16: float
    entropy_fraction_of_full: float
    ideal_information_speedup: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActivationOracleResult:
    activation_index: int
    mode: str
    exact: bool
    gate_up_mantissa_bits: int
    down_mantissa_bits: int | None
    down_mantissa_histogram: dict[str, int]
    exact_output_rows: int
    output_rows: int
    raw_fraction_of_bf16: float
    entropy_fraction_of_full: float
    ideal_information_speedup: float
    minimum_perfect_acceptance_p50: int
    minimum_perfect_acceptance_p95: int
    candidate_sha256: str
    reference_sha256: str
    control_full_precision_exact: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchRefinementResult:
    manifest: dict[str, Any]
    global_rows: list[dict[str, Any]]
    row_adaptive_rows: list[dict[str, Any]]
    precision_grid: list[dict[str, Any]]
    controls: dict[str, Any]


class CompiledGlobalBF16Refinement:
    """Checkpoint-static whole-SwiGLU BF16 prefix/refinement reference."""

    def __init__(
        self,
        *,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.gate_weight = _require_bf16(
            gate_weight.detach().contiguous().cpu(), name="gate_weight"
        )
        self.up_weight = _require_bf16(
            up_weight.detach().contiguous().cpu(), name="up_weight"
        )
        self.down_weight = _require_bf16(
            down_weight.detach().contiguous().cpu(), name="down_weight"
        )
        if self.gate_weight.shape != self.up_weight.shape:
            raise SuccessiveRefinementError("gate/up shape mismatch")
        if self.down_weight.ndim != 2 or self.gate_weight.ndim != 2:
            raise SuccessiveRefinementError("projection weights must be 2-D")
        if self.down_weight.shape[1] != self.gate_weight.shape[0]:
            raise SuccessiveRefinementError("down/intermediate shape mismatch")
        self.act_fn = act_fn
        self._prefix_weights: dict[tuple[str, int], torch.Tensor] = {}
        for bits in range(BF16_MANTISSA_BITS + 1):
            self._prefix_weights[("gate", bits)] = bf16_prefix(self.gate_weight, bits)
            self._prefix_weights[("up", bits)] = bf16_prefix(self.up_weight, bits)
            self._prefix_weights[("down", bits)] = bf16_prefix(self.down_weight, bits)

        self._parameter_counts = {
            "gate": int(self.gate_weight.numel()),
            "up": int(self.up_weight.numel()),
            "down": int(self.down_weight.numel()),
        }
        self._full_entropy = {
            "gate": symbol_entropy_bits(self.gate_weight, BF16_MANTISSA_BITS),
            "up": symbol_entropy_bits(self.up_weight, BF16_MANTISSA_BITS),
            "down": symbol_entropy_bits(self.down_weight, BF16_MANTISSA_BITS),
        }
        self._entropy = {
            name: {
                bits: symbol_entropy_bits(weight, bits)
                for bits in range(BF16_MANTISSA_BITS + 1)
            }
            for name, weight in (
                ("gate", self.gate_weight),
                ("up", self.up_weight),
                ("down", self.down_weight),
            )
        }
        self._down_row_entropy = {
            bits: row_symbol_entropy_bits(self.down_weight, bits)
            for bits in range(BF16_MANTISSA_BITS + 1)
        }
        self._full_entropy_total = sum(
            self._full_entropy[name] * self._parameter_counts[name]
            for name in ("gate", "up", "down")
        )
        if not self._full_entropy_total > 0:
            raise SuccessiveRefinementError("degenerate full entropy total")

    @classmethod
    def from_mlp(cls, mlp: nn.Module) -> "CompiledGlobalBF16Refinement":
        for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
            if not hasattr(mlp, name):
                raise SuccessiveRefinementError(f"MLP missing {name}")
        for name in ("gate_proj", "up_proj", "down_proj"):
            if getattr(mlp, name).bias is not None:
                raise SuccessiveRefinementError("EXP-086A fails closed for projection bias")
        return cls(
            gate_weight=mlp.gate_proj.weight,
            up_weight=mlp.up_proj.weight,
            down_weight=mlp.down_proj.weight,
            act_fn=mlp.act_fn,
        )

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
    def parameter_count(self) -> int:
        return sum(self._parameter_counts.values())

    def _fingerprint(self) -> str:
        payload = {
            "gate_sha256": tensor_sha256(self.gate_weight),
            "up_sha256": tensor_sha256(self.up_weight),
            "down_sha256": tensor_sha256(self.down_weight),
            "shapes": {
                "gate": list(self.gate_weight.shape),
                "up": list(self.up_weight.shape),
                "down": list(self.down_weight.shape),
            },
            "mechanism": "global_bf16_sign_exponent_mantissa_successive_refinement",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "format": "global-bf16-successive-refinement-v1",
            "mechanism": "global_bf16_sign_exponent_mantissa_successive_refinement",
            "fingerprint": self._fingerprint(),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "output_size": self.output_size,
            "parameter_count": self.parameter_count,
            "parameter_counts": dict(self._parameter_counts),
            "weight_sha256": {
                "gate": tensor_sha256(self.gate_weight),
                "up": tensor_sha256(self.up_weight),
                "down": tensor_sha256(self.down_weight),
            },
            "zero_order_entropy_bits_per_weight": {
                name: {str(bits): self._entropy[name][bits] for bits in range(8)}
                for name in ("gate", "up", "down")
            },
            "full_entropy_bits_per_weight": dict(self._full_entropy),
            "full_entropy_total_bits": self._full_entropy_total,
            "exactness_contract": {
                "prefix": "preserve BF16 sign/exponent and k MSB mantissa bits",
                "k7": "bitwise identity with original checkpoint",
                "oracle": "reference-aided, non-deployable and favorable",
                "unread_residual": "accepted only on complete MLP BF16 equality",
            },
        }

    def _validate_batch(self, x: torch.Tensor) -> torch.Tensor:
        x = _require_bf16(x.detach().contiguous().cpu(), name="activation")
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.hidden_size:
            raise SuccessiveRefinementError(
                f"expected [N,{self.hidden_size}], got {tuple(x.shape)}"
            )
        return x

    @staticmethod
    def _bitwise_row_equal(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape or left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
            raise SuccessiveRefinementError("bitwise comparison ABI mismatch")
        return left.view(torch.uint16) == right.view(torch.uint16)

    def _reference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = F.linear(x, self.gate_weight)
        up = F.linear(x, self.up_weight)
        z = self.act_fn(gate) * up
        return gate, up, F.linear(z, self.down_weight)

    def _candidate_z(self, x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = F.linear(x, self._prefix_weights[("gate", bits)])
        up = F.linear(x, self._prefix_weights[("up", bits)])
        return gate, up, self.act_fn(gate) * up

    def _global_cost(self, gate_up_bits: int, down_bits: int) -> PrecisionCost:
        counts = self._parameter_counts
        raw_bits = (
            (BF16_SIGN_EXPONENT_BITS + gate_up_bits) * (counts["gate"] + counts["up"])
            + (BF16_SIGN_EXPONENT_BITS + down_bits) * counts["down"]
        )
        entropy_bits = (
            self._entropy["gate"][gate_up_bits] * counts["gate"]
            + self._entropy["up"][gate_up_bits] * counts["up"]
            + self._entropy["down"][down_bits] * counts["down"]
        )
        raw_fraction = raw_bits / (BF16_TOTAL_BITS * self.parameter_count)
        entropy_fraction = entropy_bits / self._full_entropy_total
        return PrecisionCost(
            gate_up_bits, down_bits, int(raw_bits), float(entropy_bits),
            float(raw_fraction), float(entropy_fraction),
            float(1.0 / max(entropy_fraction, 1e-300)),
        )

    def _row_adaptive_cost(self, gate_up_bits: int, per_row: torch.Tensor) -> tuple[float, float]:
        counts = self._parameter_counts
        gate_up_raw = (BF16_SIGN_EXPONENT_BITS + gate_up_bits) * (counts["gate"] + counts["up"])
        down_raw = sum(
            (BF16_SIGN_EXPONENT_BITS + int(bits)) * self.intermediate_size
            for bits in per_row.tolist()
        )
        raw_fraction = (gate_up_raw + down_raw) / (BF16_TOTAL_BITS * self.parameter_count)
        entropy_bits = (
            self._entropy["gate"][gate_up_bits] * counts["gate"]
            + self._entropy["up"][gate_up_bits] * counts["up"]
        )
        for row_index, bits in enumerate(per_row.tolist()):
            entropy_bits += float(self._down_row_entropy[int(bits)][row_index]) * self.intermediate_size
        return float(raw_fraction), float(entropy_bits / self._full_entropy_total)

    @staticmethod
    def _minimum_acceptance(information_fraction: float, allowance: float) -> int:
        return max(1, int(math.ceil(LLAMA405_MLP_PARAMETER_SHARE * information_fraction / allowance)))

    def evaluate(self, x: torch.Tensor) -> BatchRefinementResult:
        x = self._validate_batch(x)
        reference_gate, reference_up, reference_output = self._reference(x)
        batch = int(x.shape[0])
        z_by_bits: dict[int, torch.Tensor] = {}
        gate_up_exact: dict[int, torch.Tensor] = {}
        for bits in range(8):
            gate, up, z = self._candidate_z(x, bits)
            z_by_bits[bits] = z
            gate_up_exact[bits] = self._bitwise_row_equal(gate, reference_gate) & self._bitwise_row_equal(up, reference_up)

        candidates: dict[tuple[int, int], torch.Tensor] = {}
        masks: dict[tuple[int, int], torch.Tensor] = {}
        grid: list[dict[str, Any]] = []
        for gu in range(8):
            for down in range(8):
                candidate = F.linear(z_by_bits[gu], self._prefix_weights[("down", down)])
                candidates[(gu, down)] = candidate
                mask = self._bitwise_row_equal(candidate, reference_output)
                masks[(gu, down)] = mask
                cost = self._global_cost(gu, down)
                grid.append({
                    **cost.to_dict(),
                    "exact_vectors": int(mask.all(dim=1).sum()),
                    "activation_count": batch,
                    "exact_vector_fraction": float(mask.all(dim=1).double().mean()),
                    "exact_output_row_fraction": float(mask.double().mean()),
                    "gate_up_exact_scalar_fraction": float(gate_up_exact[gu].double().mean()),
                })

        full_control = masks[(7, 7)].all(dim=1)
        if not bool(full_control.all()):
            raise SuccessiveRefinementError("full-precision candidate mismatch")
        global_rows: list[dict[str, Any]] = []
        adaptive_rows: list[dict[str, Any]] = []
        for index in range(batch):
            exact_pairs = [
                (self._global_cost(gu, down), gu, down)
                for gu in range(8) for down in range(8)
                if bool(masks[(gu, down)][index].all())
            ]
            cost, gu, down = min(exact_pairs, key=lambda item: (
                item[0].entropy_fraction_of_full,
                item[0].raw_fraction_of_bf16,
                item[1] + item[2], item[1], item[2],
            ))
            candidate = candidates[(gu, down)][index]
            global_rows.append(ActivationOracleResult(
                index, "global_uniform_precision_oracle", True, gu, down,
                {str(down): self.output_size}, self.output_size, self.output_size,
                cost.raw_fraction_of_bf16, cost.entropy_fraction_of_full,
                cost.ideal_information_speedup,
                self._minimum_acceptance(cost.entropy_fraction_of_full, P50_WHOLE_MODEL_FRACTION),
                self._minimum_acceptance(cost.entropy_fraction_of_full, P95_WHOLE_MODEL_FRACTION),
                tensor_sha256(candidate), tensor_sha256(reference_output[index]), True,
            ).to_dict())

            choices = []
            for candidate_gu in range(8):
                per_row = torch.full((self.output_size,), 7, dtype=torch.int64)
                feasible = torch.zeros(self.output_size, dtype=torch.bool)
                for candidate_down in range(8):
                    newly = masks[(candidate_gu, candidate_down)][index] & ~feasible
                    per_row[newly] = candidate_down
                    feasible |= masks[(candidate_gu, candidate_down)][index]
                if bool(feasible.all()):
                    raw_fraction, entropy_fraction = self._row_adaptive_cost(candidate_gu, per_row)
                    histogram: dict[str, int] = {}
                    for value in per_row.tolist():
                        key = str(int(value)); histogram[key] = histogram.get(key, 0) + 1
                    choices.append((entropy_fraction, raw_fraction, candidate_gu, per_row, histogram))
            entropy_fraction, raw_fraction, gu, per_row, histogram = min(
                choices, key=lambda item: (item[0], item[1], int(item[3].sum()), item[2])
            )
            adaptive_output = torch.empty(self.output_size, dtype=torch.bfloat16)
            for down in range(8):
                row_mask = per_row == down
                if bool(row_mask.any()):
                    adaptive_output[row_mask] = candidates[(gu, down)][index, row_mask]
            if not bool(self._bitwise_row_equal(adaptive_output.unsqueeze(0), reference_output[index].unsqueeze(0)).all()):
                raise SuccessiveRefinementError("row-adaptive reconstruction mismatch")
            adaptive_rows.append(ActivationOracleResult(
                index, "global_gate_up_row_adaptive_down_oracle", True, gu, None,
                histogram, self.output_size, self.output_size,
                raw_fraction, entropy_fraction, 1.0 / max(entropy_fraction, 1e-300),
                self._minimum_acceptance(entropy_fraction, P50_WHOLE_MODEL_FRACTION),
                self._minimum_acceptance(entropy_fraction, P95_WHOLE_MODEL_FRACTION),
                tensor_sha256(adaptive_output), tensor_sha256(reference_output[index]), True,
            ).to_dict())

        controls = {
            "full_precision_exact_activations": int(full_control.sum()),
            "activation_count": batch,
            "full_precision_control_passed": bool(full_control.all()),
            "prefix_identity_weight_mismatches": {
                name: int(torch.count_nonzero(self._prefix_weights[(name, 7)].view(torch.uint16) != weight.view(torch.uint16)))
                for name, weight in (("gate", self.gate_weight), ("up", self.up_weight), ("down", self.down_weight))
            },
        }
        return BatchRefinementResult(self.manifest(), global_rows, adaptive_rows, grid, controls)


def compile_global_bf16_refinement(mlp: nn.Module) -> CompiledGlobalBF16Refinement:
    return CompiledGlobalBF16Refinement.from_mlp(mlp)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise SuccessiveRefinementError("empty population")
    return float(torch.quantile(torch.tensor(list(map(float, values)), dtype=torch.float64), q, interpolation="higher"))


def aggregate_oracle_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise SuccessiveRefinementError("empty oracle population")
    entropy = [float(row["entropy_fraction_of_full"]) for row in rows]
    raw = [float(row["raw_fraction_of_bf16"]) for row in rows]
    return {
        "count": len(rows),
        "all_exact": all(bool(row["exact"]) for row in rows),
        "entropy_fraction_p05": percentile(entropy, 0.05),
        "entropy_fraction_p50": percentile(entropy, 0.50),
        "entropy_fraction_p95": percentile(entropy, 0.95),
        "raw_fraction_p50": percentile(raw, 0.50),
        "raw_fraction_p95": percentile(raw, 0.95),
        "ideal_information_speedup_p50": 1.0 / max(percentile(entropy, 0.50), 1e-300),
        "minimum_perfect_acceptance_p50": int(percentile([float(row["minimum_perfect_acceptance_p50"]) for row in rows], 0.50)),
        "minimum_perfect_acceptance_p95": int(percentile([float(row["minimum_perfect_acceptance_p95"]) for row in rows], 0.95)),
        "gate_up_mantissa_bits_p50": int(percentile([float(row["gate_up_mantissa_bits"]) for row in rows], 0.50)),
    }
