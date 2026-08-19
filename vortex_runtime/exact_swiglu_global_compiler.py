from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .fixed_public_dynamic_common import ExecutorInvariantError, dtype_name, tensor_bytes


FINAL_P50_TARGET_FRACTION = 1.2 * 4.0 / 405.0
MECHANISM_FINGERPRINT = "global_swiglu_atom_dag_exact_equivalence_cse_v1"


@dataclass(frozen=True)
class EquivalenceGroup:
    representative: int
    members: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"representative": self.representative, "members": list(self.members)}


@dataclass(frozen=True)
class SwiGLUProgramPlan:
    mechanism: str
    input_width: int
    intermediate_width: int
    output_width: int
    dtype: str
    activation: str
    gate_bias: bool
    up_bias: bool
    down_bias: bool
    reference_mac_count: int
    strict_mac_count: int
    optimistic_algebraic_mac_count: int
    strict_operation_fraction: float
    optimistic_algebraic_operation_fraction: float
    target_fraction: float
    unique_gate_rows: int
    unique_up_rows: int
    unique_gate_up_atoms: int
    duplicate_gate_rows: int
    duplicate_up_rows: int
    duplicate_gate_up_atoms: int
    antipodal_gate_pairs: int
    signed_gate_up_pairs: int
    zero_down_columns: int
    duplicate_down_columns: int
    strict_metadata_bytes: int
    strict_gate_groups: tuple[EquivalenceGroup, ...]
    strict_up_groups: tuple[EquivalenceGroup, ...]
    strict_atom_groups: tuple[EquivalenceGroup, ...]
    exact_real_identity_enabled: bool
    structural_gate_passed: bool
    decision: str
    deterministic_sha256: str

    def to_dict(self, *, include_groups: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if include_groups:
            result["strict_gate_groups"] = [group.to_dict() for group in self.strict_gate_groups]
            result["strict_up_groups"] = [group.to_dict() for group in self.strict_up_groups]
            result["strict_atom_groups"] = [group.to_dict() for group in self.strict_atom_groups]
        else:
            result.pop("strict_gate_groups", None)
            result.pop("strict_up_groups", None)
            result.pop("strict_atom_groups", None)
        return result


@dataclass(frozen=True)
class ABIProbeResult:
    mode: str
    input_rows: int
    exact_rows: int
    mismatch_rows: int
    mismatch_elements: int
    first_mismatch_row: int | None
    max_abs_error: float
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _callable_name(value: Any) -> str:
    cls = value.__class__.__name__
    name = getattr(value, "__name__", None)
    return str(name or cls)


def _is_silu(value: Any) -> bool:
    text = f"{_callable_name(value)} {value!r}".lower()
    return "silu" in text or "swish" in text


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_floating_point(tensor):
        raise ExecutorInvariantError(f"{name} must be floating point")
    if not bool(torch.isfinite(tensor).all()):
        raise ExecutorInvariantError(f"{name} contains non-finite values")


def _signature(parts: Sequence[torch.Tensor | None]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if part is None:
            digest.update(b"<none>")
            continue
        value = part.detach().contiguous().cpu()
        digest.update(dtype_name(value.dtype).encode())
        digest.update(str(tuple(int(x) for x in value.shape)).encode())
        digest.update(tensor_bytes(value))
    return digest.hexdigest()


def _row_parts(weight: torch.Tensor, bias: torch.Tensor | None, index: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    return weight[index], None if bias is None else bias[index : index + 1]


def _groups(keys: Sequence[str]) -> tuple[EquivalenceGroup, ...]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        buckets[key].append(index)
    ordered = sorted(buckets.values(), key=lambda members: members[0])
    return tuple(EquivalenceGroup(members[0], tuple(members)) for members in ordered)


def _count_antipodal_rows(weight: torch.Tensor, bias: torch.Tensor | None) -> int:
    original_keys = [_signature(_row_parts(weight, bias, i)) for i in range(int(weight.shape[0]))]
    positions: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(original_keys):
        positions[key].append(i)
    count = 0
    for i in range(int(weight.shape[0])):
        row, scalar = _row_parts(weight, bias, i)
        negative_key = _signature((-row, None if scalar is None else -scalar))
        count += sum(1 for j in positions.get(negative_key, ()) if j > i)
    return count


def _count_signed_gate_up_pairs(
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor | None,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor | None,
) -> int:
    pair_positions: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(int(gate_weight.shape[0])):
        gate_key = _signature(_row_parts(gate_weight, gate_bias, i))
        up_key = _signature(_row_parts(up_weight, up_bias, i))
        pair_positions[(gate_key, up_key)].append(i)
    count = 0
    for i in range(int(gate_weight.shape[0])):
        gate_row, gate_scalar = _row_parts(gate_weight, gate_bias, i)
        up_row, up_scalar = _row_parts(up_weight, up_bias, i)
        negative_gate = _signature((-gate_row, None if gate_scalar is None else -gate_scalar))
        same_up = _signature((up_row, up_scalar))
        negative_up = _signature((-up_row, None if up_scalar is None else -up_scalar))
        candidates = set(pair_positions.get((negative_gate, same_up), ()))
        candidates.update(pair_positions.get((negative_gate, negative_up), ()))
        count += sum(1 for j in candidates if j > i)
    return count


def _count_zero_columns(weight: torch.Tensor) -> int:
    return int((torch.count_nonzero(weight, dim=0) == 0).sum().item())


def _column_duplicate_count(weight: torch.Tensor) -> int:
    keys = [_signature((weight[:, i],)) for i in range(int(weight.shape[1]))]
    return len(keys) - len(set(keys))


def _plan_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def analyze_swiglu(mlp: nn.Module) -> SwiGLUProgramPlan:
    """Compile a complete SwiGLU expression into exact functional classes.

    The strict count permits only byte-identical gate/up row reuse and retains
    the original down accumulation. The optimistic count additionally grants
    exact-real merging of identical nonlinear atoms into one down column. The
    latter is a favorable lower bound: floating reduction order and every
    non-MAC cost are free.
    """
    for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        if not hasattr(mlp, name):
            raise ExecutorInvariantError(f"SwiGLU module missing {name}")
    gate: nn.Linear = mlp.gate_proj
    up: nn.Linear = mlp.up_proj
    down: nn.Linear = mlp.down_proj
    if gate.weight.ndim != 2 or up.weight.ndim != 2 or down.weight.ndim != 2:
        raise ExecutorInvariantError("SwiGLU projections must be matrices")
    intermediate, input_width = map(int, gate.weight.shape)
    if tuple(map(int, up.weight.shape)) != (intermediate, input_width):
        raise ExecutorInvariantError("gate/up shapes differ")
    output_width, down_input = map(int, down.weight.shape)
    if down_input != intermediate:
        raise ExecutorInvariantError("down input width does not equal intermediate width")
    if gate.weight.dtype != up.weight.dtype or gate.weight.dtype != down.weight.dtype:
        raise ExecutorInvariantError("mixed projection dtypes are outside this compiler ABI")
    for name, tensor in (("gate", gate.weight), ("up", up.weight), ("down", down.weight)):
        _require_finite(name, tensor)
    for name, bias in (("gate_bias", gate.bias), ("up_bias", up.bias), ("down_bias", down.bias)):
        if bias is not None:
            _require_finite(name, bias)

    gate_keys = [_signature(_row_parts(gate.weight, gate.bias, i)) for i in range(intermediate)]
    up_keys = [_signature(_row_parts(up.weight, up.bias, i)) for i in range(intermediate)]
    atom_keys = [_plan_hash({"gate": gate_keys[i], "up": up_keys[i]}) for i in range(intermediate)]
    gate_groups = _groups(gate_keys)
    up_groups = _groups(up_keys)
    atom_groups = _groups(atom_keys)
    unique_gate = len(gate_groups)
    unique_up = len(up_groups)
    unique_atoms = len(atom_groups)
    antipodal = _count_antipodal_rows(gate.weight, gate.bias)
    signed_pairs = _count_signed_gate_up_pairs(gate.weight, gate.bias, up.weight, up.bias)
    zero_down = _count_zero_columns(down.weight)
    duplicate_down = _column_duplicate_count(down.weight)

    reference_mac = 2 * intermediate * input_width + output_width * intermediate
    strict_mac = unique_gate * input_width + unique_up * input_width + output_width * intermediate
    optimistic_mac = unique_gate * input_width + unique_up * input_width + output_width * unique_atoms
    strict_fraction = strict_mac / reference_mac if reference_mac else math.inf
    optimistic_fraction = optimistic_mac / reference_mac if reference_mac else math.inf
    metadata_bytes = 12 * intermediate + 16 * (unique_gate + unique_up + unique_atoms)
    structural_pass = optimistic_fraction <= FINAL_P50_TARGET_FRACTION
    decision = (
        "PROMOTE_EXACT_SWIGLU_EQUIVALENCE_COMPILER_TO_NATIVE_ABI_GATE"
        if structural_pass
        else "REJECT_EXACT_SWIGLU_EQUIVALENCE_CSE_AS_CORE"
    )
    hash_payload = {
        "mechanism": MECHANISM_FINGERPRINT,
        "shape": [input_width, intermediate, output_width],
        "dtype": dtype_name(gate.weight.dtype),
        "activation": _callable_name(mlp.act_fn),
        "gate_keys": gate_keys,
        "up_keys": up_keys,
        "atom_keys": atom_keys,
        "antipodal_gate_pairs": antipodal,
        "signed_gate_up_pairs": signed_pairs,
        "zero_down_columns": zero_down,
        "duplicate_down_columns": duplicate_down,
    }
    deterministic_sha = _plan_hash(hash_payload)
    return SwiGLUProgramPlan(
        mechanism=MECHANISM_FINGERPRINT,
        input_width=input_width,
        intermediate_width=intermediate,
        output_width=output_width,
        dtype=dtype_name(gate.weight.dtype),
        activation=_callable_name(mlp.act_fn),
        gate_bias=gate.bias is not None,
        up_bias=up.bias is not None,
        down_bias=down.bias is not None,
        reference_mac_count=reference_mac,
        strict_mac_count=strict_mac,
        optimistic_algebraic_mac_count=optimistic_mac,
        strict_operation_fraction=strict_fraction,
        optimistic_algebraic_operation_fraction=optimistic_fraction,
        target_fraction=FINAL_P50_TARGET_FRACTION,
        unique_gate_rows=unique_gate,
        unique_up_rows=unique_up,
        unique_gate_up_atoms=unique_atoms,
        duplicate_gate_rows=intermediate - unique_gate,
        duplicate_up_rows=intermediate - unique_up,
        duplicate_gate_up_atoms=intermediate - unique_atoms,
        antipodal_gate_pairs=antipodal,
        signed_gate_up_pairs=signed_pairs,
        zero_down_columns=zero_down,
        duplicate_down_columns=duplicate_down,
        strict_metadata_bytes=metadata_bytes,
        strict_gate_groups=gate_groups,
        strict_up_groups=up_groups,
        strict_atom_groups=atom_groups,
        exact_real_identity_enabled=_is_silu(mlp.act_fn),
        structural_gate_passed=structural_pass,
        decision=decision,
        deterministic_sha256=deterministic_sha,
    )


class ExactGlobalSwiGLUProgram(nn.Module):
    """Executable whole-function IR for ``Wd[silu(Wg x) * (Wu x)]``.

    ``reference_order`` is a semantic control. ``joint_gate_up`` replaces the
    two independent input projections with one row-concatenated projection and
    is admitted only after a frozen native-ABI bitwise probe. The down operation
    stays in reference order; this two-call lowering is never claimed to close
    the final target by itself.
    """
    def __init__(self, mlp: nn.Module, *, mode: str = "reference_order") -> None:
        super().__init__()
        if mode not in {"reference_order", "joint_gate_up"}:
            raise ExecutorInvariantError(f"unsupported exact SwiGLU lowering: {mode}")
        self.plan = analyze_swiglu(mlp)
        self.mode = mode
        self.act_fn: Callable[[torch.Tensor], torch.Tensor] = mlp.act_fn
        gate: nn.Linear = mlp.gate_proj
        up: nn.Linear = mlp.up_proj
        down: nn.Linear = mlp.down_proj
        self.register_buffer("gate_weight", gate.weight.detach().clone(), persistent=True)
        self.register_buffer("up_weight", up.weight.detach().clone(), persistent=True)
        self.register_buffer("down_weight", down.weight.detach().clone(), persistent=True)
        self.register_buffer("gate_bias", None if gate.bias is None else gate.bias.detach().clone(), persistent=True)
        self.register_buffer("up_bias", None if up.bias is None else up.bias.detach().clone(), persistent=True)
        self.register_buffer("down_bias", None if down.bias is None else down.bias.detach().clone(), persistent=True)
        if mode == "joint_gate_up":
            self.register_buffer("joint_weight", torch.cat((self.gate_weight, self.up_weight), dim=0), persistent=True)
            if self.gate_bias is None and self.up_bias is None:
                self.register_buffer("joint_bias", None, persistent=True)
            elif self.gate_bias is not None and self.up_bias is not None:
                self.register_buffer("joint_bias", torch.cat((self.gate_bias, self.up_bias), dim=0), persistent=True)
            else:
                raise ExecutorInvariantError("joint gate/up ABI requires matching bias presence")
        else:
            self.register_buffer("joint_weight", None, persistent=False)
            self.register_buffer("joint_bias", None, persistent=False)
        self.reset_stats()

    def reset_stats(self) -> None:
        self._stats = {
            "projection_calls": 0,
            "silu_elements": 0,
            "multiply_elements": 0,
            "cold_bytes": 0,
            "hot_materialized_bytes": 0,
            "pcie_bytes": 0,
            "gpu_bytes": 0,
            "decompression_ns": 0,
            "materialization_ns": 0,
            "integrity_probes": 0,
            "peak_projection_hot_bytes": 0,
        }

    def stats_snapshot(self) -> dict[str, int]:
        return dict(self._stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "joint_gate_up":
            joint = F.linear(x, self.joint_weight, self.joint_bias)
            self._stats["projection_calls"] += 1
            gate, up = torch.split(joint, self.plan.intermediate_width, dim=-1)
        else:
            gate = F.linear(x, self.gate_weight, self.gate_bias)
            up = F.linear(x, self.up_weight, self.up_bias)
            self._stats["projection_calls"] += 2
        activated = self.act_fn(gate)
        hidden = activated * up
        self._stats["silu_elements"] += int(activated.numel())
        self._stats["multiply_elements"] += int(hidden.numel())
        output = F.linear(hidden, self.down_weight, self.down_bias)
        self._stats["projection_calls"] += 1
        return output


def compile_swiglu_function(mlp: nn.Module, *, mode: str = "reference_order") -> tuple[ExactGlobalSwiGLUProgram, SwiGLUProgramPlan]:
    program = ExactGlobalSwiGLUProgram(mlp, mode=mode)
    return program, program.plan


def probe_native_abi(
    reference_mlp: nn.Module,
    candidate: nn.Module,
    inputs: torch.Tensor | Iterable[torch.Tensor],
    *,
    mode: str,
) -> ABIProbeResult:
    rows: list[torch.Tensor]
    if isinstance(inputs, torch.Tensor):
        value = inputs.unsqueeze(0) if inputs.ndim == 1 else inputs
        rows = [value[i : i + 1] for i in range(int(value.shape[0]))]
    else:
        rows = []
        for value in inputs:
            value = value.unsqueeze(0) if value.ndim == 1 else value
            rows.extend(value[i : i + 1] for i in range(int(value.shape[0])))
    exact_rows = 0
    mismatch_elements = 0
    first_mismatch: int | None = None
    max_abs = 0.0
    with torch.inference_mode():
        for index, row in enumerate(rows):
            expected = reference_mlp(row)
            actual = candidate(row)
            if torch.equal(expected, actual):
                exact_rows += 1
                continue
            if first_mismatch is None:
                first_mismatch = index
            mismatch_elements += int(torch.count_nonzero(expected != actual).item())
            difference = (expected.float() - actual.float()).abs()
            if difference.numel():
                max_abs = max(max_abs, float(difference.max().item()))
    return ABIProbeResult(
        mode=mode,
        input_rows=len(rows),
        exact_rows=exact_rows,
        mismatch_rows=len(rows) - exact_rows,
        mismatch_elements=mismatch_elements,
        first_mismatch_row=first_mismatch,
        max_abs_error=max_abs,
        exact=exact_rows == len(rows),
    )


def aggregate_plans(plans: Sequence[SwiGLUProgramPlan]) -> dict[str, Any]:
    if not plans:
        raise ExecutorInvariantError("at least one SwiGLU plan is required")
    reference = sum(plan.reference_mac_count for plan in plans)
    strict = sum(plan.strict_mac_count for plan in plans)
    optimistic = sum(plan.optimistic_algebraic_mac_count for plan in plans)
    payload = {
        "mechanism": MECHANISM_FINGERPRINT,
        "layer_count": len(plans),
        "reference_mac_count": reference,
        "strict_mac_count": strict,
        "optimistic_algebraic_mac_count": optimistic,
        "strict_operation_fraction": strict / reference,
        "optimistic_algebraic_operation_fraction": optimistic / reference,
        "target_fraction": FINAL_P50_TARGET_FRACTION,
        "duplicate_gate_rows": sum(plan.duplicate_gate_rows for plan in plans),
        "duplicate_up_rows": sum(plan.duplicate_up_rows for plan in plans),
        "duplicate_gate_up_atoms": sum(plan.duplicate_gate_up_atoms for plan in plans),
        "antipodal_gate_pairs": sum(plan.antipodal_gate_pairs for plan in plans),
        "signed_gate_up_pairs": sum(plan.signed_gate_up_pairs for plan in plans),
        "zero_down_columns": sum(plan.zero_down_columns for plan in plans),
        "duplicate_down_columns": sum(plan.duplicate_down_columns for plan in plans),
        "metadata_bytes": sum(plan.strict_metadata_bytes for plan in plans),
    }
    payload["structural_gate_passed"] = payload["optimistic_algebraic_operation_fraction"] <= FINAL_P50_TARGET_FRACTION
    payload["decision"] = (
        "PROMOTE_EXACT_SWIGLU_EQUIVALENCE_COMPILER_TO_NATIVE_ABI_GATE"
        if payload["structural_gate_passed"]
        else "REJECT_EXACT_SWIGLU_EQUIVALENCE_CSE_AS_CORE"
    )
    payload["deterministic_sha256"] = _plan_hash(payload)
    return payload
