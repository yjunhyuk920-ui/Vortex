from __future__ import annotations

import gc
import hashlib
import json
import math
import tempfile
import time
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .fixed_public_dynamic_common import (
    ArtifactIntegrityError,
    CompiledArtifact,
    DTYPE_BY_NAME,
    ExecutorInvariantError,
    TensorRecord,
    config_dict,
    directory_bytes,
    dtype_name,
    json_sha256,
    model_tensor_sha256,
    module_parameter_bytes,
    resolved_model_identity,
    rss_bytes,
    sha256_bytes,
    tensor_bytes,
)
from .fixed_public_dynamic_executor import ExactReplacementDecoderLayer


MECHANISM = "checkpoint_whole_swiglu_proof_carrying_ir_v1"
IR_VERSION = "exact-swiglu-global-ir-v1"
DEFAULT_FIBER_TILE_ROWS = 128
DEFAULT_OUTPUT_TILE_ROWS = 128
TARGET_EQUIVALENT_FRACTION = 1.2 * 4.0 / 405.0
_FIBER_TREE_MAX_ROWS = 4096


def _pack_tensor(tensor: torch.Tensor, artifact_dir: Path, logical_name: str) -> TensorRecord:
    raw = tensor_bytes(tensor)
    compressed = zlib.compress(raw, 9)
    codec, payload = ("zlib-9", compressed) if len(compressed) < len(raw) else ("raw", raw)
    relative_path = logical_name.replace(".", "__") + ".bin"
    (artifact_dir / relative_path).write_bytes(payload)
    return TensorRecord(
        name=logical_name,
        shape=tuple(map(int, tensor.shape)),
        dtype=dtype_name(tensor.dtype),
        raw_bytes=len(raw),
        artifact_bytes=len(payload),
        codec=codec,
        relative_path=relative_path,
        raw_sha256=sha256_bytes(raw),
        artifact_sha256=sha256_bytes(payload),
    )


def _row_digests(tensor: torch.Tensor) -> list[str]:
    if tensor.ndim != 2:
        raise ExecutorInvariantError("row digest requires a two-dimensional tensor")
    cpu = tensor.detach().contiguous().cpu()
    return [sha256_bytes(tensor_bytes(row)) for row in cpu]


def _duplicate_summary(digests: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for digest in digests:
        counts[digest] = counts.get(digest, 0) + 1
    return {
        "rows": len(digests),
        "unique": len(counts),
        "duplicate_rows": len(digests) - len(counts),
        "largest_equivalence_class": max(counts.values(), default=0),
    }


def _antipodal_pair_count(tensor: torch.Tensor) -> int:
    positive = set(_row_digests(tensor))
    negative = _row_digests(-tensor)
    return sum(1 for digest in negative if digest in positive) // 2


def _symmetric_row_q4(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float().cpu()
    max_abs = value.abs().amax(dim=1, keepdim=True)
    scale = torch.where(max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs))
    return torch.round(value / scale).clamp(-8, 7).to(torch.int8)


def _collision_free_block_ids(values: torch.Tensor, block_width: int) -> torch.Tensor:
    if values.ndim != 2 or block_width <= 0:
        raise ExecutorInvariantError("invalid block-ID input")
    rows, columns = map(int, values.shape)
    blocks = math.ceil(columns / block_width)
    result = torch.empty((rows, blocks), dtype=torch.int32)
    cpu = values.detach().contiguous().cpu()
    for block in range(blocks):
        start = block * block_width
        stop = min(columns, start + block_width)
        table: dict[bytes, int] = {}
        for row in range(rows):
            key = tensor_bytes(cpu[row, start:stop])
            identifier = table.get(key)
            if identifier is None:
                identifier = len(table)
                table[key] = identifier
            result[row, block] = identifier
    return result


def _fiber_tree_q4_screen(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    block_width: int = 32,
    chunk_rows: int = 64,
) -> dict[str, Any]:
    """Favorable exact screen for a joint nonlinear-fiber Q4 surrogate tree."""
    intermediate = int(gate_weight.shape[0])
    if intermediate > _FIBER_TREE_MAX_ROWS:
        return {
            "status": "SKIPPED_SIZE_GUARD",
            "fiber_count": intermediate,
            "maximum_supported_fibers": _FIBER_TREE_MAX_ROWS,
            "native_bf16_semantics": "NOT_ESTABLISHED",
        }

    gate_q4 = _symmetric_row_q4(gate_weight)
    up_q4 = _symmetric_row_q4(up_weight)
    down_q4 = _symmetric_row_q4(down_weight)
    fibers = torch.cat((gate_q4, up_q4, down_q4.transpose(0, 1).contiguous()), dim=1)
    vertices = torch.cat((torch.zeros((1, fibers.shape[1]), dtype=torch.int8), fibers), dim=0)
    block_ids = _collision_free_block_ids(vertices, block_width)
    vertex_count, block_count = map(int, block_ids.shape)
    nearest = torch.full((vertex_count,), block_count + 1, dtype=torch.int32)

    for start in range(0, vertex_count, chunk_rows):
        stop = min(vertex_count, start + chunk_rows)
        distance = (block_ids[start:stop, None, :] != block_ids[None, :, :]).sum(dim=-1)
        local_rows = torch.arange(stop - start)
        global_rows = torch.arange(start, stop)
        distance[local_rows, global_rows] = block_count + 1
        nearest[start:stop] = distance.min(dim=1).values.to(torch.int32)

    lower_bound = (int(nearest.sum().item()) + 1) // 2
    dense_coefficients = int(fibers.numel())
    fraction = lower_bound / dense_coefficients if dense_coefficients else 0.0
    nearest_float = nearest.float()
    return {
        "status": "EXECUTED",
        "semantic_scope": "symmetric-row-Q4 integer surrogate only",
        "block_width": int(block_width),
        "fiber_count": int(fibers.shape[0]),
        "fiber_width": int(fibers.shape[1]),
        "block_count": block_count,
        "nearest_block_distance_min": int(nearest.min().item()),
        "nearest_block_distance_p50": float(torch.quantile(nearest_float, 0.5).item()),
        "nearest_block_distance_p95": float(torch.quantile(nearest_float, 0.95).item()),
        "nearest_block_distance_max": int(nearest.max().item()),
        "certified_tree_coefficient_lower_bound": lower_bound,
        "dense_fiber_coefficients": dense_coefficients,
        "certified_fraction": fraction,
        "target_equivalent_fraction": TARGET_EQUIVALENT_FRACTION,
        "passes_target_fraction": bool(fraction <= TARGET_EQUIVALENT_FRACTION),
        "native_bf16_semantics": "NOT_ESTABLISHED",
    }


def analyze_whole_swiglu(
    mlp: nn.Module,
    *,
    run_fiber_tree_screen: bool = True,
    fiber_tree_block_width: int = 32,
) -> dict[str, Any]:
    """Build a proof-carrying structural audit for a complete SwiGLU MLP."""
    for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        if not hasattr(mlp, name):
            raise ExecutorInvariantError(f"reference MLP missing {name}")

    gate = mlp.gate_proj.weight.detach()
    up = mlp.up_proj.weight.detach()
    down = mlp.down_proj.weight.detach()
    if gate.ndim != 2 or up.shape != gate.shape or down.ndim != 2:
        raise ExecutorInvariantError("unsupported SwiGLU projection shapes")
    intermediate, hidden = map(int, gate.shape)
    if tuple(map(int, down.shape)) != (hidden, intermediate):
        raise ExecutorInvariantError("down projection is not transpose-compatible with gate/up")

    gate_digests = _row_digests(gate)
    up_digests = _row_digests(up)
    down_column_digests = _row_digests(down.transpose(0, 1).contiguous())
    gate_summary = _duplicate_summary(gate_digests)
    up_summary = _duplicate_summary(up_digests)
    down_summary = _duplicate_summary(down_column_digests)
    pair_digests = [hashlib.sha256((g + ":" + u).encode()).hexdigest() for g, u in zip(gate_digests, up_digests)]
    triple_digests = [
        hashlib.sha256((g + ":" + u + ":" + d).encode()).hexdigest()
        for g, u, d in zip(gate_digests, up_digests, down_column_digests)
    ]
    total_coefficients = int(gate.numel() + up.numel() + down.numel())
    identity_cse_coefficients = gate_summary["unique"] * hidden + up_summary["unique"] * hidden + int(down.numel())

    result: dict[str, Any] = {
        "ir_version": IR_VERSION,
        "semantic_function": "down(silu(gate(x)) * up(x))",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "dtype": dtype_name(gate.dtype),
        "total_coefficients": total_coefficients,
        "gate_rows": gate_summary,
        "up_rows": up_summary,
        "gate_up_functions": _duplicate_summary(pair_digests),
        "down_columns": down_summary,
        "complete_nonlinear_fibers": _duplicate_summary(triple_digests),
        "gate_antipodal_pairs": _antipodal_pair_count(gate),
        "up_antipodal_pairs": _antipodal_pair_count(up),
        "down_column_antipodal_pairs": _antipodal_pair_count(down.transpose(0, 1).contiguous()),
        "native_safe_rewrites": [
            "jointly materialize gate/up fibers",
            "use one stacked affine call only after a frozen-ABI bitwise guard",
            "stream complete down output rows without splitting its reduction axis",
            "reuse only byte-identical affine forms when present",
        ],
        "identity_cse_optimistic_coefficient_fraction": identity_cse_coefficients / total_coefficients if total_coefficients else 0.0,
        "target_equivalent_fraction": TARGET_EQUIVALENT_FRACTION,
        "compiler_claim_boundary": (
            "Executable lowering is exact only for the frozen native ABI after compile probes and consecutive transition replay. "
            "The Q4 fiber-tree result is a search screen, not a BF16 proof."
        ),
    }
    result["q4_nonlinear_fiber_tree_screen"] = (
        _fiber_tree_q4_screen(gate, up, down, block_width=fiber_tree_block_width)
        if run_fiber_tree_screen
        else {"status": "NOT_RUN"}
    )
    return result


class ExactCompiledSwiGLU(nn.Module):
    """Whole-SwiGLU exact IR lowered to existing PyTorch operations."""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        records: Mapping[str, TensorRecord],
        gate_up_tiles: tuple[tuple[str, Optional[str], int], ...],
        down_tiles: tuple[tuple[str, Optional[str]], ...],
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        pretraining_tp: int,
        fiber_tile_rows: int,
        output_tile_rows: int,
        compiler_audit: Mapping[str, Any],
        joint_affine_mode: str = "stacked_single_call",
        verify_integrity: bool = True,
    ) -> None:
        super().__init__()
        if int(pretraining_tp) != 1:
            raise ExecutorInvariantError("global SwiGLU compiler fails closed unless pretraining_tp == 1")
        if int(fiber_tile_rows) <= 0 or int(output_tile_rows) <= 0:
            raise ExecutorInvariantError("tile rows must be positive")
        if joint_affine_mode not in {"stacked_single_call", "shared_materialization_split_calls"}:
            raise ExecutorInvariantError(f"invalid joint affine mode: {joint_affine_mode}")
        self.artifact_dir = str(artifact_dir)
        self.records = dict(records)
        self.gate_up_tiles = tuple(gate_up_tiles)
        self.down_tiles = tuple(down_tiles)
        self.act_fn = act_fn
        self.pretraining_tp = int(pretraining_tp)
        self.fiber_tile_rows = int(fiber_tile_rows)
        self.output_tile_rows = int(output_tile_rows)
        self.compiler_audit = dict(compiler_audit)
        self.joint_affine_mode = joint_affine_mode
        self.verify_integrity = bool(verify_integrity)
        self.reset_stats()

    @classmethod
    def from_reference(
        cls,
        mlp: nn.Module,
        *,
        artifact_dir: Path,
        fiber_tile_rows: int = DEFAULT_FIBER_TILE_ROWS,
        output_tile_rows: int = DEFAULT_OUTPUT_TILE_ROWS,
        verify_integrity: bool = True,
        run_fiber_tree_screen: bool = True,
    ) -> "ExactCompiledSwiGLU":
        for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
            if not hasattr(mlp, name):
                raise ExecutorInvariantError(f"reference MLP missing {name}")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        if gate.weight.shape != up.weight.shape:
            raise ExecutorInvariantError("gate/up shapes must match")
        if (gate.bias is None) != (up.bias is None):
            raise ExecutorInvariantError("gate/up bias presence must match for joint lowering")

        records: dict[str, TensorRecord] = {}
        gate_up_tiles: list[tuple[str, Optional[str], int]] = []
        intermediate = int(gate.weight.shape[0])
        for start in range(0, intermediate, int(fiber_tile_rows)):
            stop = min(intermediate, start + int(fiber_tile_rows))
            rows = stop - start
            weight_key = f"gate_up.weight.f{start:06d}-{stop:06d}"
            records[weight_key] = _pack_tensor(torch.cat((gate.weight[start:stop], up.weight[start:stop]), dim=0), artifact_dir, weight_key)
            bias_key: Optional[str] = None
            if gate.bias is not None:
                bias_key = f"gate_up.bias.f{start:06d}-{stop:06d}"
                records[bias_key] = _pack_tensor(torch.cat((gate.bias[start:stop], up.bias[start:stop]), dim=0), artifact_dir, bias_key)
            gate_up_tiles.append((weight_key, bias_key, rows))

        down_tiles: list[tuple[str, Optional[str]]] = []
        output_rows = int(down.weight.shape[0])
        for start in range(0, output_rows, int(output_tile_rows)):
            stop = min(output_rows, start + int(output_tile_rows))
            weight_key = f"down_proj.weight.r{start:06d}-{stop:06d}"
            records[weight_key] = _pack_tensor(down.weight[start:stop], artifact_dir, weight_key)
            bias_key = None
            if down.bias is not None:
                bias_key = f"down_proj.bias.r{start:06d}-{stop:06d}"
                records[bias_key] = _pack_tensor(down.bias[start:stop], artifact_dir, bias_key)
            down_tiles.append((weight_key, bias_key))

        return cls(
            artifact_dir=artifact_dir,
            records=records,
            gate_up_tiles=tuple(gate_up_tiles),
            down_tiles=tuple(down_tiles),
            act_fn=mlp.act_fn,
            pretraining_tp=int(getattr(getattr(mlp, "config", None), "pretraining_tp", 1)),
            fiber_tile_rows=int(fiber_tile_rows),
            output_tile_rows=int(output_tile_rows),
            compiler_audit=analyze_whole_swiglu(mlp, run_fiber_tree_screen=run_fiber_tree_screen),
            joint_affine_mode="stacked_single_call",
            verify_integrity=verify_integrity,
        )

    def reset_stats(self) -> None:
        self._stats = {
            key: 0
            for key in (
                "cold_bytes", "hot_materialized_bytes", "pcie_bytes", "gpu_bytes",
                "decompression_ns", "materialization_ns", "projection_calls",
                "integrity_probes", "peak_projection_hot_bytes", "joint_gate_up_calls", "down_calls",
            )
        }

    def stats_snapshot(self) -> dict[str, int]:
        return dict(self._stats)

    def _materialize(self, logical_name: str, device: torch.device) -> torch.Tensor:
        record = self.records[logical_name]
        start = time.perf_counter_ns()
        payload = (Path(self.artifact_dir) / record.relative_path).read_bytes()
        self._stats["cold_bytes"] += len(payload)
        if self.verify_integrity:
            self._stats["integrity_probes"] += 1
            if sha256_bytes(payload) != record.artifact_sha256:
                raise ArtifactIntegrityError(f"artifact payload hash mismatch: {logical_name}")
        decompression_start = time.perf_counter_ns()
        if record.codec == "zlib-9":
            raw = zlib.decompress(payload)
        elif record.codec == "raw":
            raw = payload
        else:
            raise ArtifactIntegrityError(f"unknown codec: {record.codec}")
        self._stats["decompression_ns"] += time.perf_counter_ns() - decompression_start
        if len(raw) != record.raw_bytes:
            raise ArtifactIntegrityError(f"raw byte length mismatch: {logical_name}")
        if self.verify_integrity:
            self._stats["integrity_probes"] += 1
            if sha256_bytes(raw) != record.raw_sha256:
                raise ArtifactIntegrityError(f"raw tensor hash mismatch: {logical_name}")
        dtype = DTYPE_BY_NAME.get(record.dtype)
        if dtype is None:
            raise ExecutorInvariantError(f"unsupported packed dtype: {record.dtype}")
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(record.shape)
        if tensor.device != device:
            tensor = tensor.to(device)
            if device.type == "cuda":
                self._stats["pcie_bytes"] += record.raw_bytes
        self._stats["hot_materialized_bytes"] += record.raw_bytes
        self._stats["peak_projection_hot_bytes"] = max(self._stats["peak_projection_hot_bytes"], record.raw_bytes)
        if device.type == "cuda":
            self._stats["gpu_bytes"] += record.raw_bytes
        self._stats["materialization_ns"] += time.perf_counter_ns() - start
        return tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nonlinear_tiles: list[torch.Tensor] = []
        for weight_key, bias_key, rows in self.gate_up_tiles:
            weight = self._materialize(weight_key, x.device)
            bias = self._materialize(bias_key, x.device) if bias_key is not None else None
            self._stats["joint_gate_up_calls"] += 1
            if self.joint_affine_mode == "stacked_single_call":
                self._stats["projection_calls"] += 1
                projected = F.linear(x, weight, bias)
                gate, up = projected.split((rows, rows), dim=-1)
            else:
                gate_weight, up_weight = weight.split((rows, rows), dim=0)
                gate_bias, up_bias = (None, None) if bias is None else bias.split((rows, rows), dim=0)
                self._stats["projection_calls"] += 2
                gate = F.linear(x, gate_weight, gate_bias)
                up = F.linear(x, up_weight, up_bias)
            nonlinear_tiles.append(self.act_fn(gate) * up)
        if not nonlinear_tiles:
            raise ExecutorInvariantError("empty global SwiGLU fiber program")
        intermediate = torch.cat(nonlinear_tiles, dim=-1)

        outputs: list[torch.Tensor] = []
        for weight_key, bias_key in self.down_tiles:
            weight = self._materialize(weight_key, x.device)
            bias = self._materialize(bias_key, x.device) if bias_key is not None else None
            self._stats["projection_calls"] += 1
            self._stats["down_calls"] += 1
            outputs.append(F.linear(intermediate, weight, bias))
        if not outputs:
            raise ExecutorInvariantError("empty global SwiGLU output program")
        return torch.cat(outputs, dim=-1)


def _abi_probe_inputs(weight: torch.Tensor) -> list[torch.Tensor]:
    hidden = int(weight.shape[1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260819)
    probes = [
        torch.zeros((1, 1, hidden), dtype=weight.dtype),
        torch.ones((1, 1, hidden), dtype=weight.dtype),
        -torch.ones((1, 1, hidden), dtype=weight.dtype),
    ]
    for _ in range(16):
        probes.append(torch.randn((1, 1, hidden), generator=generator, dtype=torch.float32).to(weight.dtype))
    return [probe.to(weight.device) for probe in probes]


def _run_compile_abi_guard(reference_mlp: nn.Module, compiled_mlp: ExactCompiledSwiGLU) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, probe in enumerate(_abi_probe_inputs(reference_mlp.gate_proj.weight)):
            expected = reference_mlp(probe)
            actual = compiled_mlp(probe)
            rows.append({
                "index": index,
                "input_sha256": sha256_bytes(tensor_bytes(probe)),
                "reference_sha256": sha256_bytes(tensor_bytes(expected)),
                "candidate_sha256": sha256_bytes(tensor_bytes(actual)),
                "bitwise_equal": bool(torch.equal(expected, actual)),
            })
    compiled_mlp.reset_stats()
    return {
        "mode": compiled_mlp.joint_affine_mode,
        "probe_count": len(rows),
        "all_bitwise_equal": all(row["bitwise_equal"] for row in rows),
        "rows": rows,
    }


def _select_joint_affine_mode(reference_mlp: nn.Module, compiled_mlp: ExactCompiledSwiGLU) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    weight_dtype = reference_mlp.gate_proj.weight.dtype
    if weight_dtype == torch.bfloat16:
        compiled_mlp.joint_affine_mode = "stacked_single_call"
        stacked = _run_compile_abi_guard(reference_mlp, compiled_mlp)
        attempts.append(stacked)
        selected = "stacked_single_call" if stacked["all_bitwise_equal"] else "shared_materialization_split_calls"
    else:
        selected = "shared_materialization_split_calls"
        attempts.append({
            "mode": "stacked_single_call",
            "status": "NOT_ELIGIBLE_NON_BF16_ABI",
            "all_bitwise_equal": False,
            "probe_count": 0,
            "rows": [],
        })
    if selected == "shared_materialization_split_calls":
        compiled_mlp.joint_affine_mode = selected
        split = _run_compile_abi_guard(reference_mlp, compiled_mlp)
        attempts.append(split)
        if not split["all_bitwise_equal"]:
            raise ExecutorInvariantError("global SwiGLU compiler failed proof-preserving native-ABI lowering")
    compiled_mlp.joint_affine_mode = selected
    compiled_mlp.reset_stats()
    return {
        "weight_dtype": dtype_name(weight_dtype),
        "selected_joint_affine_mode": selected,
        "attempts": attempts,
        "all_selected_mode_probes_bitwise_equal": True,
        "claim_boundary": (
            "Frozen-host compile probes plus mandatory held-out 128-transition replay; "
            "stacked BF16 is not promoted as a universal all-input theorem."
        ),
    }


def compile_checkpoint_global_swiglu(
    reference_model: nn.Module,
    *,
    artifact_dir: Optional[str | Path] = None,
    layer_index: int = 0,
    fiber_tile_rows: int = DEFAULT_FIBER_TILE_ROWS,
    output_tile_rows: int = DEFAULT_OUTPUT_TILE_ROWS,
    verify_integrity: bool = True,
    run_fiber_tree_screen: bool = True,
) -> CompiledArtifact:
    start = time.perf_counter_ns()
    layers = getattr(getattr(reference_model, "model", None), "layers", None)
    if layers is None or not 0 <= int(layer_index) < len(layers):
        raise ExecutorInvariantError("valid model.model.layers and layer index are required")
    model_id, revision = resolved_model_identity(reference_model)
    config_sha = json_sha256(config_dict(getattr(reference_model, "config", None)))
    checkpoint_hash = model_tensor_sha256(reference_model)
    root = Path(artifact_dir) if artifact_dir is not None else Path(tempfile.mkdtemp(prefix="exact-swiglu-global-compiler-"))
    root.mkdir(parents=True, exist_ok=True)

    original = layers[int(layer_index)]
    original_bytes = module_parameter_bytes(original)
    compiled_mlp = ExactCompiledSwiGLU.from_reference(
        original.mlp,
        artifact_dir=root,
        fiber_tile_rows=int(fiber_tile_rows),
        output_tile_rows=int(output_tile_rows),
        verify_integrity=verify_integrity,
        run_fiber_tree_screen=run_fiber_tree_screen,
    )
    abi_guard = _select_joint_affine_mode(original.mlp, compiled_mlp)
    replacement = ExactReplacementDecoderLayer(original, compiled_mlp)
    layers[int(layer_index)] = replacement

    manifest = {
        "format": "fixed-public-dynamic-executor-artifact-v1",
        "ir_version": IR_VERSION,
        "model_id": model_id,
        "revision": revision,
        "model_class": reference_model.__class__.__name__,
        "layer_index": int(layer_index),
        "config_sha256": config_sha,
        "checkpoint_tensor_sha256": checkpoint_hash,
        "mechanism": MECHANISM,
        "semantic_graph": [
            {
                "node": "joint_gate_up_affine_tiles",
                "fiber_tile_rows": int(fiber_tile_rows),
                "selected_mode": compiled_mlp.joint_affine_mode,
                "lowering": "torch.nn.functional.linear",
            },
            {"node": "silu_multiply", "lowering": "act_fn(gate) * up"},
            {
                "node": "down_output_row_tiles",
                "output_tile_rows": int(output_tile_rows),
                "reduction_axis_partitioned": False,
                "lowering": "torch.nn.functional.linear",
            },
        ],
        "tensor_records": [asdict(record) for _, record in sorted(compiled_mlp.records.items())],
        "gate_up_tiles": [[weight_key, bias_key, rows] for weight_key, bias_key, rows in compiled_mlp.gate_up_tiles],
        "down_tiles": [[weight_key, bias_key] for weight_key, bias_key in compiled_mlp.down_tiles],
        "compiler_audit": compiled_mlp.compiler_audit,
        "native_abi_guard": abi_guard,
        "exactness_contract": {
            "pretraining_tp": compiled_mlp.pretraining_tp,
            "verify_integrity": bool(verify_integrity),
            "arithmetic_order": "down(act(gate(x))*up(x))",
            "down_dot_product_reduction_axis_split": False,
            "successor_state_validation": "mandatory 128-transition external gate",
        },
    }
    manifest_path = root / "manifest.json"
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode()
    manifest_path.write_bytes(manifest_bytes)
    artifact = CompiledArtifact(
        reference_model,
        model_id,
        revision,
        int(layer_index),
        str(root),
        str(manifest_path),
        sha256_bytes(manifest_bytes),
        directory_bytes(root),
        0,
        rss_bytes(),
        checkpoint_hash,
        config_sha,
        original_bytes,
        module_parameter_bytes(replacement),
        replacement,
        MECHANISM,
    )
    artifact.compile_wall_ns = time.perf_counter_ns() - start
    del original
    gc.collect()
    return artifact
