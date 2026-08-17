from __future__ import annotations

import gc
import json
import math
import tempfile
import time
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .fixed_public_dynamic_common import (
    ArtifactIntegrityError, CompiledArtifact, CompiledState, DTYPE_BY_NAME, ExecutorInvariantError,
    RNGState, ResourceTrace, TensorRecord, config_dict, dtype_name, json_sha256, model_tensor_sha256,
    module_parameter_bytes, resolved_model_identity, rss_bytes, sha256_bytes, tensor_bytes,
)


def _pack_tensor(tensor: torch.Tensor, artifact_dir: Path, logical_name: str) -> TensorRecord:
    raw = tensor_bytes(tensor)
    compressed = zlib.compress(raw, 9)
    codec, payload = ("zlib-9", compressed) if len(compressed) < len(raw) else ("raw", raw)
    rel = logical_name.replace(".", "__") + ".bin"
    (artifact_dir / rel).write_bytes(payload)
    return TensorRecord(logical_name, tuple(map(int, tensor.shape)), dtype_name(tensor.dtype), len(raw), len(payload), codec, rel, sha256_bytes(raw), sha256_bytes(payload))


class ExactColdPackedMLP(nn.Module):
    """Actual checkpoint MLP projections stored losslessly and materialized one projection at a time."""

    def __init__(self, *, artifact_dir: Path, records: Mapping[str, TensorRecord], act_fn: Callable[[torch.Tensor], torch.Tensor], pretraining_tp: int, verify_integrity: bool = True) -> None:
        super().__init__()
        if int(pretraining_tp) != 1:
            raise ExecutorInvariantError("cold-packed MLP fails closed unless pretraining_tp == 1")
        self.artifact_dir, self.records, self.act_fn = str(artifact_dir), dict(records), act_fn
        self.pretraining_tp, self.verify_integrity = int(pretraining_tp), bool(verify_integrity)
        self.reset_stats()

    @classmethod
    def from_reference(cls, mlp: nn.Module, *, artifact_dir: Path, verify_integrity: bool = True) -> "ExactColdPackedMLP":
        for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
            if not hasattr(mlp, name):
                raise ExecutorInvariantError(f"reference MLP missing {name}")
        records: dict[str, TensorRecord] = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(mlp, name)
            records[f"{name}.weight"] = _pack_tensor(proj.weight, artifact_dir, f"{name}.weight")
            if proj.bias is not None:
                records[f"{name}.bias"] = _pack_tensor(proj.bias, artifact_dir, f"{name}.bias")
        return cls(artifact_dir=artifact_dir, records=records, act_fn=mlp.act_fn, pretraining_tp=int(getattr(getattr(mlp, "config", None), "pretraining_tp", 1)), verify_integrity=verify_integrity)

    def reset_stats(self) -> None:
        self._stats = {k: 0 for k in ("cold_bytes", "hot_materialized_bytes", "pcie_bytes", "gpu_bytes", "decompression_ns", "materialization_ns", "projection_calls", "integrity_probes", "peak_projection_hot_bytes")}

    def stats_snapshot(self) -> dict[str, int]:
        return dict(self._stats)

    def _materialize(self, logical_name: str, device: torch.device) -> torch.Tensor:
        rec = self.records[logical_name]
        start = time.perf_counter_ns()
        payload = (Path(self.artifact_dir) / rec.relative_path).read_bytes()
        self._stats["cold_bytes"] += len(payload)
        if self.verify_integrity:
            self._stats["integrity_probes"] += 1
            if sha256_bytes(payload) != rec.artifact_sha256:
                raise ArtifactIntegrityError(f"artifact payload hash mismatch: {logical_name}")
        d0 = time.perf_counter_ns()
        if rec.codec == "zlib-9":
            raw = zlib.decompress(payload)
        elif rec.codec == "raw":
            raw = payload
        else:
            raise ArtifactIntegrityError(f"unknown codec: {rec.codec}")
        self._stats["decompression_ns"] += time.perf_counter_ns() - d0
        if len(raw) != rec.raw_bytes:
            raise ArtifactIntegrityError(f"raw byte length mismatch: {logical_name}")
        if self.verify_integrity:
            self._stats["integrity_probes"] += 1
            if sha256_bytes(raw) != rec.raw_sha256:
                raise ArtifactIntegrityError(f"raw tensor hash mismatch: {logical_name}")
        dtype = DTYPE_BY_NAME.get(rec.dtype)
        if dtype is None:
            raise ExecutorInvariantError(f"unsupported packed dtype: {rec.dtype}")
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(rec.shape)
        if tensor.device != device:
            tensor = tensor.to(device)
            if device.type == "cuda":
                self._stats["pcie_bytes"] += rec.raw_bytes
        self._stats["hot_materialized_bytes"] += rec.raw_bytes
        self._stats["peak_projection_hot_bytes"] = max(self._stats["peak_projection_hot_bytes"], rec.raw_bytes)
        if device.type == "cuda":
            self._stats["gpu_bytes"] += rec.raw_bytes
        self._stats["materialization_ns"] += time.perf_counter_ns() - start
        return tensor

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        weight = self._materialize(f"{name}.weight", x.device)
        bias = self._materialize(f"{name}.bias", x.device) if f"{name}.bias" in self.records else None
        self._stats["projection_calls"] += 1
        return F.linear(x, weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self._linear(x, "gate_proj"))
        up = self._linear(x, "up_proj")
        return self._linear(gate * up, "down_proj")


class ExactReplacementDecoderLayer(nn.Module):
    """Complete LlamaDecoderLayer boundary with the Transformers 4.46.3 eager arithmetic order."""

    def __init__(self, layer: nn.Module, mlp: nn.Module) -> None:
        super().__init__()
        for name in ("self_attn", "input_layernorm", "post_attention_layernorm", "hidden_size"):
            if not hasattr(layer, name):
                raise ExecutorInvariantError(f"reference decoder layer missing {name}")
        self.hidden_size = layer.hidden_size
        self.self_attn, self.input_layernorm, self.post_attention_layernorm, self.mlp = layer.self_attn, layer.input_layernorm, layer.post_attention_layernorm, mlp

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_value: Any = None, output_attentions: Optional[bool] = False, use_cache: Optional[bool] = False, cache_position: Optional[torch.LongTensor] = None, position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None, **kwargs: Any) -> tuple[Any, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn, present = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=bool(output_attentions), use_cache=bool(use_cache), cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        outputs: tuple[Any, ...] = (hidden_states,)
        if output_attentions:
            outputs += (attn,)
        if use_cache:
            outputs += (present,)
        return outputs


def compile_checkpoint(reference_model: nn.Module, *, artifact_dir: Optional[str | Path] = None, layer_index: int = 0, verify_integrity: bool = True) -> CompiledArtifact:
    start = time.perf_counter_ns()
    layers = getattr(getattr(reference_model, "model", None), "layers", None)
    if layers is None or not 0 <= int(layer_index) < len(layers):
        raise ExecutorInvariantError("valid model.model.layers and layer index are required")
    model_id, revision = resolved_model_identity(reference_model)
    config_sha, checkpoint_hash = json_sha256(config_dict(getattr(reference_model, "config", None))), model_tensor_sha256(reference_model)
    root = Path(artifact_dir) if artifact_dir is not None else Path(tempfile.mkdtemp(prefix="fixed-public-executor-"))
    root.mkdir(parents=True, exist_ok=True)
    original = layers[int(layer_index)]
    original_bytes = module_parameter_bytes(original)
    packed = ExactColdPackedMLP.from_reference(original.mlp, artifact_dir=root, verify_integrity=verify_integrity)
    replacement = ExactReplacementDecoderLayer(original, packed)
    layers[int(layer_index)] = replacement
    manifest = {"format": "fixed-public-dynamic-executor-artifact-v1", "model_id": model_id, "revision": revision, "model_class": reference_model.__class__.__name__, "layer_index": int(layer_index), "config_sha256": config_sha, "checkpoint_tensor_sha256": checkpoint_hash, "mechanism": "cold_packed_projection_sequential_materialization", "tensor_records": [asdict(v) for _, v in sorted(packed.records.items())], "exactness_contract": {"pretraining_tp": packed.pretraining_tp, "verify_integrity": bool(verify_integrity), "arithmetic_order": "down(act(gate(x))*up(x))"}}
    manifest_path = root / "manifest.json"
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode()
    manifest_path.write_bytes(manifest_bytes)
    artifact = CompiledArtifact(reference_model, model_id, revision, int(layer_index), str(root), str(manifest_path), sha256_bytes(manifest_bytes), len(manifest_bytes) + sum(r.artifact_bytes for r in packed.records.values()), 0, rss_bytes(), checkpoint_hash, config_sha, original_bytes, module_parameter_bytes(replacement), replacement)
    artifact.compile_wall_ns = time.perf_counter_ns() - start
    del original
    gc.collect()
    return artifact


def make_reference_artifact(model: nn.Module) -> CompiledArtifact:
    model_id, revision = resolved_model_identity(model)
    return CompiledArtifact(model, model_id, revision, -1, "", "", "", 0, 0, rss_bytes(), "NOT_COMPUTED_REFERENCE", json_sha256(config_dict(getattr(model, "config", None))), 0, 0, None, "reference")


def _token_ids(value: Any) -> torch.LongTensor:
    tensor = value.detach().to(torch.long) if isinstance(value, torch.Tensor) else torch.tensor([[value]], dtype=torch.long) if isinstance(value, int) else torch.tensor(value, dtype=torch.long)
    if tensor.ndim == 0: tensor = tensor.reshape(1, 1)
    if tensor.ndim == 1: tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] != 1: raise ExecutorInvariantError("batch size 1 is required")
    return tensor


def _device(model: nn.Module) -> torch.device:
    try: return next(model.parameters()).device
    except StopIteration: return torch.device("cpu")


def initialize_state(artifact: CompiledArtifact, prompt: Any, *, rng_state: Optional[RNGState] = None) -> CompiledState:
    ids = _token_ids(prompt)
    if ids.shape[1] == 0: return CompiledState(None, 0, 0, rng_state or RNGState.greedy(), sha256_bytes(b""))
    ids = ids.to(_device(artifact.model))
    with torch.inference_mode(): output = artifact.model(input_ids=ids, use_cache=True, return_dict=True)
    return CompiledState(output.past_key_values, int(ids.shape[1]), 0, rng_state or RNGState.greedy(), sha256_bytes(tensor_bytes(ids.cpu())))


def _select(logits: torch.Tensor, rng: RNGState) -> tuple[int, RNGState]:
    vector = logits.detach().reshape(-1)
    if rng.mode == "greedy": return int(torch.argmax(vector)), rng
    if rng.mode != "sample" or rng.temperature <= 0: raise ExecutorInvariantError(f"invalid RNG state: {rng.mode}")
    probs = torch.softmax(vector.float().cpu() / rng.temperature, dim=-1)
    if 0 < rng.top_k < probs.numel():
        threshold = torch.topk(probs, rng.top_k).values[-1]; probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
    if 0 < rng.top_p < 1:
        vals, idx = torch.sort(probs, descending=True); keep = torch.cumsum(vals, -1) - vals < rng.top_p; vals = torch.where(keep, vals, torch.zeros_like(vals)); filtered = torch.zeros_like(probs); filtered.scatter_(0, idx, vals); probs = filtered
    probs = probs / probs.sum()
    generator = torch.Generator(device="cpu"); generator.set_state(torch.tensor(list(rng.generator_state), dtype=torch.uint8))
    token = int(torch.multinomial(probs, 1, generator=generator))
    return token, RNGState("sample", bytes(generator.get_state().tolist()), rng.seed, rng.temperature, rng.top_k, rng.top_p)


def _cache_tensors(cache: Any) -> Iterable[torch.Tensor]:
    if cache is None: return
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for item in list(cache.key_cache) + list(cache.value_cache):
            if isinstance(item, torch.Tensor): yield item
    elif isinstance(cache, torch.Tensor): yield cache
    elif isinstance(cache, (tuple, list)):
        for item in cache: yield from _cache_tensors(item)


def cache_bytes(cache: Any) -> int:
    return sum(int(t.numel() * t.element_size()) for t in _cache_tensors(cache))


def state_sha256(state: CompiledState) -> str:
    import hashlib
    h = hashlib.sha256(); h.update(f"{state.position}:{state.step}:{state.prefix_sha256}:{state.rng_state.mode}".encode()); h.update(state.rng_state.generator_state)
    for tensor in _cache_tensors(state.past_key_values): h.update(str(tuple(tensor.shape)).encode()); h.update(dtype_name(tensor.dtype).encode()); h.update(tensor_bytes(tensor))
    return h.hexdigest()


def states_exact(a: CompiledState, b: CompiledState) -> bool:
    if a.position != b.position or a.step != b.step or a.rng_state != b.rng_state: return False
    at, bt = list(_cache_tensors(a.past_key_values)), list(_cache_tensors(b.past_key_values))
    return len(at) == len(bt) and all(x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y) for x, y in zip(at, bt))


def _flops(artifact: CompiledArtifact) -> int:
    config = getattr(artifact.model, "config", None); h = int(getattr(config, "hidden_size", 0) or 0); i = int(getattr(config, "intermediate_size", 0) or 0)
    return 2 * (2 * h * i + i * h) if artifact.runtime_layer is not None else 0


def decode_step(artifact: CompiledArtifact, compiled_state: CompiledState, input_token: Any, rng_state: Optional[RNGState] = None) -> tuple[int, CompiledState, ResourceTrace]:
    token = _token_ids(input_token)
    if token.shape[1] != 1: raise ExecutorInvariantError("decode_step consumes exactly one token")
    device, active_rng = _device(artifact.model), rng_state or compiled_state.rng_state
    token = token.to(device)
    runtime_mlp = artifact.runtime_layer.mlp if artifact.runtime_layer is not None else None
    if runtime_mlp is not None and hasattr(runtime_mlp, "reset_stats"): runtime_mlp.reset_stats()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter_ns()
    with torch.inference_mode(): output = artifact.model(input_ids=token, past_key_values=compiled_state.past_key_values, use_cache=True, return_dict=True)
    next_token, next_rng = _select(output.logits[:, -1, :], active_rng)
    wall = time.perf_counter_ns() - start
    nxt = CompiledState(output.past_key_values, compiled_state.position + 1, compiled_state.step + 1, next_rng, compiled_state.prefix_sha256)
    stats = runtime_mlp.stats_snapshot() if runtime_mlp is not None and hasattr(runtime_mlp, "stats_snapshot") else {}
    g = lambda k: int(stats.get(k, 0))
    trace = ResourceTrace(nxt.step, wall, g("cold_bytes"), g("hot_materialized_bytes"), g("pcie_bytes"), g("gpu_bytes"), cache_bytes(nxt.past_key_values), int(artifact.artifact_bytes), module_parameter_bytes(artifact.model), int(artifact.compiled_layer_resident_parameter_bytes), int(artifact.replaced_layer_reference_parameter_bytes), g("decompression_ns"), g("materialization_ns"), g("projection_calls"), g("integrity_probes"), _flops(artifact), rss_bytes(), int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, isa_status="NOT_MEASURED_NO_PERF_EVENT")
    return next_token, nxt, trace


def physical_gate(*, baseline_p50_ns: Optional[int], baseline_p95_ns: Optional[int], candidate_p50_ns: Optional[int], candidate_p95_ns: Optional[int], reference_layer_parameter_bytes: int, artifact_bytes: int, candidate_layer_resident_parameter_bytes: int, peak_projection_hot_bytes: int) -> dict[str, Any]:
    footprint = int(artifact_bytes) + int(candidate_layer_resident_parameter_bytes) + int(peak_projection_hot_bytes)
    footprint_saved = footprint < int(reference_layer_parameter_bytes)
    latency_saved = None not in (baseline_p50_ns, baseline_p95_ns, candidate_p50_ns, candidate_p95_ns) and candidate_p50_ns < baseline_p50_ns and candidate_p95_ns < baseline_p95_ns
    return {"passed": bool(footprint_saved or latency_saved), "reference_layer_parameter_bytes": int(reference_layer_parameter_bytes), "compiled_artifact_plus_resident_plus_peak_hot_bytes": footprint, "footprint_saved": bool(footprint_saved), "latency_saved_p50_and_p95": bool(latency_saved), "baseline_p50_ns": baseline_p50_ns, "baseline_p95_ns": baseline_p95_ns, "candidate_p50_ns": candidate_p50_ns, "candidate_p95_ns": candidate_p95_ns, "rule": "PASS only if total accounted physical footprint shrinks OR both p50 and p95 wall time shrink"}
