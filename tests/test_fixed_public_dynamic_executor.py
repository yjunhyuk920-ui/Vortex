import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vortex_runtime import fixed_public_dynamic_executor as f
from vortex_runtime.fixed_public_dynamic_audit import audit_checkpoint_layer
from vortex_runtime.fixed_public_dynamic_streaming import compile_checkpoint_row_streamed


class FakeConfig:
    _name_or_path = "unit/fake-public-checkpoint"
    _commit_hash = "0123456789abcdef0123456789abcdef01234567"
    pretraining_tp = 1

    def to_dict(self):
        return {"model_type": "llama", "hidden_size": 8, "intermediate_size": 16, "pretraining_tp": 1}


class FakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FakeConfig()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        return self.proj(hidden_states), None, past_key_value


class FakeLayer(nn.Module):
    hidden_size = 8

    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()
        self.input_layernorm = nn.LayerNorm(8)
        self.post_attention_layernorm = nn.LayerNorm(8)
        self.mlp = FakeMLP()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn, present = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        out = (hidden_states,)
        if output_attentions:
            out += (attn,)
        if use_cache:
            out += (present,)
        return out


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(17, 8)
        self.layers = nn.ModuleList([FakeLayer()])


class FakeCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FakeConfig()
        self.model = FakeBackbone()
        self.lm_head = nn.Linear(8, 17, bias=False)

    def forward(self, input_ids, past_key_values=None, use_cache=True, return_dict=True):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x, use_cache=False)[0]
        logits = self.lm_head(x)
        if past_key_values is None:
            history = input_ids.detach().clone()
        else:
            history = torch.cat([past_key_values, input_ids.detach().clone()], dim=1)
        return SimpleNamespace(logits=logits, past_key_values=history)


def seeded_model():
    torch.manual_seed(20260817)
    return FakeCausalLM().eval()


def test_full_layer_replacement_is_exact_and_removes_mlp_parameters(tmp_path):
    reference = seeded_model()
    candidate = copy.deepcopy(reference)
    x = torch.randn(1, 3, 8)
    expected = reference.model.layers[0](x, use_cache=False)[0]
    artifact = f.compile_checkpoint(candidate, artifact_dir=tmp_path, layer_index=0)
    actual = candidate.model.layers[0](x, use_cache=False)[0]
    assert torch.equal(actual, expected)
    assert artifact.replaced_layer_reference_parameter_bytes > artifact.compiled_layer_resident_parameter_bytes
    names = [name for name, _ in candidate.model.layers[0].named_parameters()]
    assert not any("mlp.gate_proj" in name or "mlp.up_proj" in name or "mlp.down_proj" in name for name in names)
    assert Path(artifact.manifest_path).is_file()


def test_dynamic_executor_128_steps_and_successor_state_exact(tmp_path):
    reference = seeded_model()
    candidate = copy.deepcopy(reference)
    ref_artifact = f.make_reference_artifact(reference)
    cand_artifact = f.compile_checkpoint(candidate, artifact_dir=tmp_path, layer_index=0)
    prefix = torch.tensor([[1, 2, 3]])
    ref_state = f.initialize_state(ref_artifact, prefix)
    cand_state = f.initialize_state(cand_artifact, prefix)
    token = 4
    for _ in range(128):
        ref_token, ref_state, _ = f.decode_step(ref_artifact, ref_state, token)
        cand_token, cand_state, trace = f.decode_step(cand_artifact, cand_state, token)
        assert cand_token == ref_token
        assert f.states_exact(cand_state, ref_state)
        assert trace.projection_calls == 3
        assert trace.integrity_probes == 6
        token = ref_token


def test_row_streamed_full_layer_exact_and_peak_hot_bounded(tmp_path):
    reference = seeded_model()
    candidate = copy.deepcopy(reference)
    x = torch.randn(1, 3, 8)
    expected = reference.model.layers[0](x, use_cache=False)[0]
    artifact = compile_checkpoint_row_streamed(candidate, artifact_dir=tmp_path, layer_index=0, output_tile_rows=4)
    actual = candidate.model.layers[0](x, use_cache=False)[0]
    assert torch.equal(actual, expected)
    assert artifact.mechanism == "checkpoint_mlp_output_row_streamed_lossless_existing_isa"
    assert artifact.replaced_layer_reference_parameter_bytes > artifact.compiled_layer_resident_parameter_bytes
    records = artifact.runtime_layer.mlp.records
    assert len(records) == 10
    assert max(record.raw_bytes for record in records.values()) <= 4 * 16 * 4
    manifest = json.loads(Path(artifact.manifest_path).read_text())
    assert manifest["streaming"]["axis"] == "output_rows"
    assert manifest["streaming"]["output_tile_rows"] == 4
    assert manifest["streaming"]["reduction_axis_partitioned"] is False


def test_row_streamed_dynamic_executor_128_steps_and_successor_state_exact(tmp_path):
    reference = seeded_model()
    candidate = copy.deepcopy(reference)
    ref_artifact = f.make_reference_artifact(reference)
    cand_artifact = compile_checkpoint_row_streamed(candidate, artifact_dir=tmp_path, layer_index=0, output_tile_rows=4)
    prefix = torch.tensor([[1, 2, 3]])
    ref_state = f.initialize_state(ref_artifact, prefix)
    cand_state = f.initialize_state(cand_artifact, prefix)
    token = 4
    for _ in range(128):
        ref_token, ref_state, _ = f.decode_step(ref_artifact, ref_state, token)
        cand_token, cand_state, trace = f.decode_step(cand_artifact, cand_state, token)
        assert cand_token == ref_token
        assert f.states_exact(cand_state, ref_state)
        assert trace.projection_calls == 10
        assert trace.integrity_probes == 20
        assert trace.hot_materialized_bytes > trace.peak_vram_bytes if trace.peak_vram_bytes is not None else True
        token = ref_token


def test_sampling_rng_state_is_part_of_successor_state(tmp_path):
    reference = seeded_model()
    candidate = copy.deepcopy(reference)
    ref_artifact = f.make_reference_artifact(reference)
    cand_artifact = f.compile_checkpoint(candidate, artifact_dir=tmp_path, layer_index=0)
    rng = f.RNGState.sampled(12345, top_k=0, top_p=1.0)
    prefix = torch.tensor([[1, 2]])
    ref_state = f.initialize_state(ref_artifact, prefix, rng_state=rng)
    cand_state = f.initialize_state(cand_artifact, prefix, rng_state=rng)
    token = 3
    for _ in range(16):
        r, ref_state, _ = f.decode_step(ref_artifact, ref_state, token)
        c, cand_state, _ = f.decode_step(cand_artifact, cand_state, token)
        assert c == r
        assert f.states_exact(cand_state, ref_state)
        token = r


def test_corrupted_artifact_fails_closed(tmp_path):
    candidate = seeded_model()
    artifact = f.compile_checkpoint(candidate, artifact_dir=tmp_path, layer_index=0)
    record = artifact.runtime_layer.mlp.records["gate_proj.weight"]
    path = Path(artifact.artifact_dir) / record.relative_path
    payload = bytearray(path.read_bytes())
    payload[0] ^= 0x01
    path.write_bytes(payload)
    with pytest.raises(f.ArtifactIntegrityError):
        artifact.runtime_layer.mlp(torch.randn(1, 1, 8))


def test_corrupted_row_streamed_artifact_fails_closed(tmp_path):
    candidate = seeded_model()
    artifact = compile_checkpoint_row_streamed(candidate, artifact_dir=tmp_path, layer_index=0, output_tile_rows=4)
    first_key = sorted(artifact.runtime_layer.mlp.records)[0]
    record = artifact.runtime_layer.mlp.records[first_key]
    path = Path(artifact.artifact_dir) / record.relative_path
    payload = bytearray(path.read_bytes())
    payload[0] ^= 0x01
    path.write_bytes(payload)
    with pytest.raises(f.ArtifactIntegrityError):
        artifact.runtime_layer.mlp(torch.randn(1, 1, 8))


def test_actual_audit_schema_and_conservative_physical_gate():
    model = seeded_model()
    audit = audit_checkpoint_layer(model.model.layers[0], layer_index=0)
    assert audit["audited_from_actual_parameters"] is True
    assert audit["tensor_count"] > 0
    first = audit["tensors"][0]
    for key in ("exact_block_duplicate", "exact_coefficient_dictionary", "bit_plane_circuit", "grammar_dictionary"):
        assert key in first
    gate = f.physical_gate(
        baseline_p50_ns=100,
        baseline_p95_ns=150,
        candidate_p50_ns=101,
        candidate_p95_ns=151,
        reference_layer_parameter_bytes=1000,
        artifact_bytes=900,
        candidate_layer_resident_parameter_bytes=300,
        peak_projection_hot_bytes=300,
    )
    assert gate["passed"] is False


def test_physical_gate_accepts_accounted_row_stream_footprint_saving():
    gate = f.physical_gate(
        baseline_p50_ns=100,
        baseline_p95_ns=150,
        candidate_p50_ns=200,
        candidate_p95_ns=300,
        reference_layer_parameter_bytes=7080192,
        artifact_bytes=4208296,
        candidate_layer_resident_parameter_bytes=1771776,
        peak_projection_hot_bytes=393216,
    )
    assert gate["passed"] is True
    assert gate["footprint_saved"] is True
    assert gate["latency_saved_p50_and_p95"] is False
