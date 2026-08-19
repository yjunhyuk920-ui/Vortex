from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vortex_runtime import fixed_public_dynamic_executor as executor
from vortex_runtime.exact_swiglu_program import (
    IR_VERSION,
    MECHANISM,
    analyze_whole_swiglu,
    compile_checkpoint_global_swiglu,
)


class FakeConfig:
    _name_or_path = "unit/fake-public-checkpoint"
    _commit_hash = "0123456789abcdef0123456789abcdef01234567"
    pretraining_tp = 1

    def to_dict(self):
        return {
            "model_type": "llama",
            "hidden_size": 8,
            "intermediate_size": 16,
            "pretraining_tp": 1,
        }


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
        hidden_states = residual + self.mlp(hidden_states)
        result = (hidden_states,)
        if output_attentions:
            result += (attn,)
        if use_cache:
            result += (present,)
        return result


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
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden, use_cache=False)[0]
        logits = self.lm_head(hidden)
        history = (
            input_ids.detach().clone()
            if past_key_values is None
            else torch.cat((past_key_values, input_ids.detach().clone()), dim=1)
        )
        return SimpleNamespace(logits=logits, past_key_values=history)


def seeded_model(*, dtype=torch.float32):
    torch.manual_seed(20260819)
    return FakeCausalLM().eval().to(dtype=dtype)


def test_float32_selects_proof_preserving_split_lowering(tmp_path):
    reference = seeded_model()
    candidate = copy.deepcopy(reference)
    value = torch.randn(1, 1, 8)
    expected = reference.model.layers[0](value, use_cache=False)[0]
    artifact = compile_checkpoint_global_swiglu(
        candidate,
        artifact_dir=tmp_path,
        layer_index=0,
        fiber_tile_rows=4,
        output_tile_rows=4,
    )
    actual = candidate.model.layers[0](value, use_cache=False)[0]
    manifest = json.loads(Path(artifact.manifest_path).read_text())
    assert torch.equal(actual, expected)
    assert artifact.mechanism == MECHANISM
    assert manifest["ir_version"] == IR_VERSION
    assert manifest["native_abi_guard"]["selected_joint_affine_mode"] == "shared_materialization_split_calls"
    assert manifest["semantic_graph"][2]["reduction_axis_partitioned"] is False
    assert artifact.runtime_layer.mlp.stats_snapshot()["projection_calls"] == 10
    assert artifact.runtime_layer.mlp.stats_snapshot()["integrity_probes"] == 12


def test_bf16_lowering_is_bitwise_guarded(tmp_path):
    reference = seeded_model(dtype=torch.bfloat16)
    candidate = copy.deepcopy(reference)
    value = torch.randn(2, 3, 8).to(torch.bfloat16)
    expected = reference.model.layers[0](value, use_cache=False)[0]
    artifact = compile_checkpoint_global_swiglu(
        candidate,
        artifact_dir=tmp_path,
        layer_index=0,
        fiber_tile_rows=4,
        output_tile_rows=4,
    )
    actual = candidate.model.layers[0](value, use_cache=False)[0]
    manifest = json.loads(Path(artifact.manifest_path).read_text())
    selected = manifest["native_abi_guard"]["selected_joint_affine_mode"]
    assert torch.equal(actual, expected)
    assert selected in {"stacked_single_call", "shared_materialization_split_calls"}
    assert manifest["native_abi_guard"]["all_selected_mode_probes_bitwise_equal"] is True


def test_128_dynamic_transitions_preserve_token_and_successor_state(tmp_path):
    reference = seeded_model()
    candidate = copy.deepcopy(reference)
    reference_artifact = executor.make_reference_artifact(reference)
    candidate_artifact = compile_checkpoint_global_swiglu(
        candidate,
        artifact_dir=tmp_path,
        layer_index=0,
        fiber_tile_rows=4,
        output_tile_rows=4,
    )
    prefix = torch.tensor([[1, 2, 3]])
    reference_state = executor.initialize_state(reference_artifact, prefix)
    candidate_state = executor.initialize_state(candidate_artifact, prefix)
    token = 4
    for _ in range(128):
        reference_token, reference_state, _ = executor.decode_step(reference_artifact, reference_state, token)
        candidate_token, candidate_state, trace = executor.decode_step(candidate_artifact, candidate_state, token)
        assert candidate_token == reference_token
        assert executor.states_exact(candidate_state, reference_state)
        assert trace.projection_calls == 10
        assert trace.integrity_probes == 12
        token = reference_token


def test_compiler_audit_finds_exact_nonlinear_fiber_symmetry():
    mlp = FakeMLP().eval()
    with torch.no_grad():
        mlp.gate_proj.weight[1].copy_(mlp.gate_proj.weight[0])
        mlp.up_proj.weight[1].copy_(mlp.up_proj.weight[0])
        mlp.down_proj.weight[:, 1].copy_(mlp.down_proj.weight[:, 0])
        mlp.gate_proj.weight[3].copy_(-mlp.gate_proj.weight[2])
    audit = analyze_whole_swiglu(mlp)
    assert audit["gate_rows"]["duplicate_rows"] >= 1
    assert audit["gate_up_functions"]["duplicate_rows"] >= 1
    assert audit["complete_nonlinear_fibers"]["duplicate_rows"] >= 1
    assert audit["gate_antipodal_pairs"] >= 1
    assert audit["q4_nonlinear_fiber_tree_screen"]["status"] == "EXECUTED"
    assert audit["q4_nonlinear_fiber_tree_screen"]["native_bf16_semantics"] == "NOT_ESTABLISHED"


def test_q4_fiber_tree_screen_is_deterministic():
    mlp = FakeMLP().eval()
    first = analyze_whole_swiglu(mlp)["q4_nonlinear_fiber_tree_screen"]
    second = analyze_whole_swiglu(mlp)["q4_nonlinear_fiber_tree_screen"]
    assert first == second
    assert first["dense_fiber_coefficients"] == 16 * 24
    assert first["certified_tree_coefficient_lower_bound"] > 0


def test_corrupted_global_compiler_artifact_fails_closed(tmp_path):
    candidate = seeded_model()
    artifact = compile_checkpoint_global_swiglu(
        candidate,
        artifact_dir=tmp_path,
        layer_index=0,
        fiber_tile_rows=4,
        output_tile_rows=4,
    )
    first_key = sorted(artifact.runtime_layer.mlp.records)[0]
    record = artifact.runtime_layer.mlp.records[first_key]
    path = Path(artifact.artifact_dir) / record.relative_path
    payload = bytearray(path.read_bytes())
    payload[0] ^= 0x01
    path.write_bytes(payload)
    with pytest.raises(executor.ArtifactIntegrityError):
        artifact.runtime_layer.mlp(torch.randn(1, 1, 8))
