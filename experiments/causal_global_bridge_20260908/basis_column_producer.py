"""Exact no-dense-forward producer for the preregistered basis-column family.

This is deliberately *not* a universal VORTEX executor.  It completes the
positive side of Principle A: once the ordinary Llama checkpoint is restricted
to basis embeddings, zero q/k/o+MLP paths and binary ``v_proj`` matrices, a
finite lossless compiler can emit the exact public logits and ``DynamicCache``
successor without evaluating any original dense matrix kernel at runtime.

The construction is useful because it separates two facts that must not be
conflated:

* arbitrary dense checkpoint bits can be causally exposed by native HF state;
* a specially restricted causal query family can still have a trivial exact
  producer (column addressing).

Therefore the earlier independently-selectable rank-one hard tuple does not
automatically transfer to one batch-1 causal trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from causal_kv_exposure import (
    ExposureConfig,
    build_exposure_model,
    cache_layer_keys,
    cache_layer_values,
    checkpoint_sha256,
    make_binary_matrices,
    native_basis_scale_patterns,
    run_incremental,
    tensor_raw_bytes,
    validate_exposure_config,
)


WORD_BITS = 64


@dataclass(frozen=True)
class CompiledBasisColumnStore:
    hidden_size: int
    value_size: int
    layers: int
    head_dim: int
    num_key_value_heads: int
    scale_patterns: tuple[int, ...]
    # Source-independent native RoPE image of an all-zero key for each legal
    # position. Signed-zero bits matter to the exact cache contract.
    key_zero_template: torch.Tensor
    # [layer][token/column][word]
    packed_columns: tuple[tuple[tuple[int, ...], ...], ...]

    @property
    def words_per_column(self) -> int:
        return ceil(self.value_size / WORD_BITS)

    @property
    def stored_payload_bits(self) -> int:
        return self.layers * self.hidden_size * self.words_per_column * WORD_BITS

    @property
    def stored_auxiliary_bits(self) -> int:
        return int(self.key_zero_template.numel()) * 16 + self.layers * 16

    @property
    def source_binary_bits(self) -> int:
        return self.layers * self.hidden_size * self.value_size

    @property
    def runtime_payload_bits_per_token(self) -> int:
        return self.layers * self.words_per_column * WORD_BITS

    @property
    def runtime_value_coordinates_per_token(self) -> int:
        return self.layers * self.value_size


def _pack_column(column: torch.Tensor) -> tuple[int, ...]:
    if column.ndim != 1 or column.dtype != torch.uint8:
        raise ValueError("column must be a flat uint8 tensor")
    if not bool(torch.all((column == 0) | (column == 1))):
        raise ValueError("column must be binary")
    words: list[int] = []
    for start in range(0, column.numel(), WORD_BITS):
        word = 0
        stop = min(start + WORD_BITS, column.numel())
        for index in range(start, stop):
            word |= int(column[index].item()) << (index - start)
        words.append(word)
    return tuple(words)


def compile_basis_column_store(
    config: ExposureConfig,
    matrices: list[torch.Tensor],
) -> tuple[CompiledBasisColumnStore, object]:
    """Compile the declared family and return the store plus native control model.

    The native model is returned only so the experiment can compare the compiled
    producer against the official HF forward.  The runtime producer below never
    invokes it.
    """

    value_size, head_dim, _, kv_heads = validate_exposure_config(config)
    model = build_exposure_model(config, matrices)
    scales = tuple(native_basis_scale_patterns(model))
    maximum_positions = int(model.config.max_position_embeddings)
    positions = torch.arange(maximum_positions, dtype=torch.long).unsqueeze(0)
    rope_probe = torch.zeros(
        (1, maximum_positions, config.hidden_size), dtype=torch.bfloat16
    )
    with torch.inference_mode():
        cos, sin = model.model.rotary_emb(rope_probe, positions)
        zero = torch.zeros(
            (1, 1, maximum_positions, head_dim), dtype=torch.bfloat16
        )
        _, rotated_zero_key = apply_rotary_pos_emb(zero, zero, cos, sin)
        key_zero_template = rotated_zero_key[0, 0].contiguous().clone()
    packed_layers: list[tuple[tuple[int, ...], ...]] = []
    for matrix in matrices:
        columns = tuple(
            _pack_column(matrix[:, token].contiguous())
            for token in range(config.hidden_size)
        )
        packed_layers.append(columns)
    store = CompiledBasisColumnStore(
        hidden_size=config.hidden_size,
        value_size=value_size,
        layers=config.layers,
        head_dim=head_dim,
        num_key_value_heads=kv_heads,
        scale_patterns=scales,
        key_zero_template=key_zero_template,
        packed_columns=tuple(packed_layers),
    )
    return store, model


def _bf16_scalar_from_pattern(pattern: int) -> torch.Tensor:
    if not 0 <= pattern <= 0xFFFF:
        raise ValueError("BF16 raw pattern must fit uint16")
    return torch.tensor([pattern], dtype=torch.uint16).view(torch.bfloat16)[0]


def decode_value_vector(
    store: CompiledBasisColumnStore,
    layer_index: int,
    token_id: int,
) -> torch.Tensor:
    """Decode one exact native value-projection output from packed source bits."""

    if not 0 <= token_id < store.hidden_size:
        raise ValueError("token id is outside the declared vocabulary")
    if not 0 <= layer_index < store.layers:
        raise ValueError("layer index is outside the compiled store")
    words = store.packed_columns[layer_index][token_id]
    scale = _bf16_scalar_from_pattern(store.scale_patterns[layer_index])
    vector = torch.zeros(store.value_size, dtype=torch.bfloat16)
    for output_index in range(store.value_size):
        word = words[output_index // WORD_BITS]
        if (word >> (output_index % WORD_BITS)) & 1:
            vector[output_index] = scale
    return vector


def compiled_incremental_step(
    store: CompiledBasisColumnStore,
    token_id: int,
    cache: DynamicCache | None,
) -> tuple[torch.Tensor, DynamicCache]:
    """Emit one exact public logit/cache transition without a dense forward."""

    if cache is None:
        cache = DynamicCache()
    position = cache.get_seq_length()
    if position >= store.key_zero_template.shape[0]:
        raise ValueError("position exceeds the compiled RoPE template")
    for layer_index in range(store.layers):
        flat = decode_value_vector(store, layer_index, token_id)
        values = flat.reshape(
            1,
            store.num_key_value_heads,
            1,
            store.head_dim,
        )
        key_row = store.key_zero_template[position]
        keys = (
            key_row.reshape(1, 1, 1, store.head_dim)
            .expand(1, store.num_key_value_heads, 1, store.head_dim)
            .contiguous()
        )
        cache.update(keys, values, layer_index)
    # The family fixes lm_head exactly to zero, so these are the exact native
    # public logits under the frozen BF16 return_dict/use_cache ABI.
    logits = torch.zeros((1, 1, store.hidden_size), dtype=torch.bfloat16)
    return logits, cache


def run_compiled_incremental(
    store: CompiledBasisColumnStore,
    token_ids: list[int],
) -> tuple[DynamicCache, list[torch.Tensor]]:
    if not token_ids:
        raise ValueError("token list must be nonempty")
    cache: DynamicCache | None = None
    logits: list[torch.Tensor] = []
    for token_id in token_ids:
        current_logits, cache = compiled_incremental_step(store, token_id, cache)
        logits.append(current_logits)
    if cache is None:  # pragma: no cover - guarded by the nonempty check.
        raise AssertionError("compiled cache unexpectedly absent")
    return cache, logits


def public_transition_bit_equal(
    native_cache: object,
    native_logits: list[torch.Tensor],
    compiled_cache: object,
    compiled_logits: list[torch.Tensor],
    layers: int,
) -> bool:
    if len(native_logits) != len(compiled_logits):
        return False
    for left, right in zip(native_logits, compiled_logits):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if tensor_raw_bytes(left) != tensor_raw_bytes(right):
            return False
    for layer_index in range(layers):
        for accessor in (cache_layer_keys, cache_layer_values):
            left = accessor(native_cache, layer_index)
            right = accessor(compiled_cache, layer_index)
            if left.shape != right.shape or left.dtype != right.dtype:
                return False
            if tensor_raw_bytes(left) != tensor_raw_bytes(right):
                return False
    return True


def compiled_producer_audit(
    config: ExposureConfig,
    token_ids: list[int],
) -> dict[str, object]:
    """Run native-vs-compiled controls and expose fully paid local costs."""

    if not token_ids:
        raise ValueError("token list must be nonempty")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    matrices = make_binary_matrices(config)
    store, model = compile_basis_column_store(config, matrices)
    checkpoint_before = checkpoint_sha256(model)
    native_cache, native_logits = run_incremental(model, token_ids)
    checkpoint_after_native = checkpoint_sha256(model)
    rng_before_compiled = tensor_raw_bytes(torch.random.get_rng_state())
    compiled_cache, compiled_logits = run_compiled_incremental(store, token_ids)
    rng_after_compiled = tensor_raw_bytes(torch.random.get_rng_state())
    exact = public_transition_bit_equal(
        native_cache,
        native_logits,
        compiled_cache,
        compiled_logits,
        config.layers,
    )
    native_variable_vproj_bits = (
        store.layers * store.value_size * store.hidden_size * 16
    )
    runtime_ratio = store.runtime_payload_bits_per_token / native_variable_vproj_bits
    return {
        "classification": "E1_RESTRICTED_CAUSAL_BASIS_COLUMN_PRODUCER",
        "hidden_size": store.hidden_size,
        "value_size": store.value_size,
        "layers": store.layers,
        "head_dim": store.head_dim,
        "token_ids": token_ids,
        "native_public_logits_kv_bit_exact": exact,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after_native": checkpoint_after_native,
        "checkpoint_unchanged": checkpoint_before == checkpoint_after_native,
        "compiled_runtime_rng_unchanged": rng_before_compiled == rng_after_compiled,
        "cost": {
            "source_binary_bits": store.source_binary_bits,
            "compiled_padded_source_bits": store.stored_payload_bits,
            "compiled_auxiliary_bits": store.stored_auxiliary_bits,
            "runtime_source_payload_bits_per_token": store.runtime_payload_bits_per_token,
            "runtime_value_coordinates_per_token": store.runtime_value_coordinates_per_token,
            "runtime_materialized_kv_bits_per_token": store.layers * store.value_size * 2 * 16,
            "native_variable_vproj_bits_if_full_matrix_read_per_token": native_variable_vproj_bits,
            "compiled_payload_over_native_variable_vproj_full_read": runtime_ratio,
            "variable_vproj_read_fraction_removed": 1.0 - runtime_ratio,
            "dense_vproj_kernel_calls_in_compiled_runtime": 0,
            "bit_tests_to_decode_values_per_token": store.layers * store.value_size,
        },
        "claim_boundary": {
            "finite_encoder_address_decoder_constructed": True,
            "complete_declared_logits_kv_rng_transition_constructed": True,
            "removes_at_least_90pct_variable_vproj_read": runtime_ratio <= 0.1,
            "arbitrary_public_checkpoint": False,
            "arbitrary_finite_word_weight_alphabet": False,
            "general_causal_hidden_state": False,
            "whole_model_target_latency_closed": False,
            "mission_solution": False,
        },
    }
