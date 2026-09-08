"""Finite legal-causal exposure compiler for native Hugging Face Llama KV state.

This is an E0/E1 bridge, not a VORTEX acceleration algorithm.  It constructs a
standard ``LlamaForCausalLM`` whose required value-cache state exposes selected
columns of arbitrary binary dense matrices placed in native ``v_proj`` tensors.

The construction deliberately keeps native RMSNorm, native BF16 linear
evaluation and transformers ``DynamicCache``.  The only simplifying checkpoint
choices are ordinary tensor values: basis embeddings, zero output/MLP paths,
unit normalization weights and binary value-projection matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable

import torch
from transformers import LlamaConfig, LlamaForCausalLM


@dataclass(frozen=True)
class ExposureConfig:
    hidden_size: int = 32
    layers: int = 3
    seed: int = 260908
    # ``None`` preserves the original square/single-head construction.  A
    # smaller value size exercises ordinary grouped-query attention (GQA):
    # native Llama then has a rectangular ``v_proj[value_size, hidden_size]``.
    value_size: int | None = None
    head_dim: int | None = None


def resolved_value_size(config: ExposureConfig) -> int:
    return config.hidden_size if config.value_size is None else config.value_size


def resolved_head_dim(config: ExposureConfig) -> int:
    return config.hidden_size if config.head_dim is None else config.head_dim


def validate_exposure_config(config: ExposureConfig) -> tuple[int, int, int, int]:
    """Return ``(value_size, head_dim, q_heads, kv_heads)`` after HF checks."""

    d = config.hidden_size
    value_size = resolved_value_size(config)
    head_dim = resolved_head_dim(config)
    if d <= 0 or value_size <= 0 or config.layers <= 0 or head_dim <= 0:
        raise ValueError("hidden/value/head sizes and layer count must be positive")
    if d % head_dim or value_size % head_dim:
        raise ValueError("head_dim must divide both hidden_size and value_size")
    if head_dim % 2:
        raise ValueError("head_dim must be even for the native Llama RoPE layout")
    q_heads = d // head_dim
    kv_heads = value_size // head_dim
    # HF's eager/SDPA GQA path requires an integral Q-head/KV-head replication
    # factor.  Enforcing it here prevents a syntactically constructible config
    # from being mislabeled as a legal executable Llama forward.
    if q_heads % kv_heads:
        raise ValueError("num_key_value_heads must divide num_attention_heads")
    return value_size, head_dim, q_heads, kv_heads


def make_binary_matrices(config: ExposureConfig) -> list[torch.Tensor]:
    """Create deterministic arbitrary-looking binary source matrices."""

    value_size, _, _, _ = validate_exposure_config(config)
    rng = random.Random(config.seed)
    matrices: list[torch.Tensor] = []
    for _ in range(config.layers):
        bits = [
            rng.getrandbits(1)
            for _ in range(config.hidden_size * value_size)
        ]
        matrix = torch.tensor(bits, dtype=torch.uint8).reshape(
            value_size, config.hidden_size
        )
        matrices.append(matrix)
    return matrices


def build_exposure_model(
    config: ExposureConfig, matrices: list[torch.Tensor]
) -> LlamaForCausalLM:
    """Compile binary matrices into a normal HF Llama causal checkpoint."""

    d = config.hidden_size
    value_size, head_dim, q_heads, kv_heads = validate_exposure_config(config)
    if len(matrices) != config.layers:
        raise ValueError("one source matrix is required per layer")
    for matrix in matrices:
        if matrix.shape != (value_size, d):
            raise ValueError("source matrix shape mismatch")
        if matrix.dtype != torch.uint8:
            raise ValueError("source matrix must use uint8 binary entries")
        if not bool(torch.all((matrix == 0) | (matrix == 1))):
            raise ValueError("source matrix must be binary")

    hf_config = LlamaConfig(
        vocab_size=d,
        hidden_size=d,
        intermediate_size=d,
        num_hidden_layers=config.layers,
        num_attention_heads=q_heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max(64, 2 * d),
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
        use_cache=True,
    )
    model = LlamaForCausalLM(hf_config).eval().to(torch.bfloat16)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        # Every legal token ID is a basis direction.  No extra/private token
        # alphabet is used.
        for token in range(d):
            model.model.embed_tokens.weight[token, token] = 1

        model.model.norm.weight.fill_(1)
        for layer_index, layer in enumerate(model.model.layers):
            layer.input_layernorm.weight.fill_(1)
            layer.post_attention_layernorm.weight.fill_(1)
            layer.self_attn.v_proj.weight.copy_(
                matrices[layer_index].to(torch.bfloat16)
            )
            # q/k/o and the complete MLP remain zero from the blanket fill.
            # Hence attention/MLP residual updates are exactly zero, while the
            # native value projection is still executed and cached.

    return model


def tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        return cpu.view(torch.uint16).numpy().tobytes()
    return cpu.numpy().tobytes()


def checkpoint_sha256(model: LlamaForCausalLM) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor_raw_bytes(tensor))
    return digest.hexdigest()


def cache_layer_values(cache: object, layer_index: int) -> torch.Tensor:
    """Return native value cache across supported transformers cache APIs."""

    if hasattr(cache, "layers"):
        layer = cache.layers[layer_index]
        return layer.values
    return cache[layer_index][1]  # type: ignore[index]


def cache_layer_keys(cache: object, layer_index: int) -> torch.Tensor:
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_index]
        return layer.keys
    return cache[layer_index][0]  # type: ignore[index]


def cache_value_bit_pattern(value: torch.Tensor) -> int:
    if value.numel() != 1 or value.dtype != torch.bfloat16:
        raise ValueError("expected one BF16 scalar")
    return int(value.detach().cpu().view(torch.uint16).item())


def flatten_value_cache_by_position(values: torch.Tensor) -> torch.Tensor:
    """Convert native ``[B,KVH,T,HD]`` values to ``[T,KVH*HD]`` rows."""

    if values.ndim != 4 or values.shape[0] != 1:
        raise AssertionError(f"unexpected native value-cache rank/shape {values.shape}")
    return values[0].permute(1, 0, 2).contiguous().reshape(values.shape[2], -1)


def native_basis_scale_patterns(model: LlamaForCausalLM) -> list[int]:
    """Read the exact BF16 scalar produced by each native input RMSNorm.

    This is constructor-time work on one basis coordinate, not a replacement
    formula for RMSNorm.  The stored bit pattern is subsequently enough to emit
    a binary ``v_proj`` column because each row has exactly one potentially
    nonzero multiplicand on a basis input.
    """

    d = model.config.hidden_size
    basis = torch.zeros((1, 1, d), dtype=torch.bfloat16)
    basis[0, 0, 0] = 1
    patterns: list[int] = []
    with torch.inference_mode():
        for layer in model.model.layers:
            normalized = layer.input_layernorm(basis)
            if int(torch.count_nonzero(normalized[0, 0, 1:]).item()) != 0:
                raise AssertionError("basis RMSNorm unexpectedly mixed coordinates")
            patterns.append(cache_value_bit_pattern(normalized[0, 0, 0].reshape(1)))
    return patterns


def verify_cache_exposure(
    cache: object,
    matrices: list[torch.Tensor],
    token_ids: list[int],
    expected_nonzero_patterns: list[int] | None = None,
) -> dict[str, object]:
    """Verify exact BF16 cache words for every exposed source bit."""

    if not token_ids:
        raise ValueError("token list must be nonempty")
    total = 0
    mismatches = 0
    nonzero_patterns: set[int] = set()
    reconstructed: list[list[list[int]]] = []

    for layer_index, matrix in enumerate(matrices):
        values = cache_layer_values(cache, layer_index)
        flattened = flatten_value_cache_by_position(values)
        if flattened.shape != (len(token_ids), matrix.shape[0]):
            raise AssertionError(f"unexpected value cache shape {values.shape}")
        expected_pattern = None
        if expected_nonzero_patterns is not None:
            expected_pattern = expected_nonzero_patterns[layer_index]
        layer_reconstruction: list[list[int]] = []
        for position, token in enumerate(token_ids):
            vector = flattened[position]
            observed = (vector != 0).to(torch.uint8).cpu()
            expected = matrix[:, token].cpu()
            mismatches += int(torch.count_nonzero(observed != expected).item())
            total += matrix.shape[0]
            layer_reconstruction.append([int(bit) for bit in observed.tolist()])
            for output_index, scalar in enumerate(vector):
                raw = cache_value_bit_pattern(scalar.reshape(1))
                source_bit = int(expected[output_index].item())
                expected_raw = expected_pattern if source_bit else 0
                if expected_raw is not None and raw != expected_raw:
                    mismatches += 1
                if raw:
                    nonzero_patterns.add(raw)
        reconstructed.append(layer_reconstruction)

    return {
        "coordinates_checked": total,
        "mismatches": mismatches,
        "nonzero_bf16_bit_patterns": sorted(nonzero_patterns),
        "reconstructed_columns_by_layer_position": reconstructed,
    }


def caches_bit_equal(left: object, right: object, layers: int) -> bool:
    for layer_index in range(layers):
        for accessor in (cache_layer_keys, cache_layer_values):
            a = accessor(left, layer_index)
            b = accessor(right, layer_index)
            if a.shape != b.shape or a.dtype != b.dtype:
                return False
            if tensor_raw_bytes(a) != tensor_raw_bytes(b):
                return False
    return True


def run_full_prefill(
    model: LlamaForCausalLM, token_ids: list[int]
) -> object:
    inputs = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        return model(input_ids=inputs, use_cache=True, return_dict=True)


def run_incremental(
    model: LlamaForCausalLM, token_ids: list[int]
) -> tuple[object, list[torch.Tensor]]:
    if not token_ids:
        raise ValueError("token list must be nonempty")
    cache = None
    logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for token in token_ids:
            output = model(
                input_ids=torch.tensor([[token]], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = output.past_key_values
            logits.append(output.logits.detach().clone())
    return cache, logits


def derive_exposure_audit(config: ExposureConfig) -> dict[str, object]:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)

    matrices = make_binary_matrices(config)
    model = build_exposure_model(config, matrices)
    value_size, head_dim, q_heads, kv_heads = validate_exposure_config(config)
    basis_scale_patterns = native_basis_scale_patterns(model)
    parameter_hash_before = checkpoint_sha256(model)

    # The full vocabulary is the complete basis-query population.  A legal
    # prefill containing all tokens therefore exposes every source column.
    token_ids = list(range(config.hidden_size))
    rng_before = tensor_raw_bytes(torch.random.get_rng_state())
    full = run_full_prefill(model, token_ids)
    rng_after_full = tensor_raw_bytes(torch.random.get_rng_state())
    incremental_cache, incremental_logits = run_incremental(model, token_ids)
    rng_after_incremental = tensor_raw_bytes(torch.random.get_rng_state())

    exposure = verify_cache_exposure(
        full.past_key_values,
        matrices,
        token_ids,
        expected_nonzero_patterns=basis_scale_patterns,
    )
    if exposure["mismatches"] != 0:
        raise AssertionError("native cache no longer exposes the binary matrices")
    if not caches_bit_equal(
        full.past_key_values, incremental_cache, config.layers
    ):
        raise AssertionError("full prefill and incremental native caches differ")

    # lm_head is fixed zero, so this also checks the externally exposed logits
    # without making them the information carrier for the theorem.
    if int(torch.count_nonzero(full.logits).item()) != 0:
        raise AssertionError("fixed zero lm_head unexpectedly produced nonzero logits")
    for logits in incremental_logits:
        if int(torch.count_nonzero(logits).item()) != 0:
            raise AssertionError("incremental fixed-zero logits changed")

    parameter_hash_after = checkpoint_sha256(model)
    if parameter_hash_before != parameter_hash_after:
        raise AssertionError("checkpoint tensors mutated during forward")

    source_bits = config.layers * value_size * config.hidden_size
    value_cache_coordinates = source_bits
    return {
        "classification": "E1_HF_CAUSAL_KV_EXPOSURE_COMPILER",
        "environment": {
            "torch": torch.__version__,
            "transformers_model": type(model).__name__,
            "dtype": "bfloat16",
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "constructor": {
            "hidden_size": config.hidden_size,
            "value_size": value_size,
            "layers": config.layers,
            "vocab_size": config.hidden_size,
            "source_binary_vproj_bits": source_bits,
            "variable_source_tensors": [
                f"model.layers.{layer}.self_attn.v_proj.weight"
                for layer in range(config.layers)
            ],
            "basis_embedding_tokens": config.hidden_size,
            "head_dim": head_dim,
            "num_attention_heads": q_heads,
            "num_key_value_heads": kv_heads,
            "native_model_class": "transformers.LlamaForCausalLM",
            "trust_remote_code": False,
        },
        "legal_input": {
            "token_ids": token_ids,
            "sequence_length": len(token_ids),
            "all_ids_inside_vocab": True,
            "use_cache": True,
            "batch_size": 1,
        },
        "exact_equation": {
            "statement": (
                "Vcache[layer,position,i] != 0 iff "
                "W_layer[i,token_id[position]] == 1"
            ),
            "coordinates_checked": exposure["coordinates_checked"],
            "mismatches": exposure["mismatches"],
            "nonzero_bf16_bit_patterns": exposure["nonzero_bf16_bit_patterns"],
            "native_input_rmsnorm_basis_patterns": basis_scale_patterns,
            "full_vocab_prefill_reconstructs_every_binary_source_bit": True,
        },
        "causal_state": {
            "full_prefill_incremental_cache_bit_equal": True,
            "value_cache_coordinates": value_cache_coordinates,
            "key_cache_is_native_and_zero_by_checkpoint_construction": all(
                int(
                    torch.count_nonzero(
                        cache_layer_keys(full.past_key_values, layer)
                    ).item()
                )
                == 0
                for layer in range(config.layers)
            ),
            "full_logits_nonzero_coordinates": int(
                torch.count_nonzero(full.logits).item()
            ),
        },
        "rng": {
            "unchanged_after_full_prefill": rng_before == rng_after_full,
            "unchanged_after_incremental": rng_before == rng_after_incremental,
        },
        "checkpoint": {
            "sha256_before": parameter_hash_before,
            "sha256_after": parameter_hash_after,
            "unchanged": parameter_hash_before == parameter_hash_after,
        },
        "claim_boundary": {
            "actual_hf_native_forward_executed": True,
            "actual_dynamic_cache_executed": True,
            "every_binary_vproj_source_matrix_in_declared_family_covered_by_proof": True,
            "bounded_random_source_instance_executed": True,
            "arbitrary_checkpoint_accelerator_constructed": False,
            "dense_later_layer_hidden_state_producer_constructed": False,
            "global_cross_matrix_encoder_constructed": False,
            "hard_independent_32_query_union_transferred_to_one_causal_trace": False,
            "target_405b_executed": False,
            "cuda_executed": False,
            "target_hardware_executed": False,
        },
        "obligation_update": {
            "O1": "OPEN",
            "O2": "PARTIAL_CAUSAL_REACHABILITY_ONLY",
            "O3": "PARTIAL_DECLARED_BINARY_FAMILY_ONLY",
            "O4": "OPEN",
            "O5": "OPEN",
            "O6": "PARTIAL_BOUNDED_REPRODUCIBLE",
        },
        "decision": (
            "ESTABLISH_LEGAL_HF_SUCCESSOR_KV_COLUMN_EXPOSURE_KEEP_GLOBAL_"
            "CAUSAL_PRODUCER_AND_HARD_TUPLE_TRANSFER_OPEN"
        ),
    }


def write_audit(output_dir: Path, config: ExposureConfig) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = derive_exposure_audit(config)
    summary = output_dir / "summary.json"
    summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum = hashlib.sha256(summary.read_bytes()).hexdigest()
    (output_dir / "checksums.sha256").write_text(
        f"{checksum}  summary.json\n", encoding="utf-8"
    )
    return payload
