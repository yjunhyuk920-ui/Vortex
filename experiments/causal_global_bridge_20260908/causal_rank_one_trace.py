"""Legal causal compiler for many independently chosen right-factor queries.

This construction targets the exact scope gap left by the nonlinear-router
round: independently selected rank-one queries were not known to be reachable
inside one batch-1 causal trace.

For a bounded ordinary Llama family, token embeddings are signed basis vectors,
q/k are zero, v/o are identities, and the MLP is zero.  Hence a zero-embedding
query marker attends uniformly to the complete prefix.  The first block writes
one signed basis vector per coordinate.  A later bit flip writes *two* copies of
the new sign, changing the cumulative signed basis sum from old to new; an
unchanged bit writes two zero markers.  Thus each query marker is intended to
carry a separately chosen sign vector through the native causal path.

An arbitrary binary matrix W is placed in the first rows of ``lm_head``.  If
the final normalized query hidden is a common scalar times the intended sign
vector, the returned logits reveal signed row sums and therefore every GF(2)
rank-one parity for that right factor.  Native finite-word validation below is
mandatory: floating attention cancellation is not assumed algebraically.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from causal_kv_exposure import tensor_raw_bytes


@dataclass(frozen=True)
class RankOneTraceConfig:
    right_bits: int = 8
    output_rows: int = 4
    queries: int = 32
    seed: int = 260908


def make_binary_source(config: RankOneTraceConfig) -> torch.Tensor:
    if config.right_bits <= 0 or config.right_bits % 2:
        raise ValueError("right_bits must be positive and even for native Llama RoPE")
    if not 0 < config.output_rows <= 2 * config.right_bits + 1:
        raise ValueError("output_rows must fit the constructed vocabulary")
    rng = random.Random(config.seed)
    bits = [
        rng.getrandbits(1)
        for _ in range(config.output_rows * config.right_bits)
    ]
    return torch.tensor(bits, dtype=torch.uint8).reshape(
        config.output_rows, config.right_bits
    )


def make_query_vectors(config: RankOneTraceConfig) -> list[list[int]]:
    rng = random.Random(config.seed ^ 0x5A17)
    return [
        [rng.getrandbits(1) for _ in range(config.right_bits)]
        for _ in range(config.queries)
    ]


def build_rank_one_trace_model(
    config: RankOneTraceConfig,
    matrix: torch.Tensor,
) -> LlamaForCausalLM:
    n = config.right_bits
    if matrix.shape != (config.output_rows, n) or matrix.dtype != torch.uint8:
        raise ValueError("binary source matrix shape/dtype mismatch")
    if not bool(torch.all((matrix == 0) | (matrix == 1))):
        raise ValueError("source matrix must be binary")
    vocabulary = 2 * n + 1
    maximum_length = n + 1 + max(0, config.queries - 1) * (2 * n + 1)
    hf_config = LlamaConfig(
        vocab_size=vocabulary,
        hidden_size=n,
        intermediate_size=n,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=max(64, maximum_length + 8),
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
        use_cache=True,
    )
    model = LlamaForCausalLM(hf_config).eval().to(torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # token 2*j is +e_j, 2*j+1 is -e_j.  Token 2*n is the zero marker.
        for coordinate in range(n):
            model.model.embed_tokens.weight[2 * coordinate, coordinate] = 1
            model.model.embed_tokens.weight[2 * coordinate + 1, coordinate] = -1
        layer = model.model.layers[0]
        layer.input_layernorm.weight.fill_(1)
        layer.post_attention_layernorm.weight.fill_(1)
        layer.self_attn.v_proj.weight.copy_(torch.eye(n, dtype=torch.bfloat16))
        layer.self_attn.o_proj.weight.copy_(torch.eye(n, dtype=torch.bfloat16))
        # q/k and the complete MLP remain zero.
        model.model.norm.weight.fill_(1)
        model.lm_head.weight[: config.output_rows].copy_(matrix.to(torch.bfloat16))
    return model


def _sign_token(coordinate: int, bit: int) -> int:
    return 2 * coordinate if bit else 2 * coordinate + 1


def encode_query_trace(
    config: RankOneTraceConfig,
    query_vectors: list[list[int]],
) -> tuple[list[int], list[int]]:
    """Return legal token IDs and positions of zero-embedding query markers."""

    if len(query_vectors) != config.queries:
        raise ValueError("query vector count mismatch")
    n = config.right_bits
    for vector in query_vectors:
        if len(vector) != n or any(bit not in (0, 1) for bit in vector):
            raise ValueError("every query vector must be binary and right_bits long")
    zero = 2 * n
    tokens: list[int] = []
    query_positions: list[int] = []
    previous = query_vectors[0]
    for coordinate, bit in enumerate(previous):
        tokens.append(_sign_token(coordinate, bit))
    tokens.append(zero)
    query_positions.append(len(tokens) - 1)
    for vector in query_vectors[1:]:
        for coordinate, (old, new) in enumerate(zip(previous, vector)):
            if old == new:
                tokens.extend((zero, zero))
            else:
                token = _sign_token(coordinate, new)
                tokens.extend((token, token))
        tokens.append(zero)
        query_positions.append(len(tokens) - 1)
        previous = vector
    return tokens, query_positions


def run_trace_full(
    model: LlamaForCausalLM,
    token_ids: list[int],
) -> object:
    with torch.inference_mode():
        return model(
            input_ids=torch.tensor([token_ids], dtype=torch.long),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )


def run_trace_incremental(
    model: LlamaForCausalLM,
    token_ids: list[int],
) -> list[torch.Tensor]:
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
    return logits


def query_hidden_common_magnitude(
    hidden: torch.Tensor,
    query_vector: list[int],
) -> tuple[bool, int | None]:
    """Check exact signs and one common nonzero BF16 magnitude at a query."""

    if hidden.dtype != torch.bfloat16 or hidden.ndim != 1:
        raise ValueError("expected one BF16 hidden row")
    raw = hidden.detach().cpu().view(torch.uint16)
    magnitudes: set[int] = set()
    for coordinate, bit in enumerate(query_vector):
        value = float(hidden[coordinate].item())
        if (value > 0) != bool(bit):
            return False, None
        if value == 0:
            return False, None
        magnitudes.add(int(raw[coordinate].item()) & 0x7FFF)
    if len(magnitudes) != 1:
        return False, None
    return True, next(iter(magnitudes))


def parity_from_native_logits(
    matrix: torch.Tensor,
    query_vector: list[int],
) -> list[int]:
    """Reference GF(2) row parities for one right factor."""

    u = torch.tensor(query_vector, dtype=torch.int64)
    counts = matrix.to(torch.int64) @ u
    return [int(value.item()) & 1 for value in counts]


def _bf16_raw(value: torch.Tensor) -> int:
    return int(value.detach().cpu().reshape(1).view(torch.uint16).item())


def signed_sum_codebook(
    magnitude: torch.Tensor,
    right_bits: int,
) -> tuple[dict[tuple[int, int], int], int]:
    """Map ``(BF16 output word, row-sum parity)`` back to signed sum.

    For a binary row with ``h`` selected coefficients and a sign query ``s``,
    the exact integer signed sum has parity ``h mod 2``.  Opposite-parity
    collisions are therefore harmless; same-parity collisions would destroy
    the GF(2) row-parity decoder and are counted explicitly.
    """

    positive = abs(float(magnitude.item()))
    mapping: dict[tuple[int, int], int] = {}
    collisions = 0
    for signed_sum in range(-right_bits, right_bits + 1):
        value = torch.tensor(
            positive * signed_sum, dtype=torch.float32
        ).to(torch.bfloat16)
        raw = _bf16_raw(value)
        if (raw & 0x7FFF) == 0:
            raw = 0
        key = (raw, signed_sum & 1)
        prior = mapping.get(key)
        if prior is not None and prior != signed_sum:
            collisions += 1
        else:
            mapping[key] = signed_sum
    return mapping, collisions


def decode_row_parity(
    logit: torch.Tensor,
    row_sum: int,
    codebook: dict[tuple[int, int], int],
) -> tuple[int | None, int | None]:
    raw = _bf16_raw(logit)
    if (raw & 0x7FFF) == 0:
        raw = 0
    signed_sum = codebook.get((raw, row_sum & 1))
    if signed_sum is None:
        return None, None
    numerator = signed_sum + row_sum
    if numerator & 1:
        return signed_sum, None
    return signed_sum, (numerator // 2) & 1


def exhaustive_binary_row_decode_check(
    hidden: torch.Tensor,
    query_vector: list[int],
) -> dict[str, int]:
    """Check every binary lm-head row for small hidden widths."""

    n = hidden.numel()
    if n > 12:
        return {"rows_checked": 0, "decode_mismatches": 0, "codebook_collisions": 0}
    weights = torch.tensor(
        [
            [(mask >> coordinate) & 1 for coordinate in range(n)]
            for mask in range(1 << n)
        ],
        dtype=torch.bfloat16,
    )
    logits = F.linear(hidden.reshape(1, -1), weights)[0]
    codebook, collisions = signed_sum_codebook(abs(hidden[0]), n)
    signs = [1 if bit else -1 for bit in query_vector]
    mismatches = 0
    for mask in range(1 << n):
        selected = [(mask >> coordinate) & 1 for coordinate in range(n)]
        row_sum = sum(selected)
        true_signed = sum(bit * sign for bit, sign in zip(selected, signs))
        decoded_signed, decoded_parity = decode_row_parity(
            logits[mask], row_sum, codebook
        )
        true_parity = sum(
            bit * query_bit for bit, query_bit in zip(selected, query_vector)
        ) & 1
        if decoded_signed != true_signed or decoded_parity != true_parity:
            mismatches += 1
    return {
        "rows_checked": 1 << n,
        "decode_mismatches": mismatches,
        "codebook_collisions": collisions,
    }


def trace_audit(config: RankOneTraceConfig) -> dict[str, object]:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    matrix = make_binary_source(config)
    queries = make_query_vectors(config)
    model = build_rank_one_trace_model(config, matrix)
    tokens, query_positions = encode_query_trace(config, queries)
    rng_before = tensor_raw_bytes(torch.random.get_rng_state())
    full = run_trace_full(model, tokens)
    rng_after_full = tensor_raw_bytes(torch.random.get_rng_state())
    incremental_logits = run_trace_incremental(model, tokens)
    rng_after_incremental = tensor_raw_bytes(torch.random.get_rng_state())

    hidden_rows = full.hidden_states[-1][0]
    hidden_failures = 0
    magnitude_patterns: list[int] = []
    incremental_logit_mismatches = 0
    row_parity_decode_mismatches = 0
    codebook_collisions = 0
    rank_one_scalar_decode_mismatches = 0
    exhaustive_rows_checked = 0
    exhaustive_row_decode_mismatches = 0
    rng = random.Random(config.seed ^ 0x71E57)
    for query_index, position in enumerate(query_positions):
        good, magnitude = query_hidden_common_magnitude(
            hidden_rows[position], queries[query_index]
        )
        if not good or magnitude is None:
            hidden_failures += 1
            magnitude_patterns.append(-1)
        else:
            magnitude_patterns.append(magnitude)
        if tensor_raw_bytes(full.logits[:, position : position + 1]) != tensor_raw_bytes(
            incremental_logits[position]
        ):
            incremental_logit_mismatches += 1
        if good:
            codebook, collisions = signed_sum_codebook(
                abs(hidden_rows[position, 0]), config.right_bits
            )
            codebook_collisions += collisions
            decoded_rows: list[int] = []
            true_rows = parity_from_native_logits(matrix, queries[query_index])
            for row_index in range(config.output_rows):
                row_sum = int(matrix[row_index].sum().item())
                _, decoded = decode_row_parity(
                    full.logits[0, position, row_index], row_sum, codebook
                )
                if decoded is None:
                    row_parity_decode_mismatches += 1
                    decoded_rows.append(-1)
                else:
                    decoded_rows.append(decoded)
                    if decoded != true_rows[row_index]:
                        row_parity_decode_mismatches += 1
            left_mask = [rng.getrandbits(1) for _ in range(config.output_rows)]
            if -1 in decoded_rows:
                rank_one_scalar_decode_mismatches += 1
            else:
                decoded_scalar = sum(
                    left * parity for left, parity in zip(left_mask, decoded_rows)
                ) & 1
                true_scalar = sum(
                    left * parity for left, parity in zip(left_mask, true_rows)
                ) & 1
                if decoded_scalar != true_scalar:
                    rank_one_scalar_decode_mismatches += 1
            exhaustive = exhaustive_binary_row_decode_check(
                hidden_rows[position], queries[query_index]
            )
            exhaustive_rows_checked += exhaustive["rows_checked"]
            exhaustive_row_decode_mismatches += exhaustive["decode_mismatches"]
            codebook_collisions += exhaustive["codebook_collisions"]

    return {
        "classification": "E1_CAUSAL_MULTI_RIGHT_FACTOR_TRACE",
        "right_bits": config.right_bits,
        "output_rows": config.output_rows,
        "queries": config.queries,
        "sequence_length": len(tokens),
        "query_positions": query_positions,
        "query_hidden_sign_or_common_magnitude_failures": hidden_failures,
        "query_hidden_magnitude_patterns": magnitude_patterns,
        "full_vs_incremental_query_logit_mismatches": incremental_logit_mismatches,
        "row_parity_decode_mismatches": row_parity_decode_mismatches,
        "random_left_rank_one_scalar_decode_mismatches": rank_one_scalar_decode_mismatches,
        "same_parity_bf16_codebook_collisions": codebook_collisions,
        "exhaustive_binary_rows_checked": exhaustive_rows_checked,
        "exhaustive_binary_row_decode_mismatches": exhaustive_row_decode_mismatches,
        "rng_unchanged_after_full": rng_before == rng_after_full,
        "rng_unchanged_after_incremental": rng_before == rng_after_incremental,
        "claim_boundary": {
            "one_batch1_legal_trace_contains_all_declared_right_factor_queries": hidden_failures
            == 0,
            "tested_left_rank_one_masks_decodable_from_full_output_rows": (
                row_parity_decode_mismatches == 0
                and rank_one_scalar_decode_mismatches == 0
                and codebook_collisions == 0
            ),
            "arbitrary_left_rank_one_masks_decodable_for_small_exhaustive_width": (
                exhaustive_rows_checked > 0 and exhaustive_row_decode_mismatches == 0
            ),
            "area5400_32_query_native_trace_executed": False,
            "global_8gib_advice_direct_sum_proved": False,
            "native_cuda_semantics_proved": False,
            "mission_producer_constructed": False,
        },
    }
