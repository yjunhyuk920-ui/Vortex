from __future__ import annotations

import math
import zlib
from typing import Any, Iterable

import torch
from torch import nn

from .fixed_public_dynamic_common import dtype_name, module_parameter_bytes, sha256_bytes, tensor_bytes


def byte_entropy(raw: bytes) -> float:
    if not raw: return 0.0
    counts = [0] * 256
    for value in raw: counts[value] += 1
    total = len(raw)
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def _duplicates(chunks: Iterable[bytes]) -> tuple[int, int]:
    counts: dict[str, int] = {}; total = 0
    for chunk in chunks:
        total += 1; digest = sha256_bytes(chunk); counts[digest] = counts.get(digest, 0) + 1
    return total - len(counts), total


def audit_tensor_structure(tensor: torch.Tensor, *, name: str, block_bytes: int = 256, matrix_block: int = 16) -> dict[str, Any]:
    raw = tensor_bytes(tensor); packed = zlib.compress(raw, 9)
    result: dict[str, Any] = {
        "name": name, "shape": list(map(int, tensor.shape)), "dtype": dtype_name(tensor.dtype), "raw_bytes": len(raw),
        "byte_entropy_bits_per_byte": byte_entropy(raw), "zlib9_bytes": len(packed), "zlib9_ratio": len(packed) / len(raw) if raw else 1.0,
        "sha256": sha256_bytes(raw),
    }
    dup, total = _duplicates(raw[i:i + block_bytes] for i in range(0, len(raw), block_bytes))
    result["exact_block_duplicate"] = {"block_bytes": block_bytes, "duplicate_blocks": dup, "total_blocks": total}
    flat = tensor.detach().reshape(-1)
    try: unique = int(torch.unique(flat).numel())
    except Exception: unique = -1
    result["exact_coefficient_dictionary"] = {"unique_values": unique, "total_values": int(flat.numel()), "repeat_values": int(flat.numel() - unique) if unique >= 0 else None}
    if tensor.ndim == 2:
        cpu = tensor.detach().contiguous().cpu()
        rows = [tensor_bytes(cpu[i]) for i in range(cpu.shape[0])]
        cols = [tensor_bytes(cpu[:, j].contiguous()) for j in range(cpu.shape[1])]
        dup_rows, total_rows = _duplicates(rows); dup_cols, total_cols = _duplicates(cols)
        row_hashes = {sha256_bytes(v) for v in rows}
        negated = sum(1 for i in range(cpu.shape[0]) if sha256_bytes(tensor_bytes((-cpu[i]).contiguous())) in row_hashes)
        result["exact_row_column_duplicate"] = {"duplicate_rows": dup_rows, "total_rows": total_rows, "duplicate_columns": dup_cols, "total_columns": total_cols, "exact_negated_row_matches": negated}
        b = int(matrix_block)
        blocks = [tensor_bytes(cpu[r:r+b, c:c+b].contiguous()) for r in range(0, cpu.shape[0], b) for c in range(0, cpu.shape[1], b)]
        dup_blocks, total_blocks = _duplicates(blocks)
        result["exact_block_factorization_dictionary"] = {"block_shape": [b, b], "duplicate_blocks": dup_blocks, "total_blocks": total_blocks}
        result["straight_line_program_bound"] = {"baseline_multiply_add_pairs": int(cpu.numel()), "reusable_exact_rows": dup_rows, "reusable_exact_columns": dup_cols}
        result["exact_addition_chain_candidates"] = {"negated_row_reuse": negated, "duplicate_row_reuse": dup_rows, "status": "EXACT_CANDIDATES_ONLY_NO_APPROXIMATE_SCALING"}
    byte_values = torch.frombuffer(bytearray(raw), dtype=torch.uint8) if raw else torch.empty(0, dtype=torch.uint8)
    result["bit_plane_circuit"] = {"bit_ones_per_byte_plane": [int(((byte_values >> bit) & 1).sum()) if raw else 0 for bit in range(8)], "literal_input_bits": len(raw) * 8, "status": "EXACT_LITERAL_COST_RECORDED_NO_UNPROVEN_MINIMIZATION"}
    result["grammar_dictionary"] = {"zlib9_bytes": len(packed), "dictionary_gain_bytes": len(raw) - len(packed)}
    result["fused_lossless_decode_compute"] = {"candidate": "projection-sequential exact decode then F.linear", "status": "IMPLEMENTED_FOR_SELECTED_MLP_PRIMITIVE"}
    result["decision_certificate"] = {"status": "NOT_USED_TO_SKIP_DENSE_ARITHMETIC"}
    return result


def audit_checkpoint_layer(layer: nn.Module, *, layer_index: int) -> dict[str, Any]:
    tensors = [audit_tensor_structure(p, name=f"model.layers.{layer_index}.{name}") for name, p in layer.named_parameters() if p.ndim >= 1]
    return {"layer_index": int(layer_index), "tensor_count": len(tensors), "tensors": tensors, "reference_parameter_bytes": module_parameter_bytes(layer), "audited_from_actual_parameters": True}
