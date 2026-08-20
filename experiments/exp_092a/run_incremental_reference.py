from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_092a import run_experiment as base  # noqa: E402

_LAYER_INDICES: tuple[int, ...] = ()
_ROLES: tuple[str, ...] = ()
_PENDING: dict[str, dict[str, Any]] = {}
_ORIGINAL_CAPTURE_BLOCK = base.capture_block


def _capture_incremental_reference(
    model: Any,
    prefill: Any,
    *,
    block_length: int,
) -> tuple[tuple[int, ...], dict[str, torch.Tensor], dict[str, Any]]:
    capture_lists: dict[str, list[torch.Tensor]] = {
        f"layer_{layer_index}:{role}": []
        for layer_index in _LAYER_INDICES
        for role in _ROLES
    }
    handles: list[Any] = []

    def make_hook(key: str):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            value = args[0]
            if (
                value.ndim != 3
                or int(value.shape[0]) != 1
                or int(value.shape[1]) != 1
            ):
                raise RuntimeError(
                    f"unexpected incremental input for {key}: {tuple(value.shape)}"
                )
            capture_lists[key].append(
                value[0, 0].detach().contiguous().cpu()
            )

        return hook

    for layer_index in _LAYER_INDICES:
        modules = base.role_modules(model.model.layers[layer_index])
        for role in _ROLES:
            key = f"layer_{layer_index}:{role}"
            handles.append(
                modules[role].register_forward_pre_hook(make_hook(key))
            )

    started_ns = time.perf_counter_ns()
    cache: Any = base.fresh_cache(prefill.prefix_cache)
    current = int(prefill.boundary_token)
    tokens: list[int] = []
    step_logits_sha256: list[str] = []
    try:
        for _ in range(int(block_length)):
            with torch.inference_mode():
                output = model(
                    input_ids=torch.tensor([[current]], dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
            logits = output.logits[:, -1, :].detach().to(torch.float32).cpu()
            token = int(base.stable_argmax_rows(logits)[0].item())
            tokens.append(token)
            step_logits_sha256.append(base.tensor_sha256(logits))
            current = token
            cache = output.past_key_values
    finally:
        for handle in handles:
            handle.remove()

    captures: dict[str, torch.Tensor] = {}
    for key, values in capture_lists.items():
        if len(values) != int(block_length):
            raise RuntimeError(
                f"incremental capture count {key}: {len(values)} != {block_length}"
            )
        captures[key] = torch.stack(values, dim=0).contiguous()
    completed_ns = time.perf_counter_ns()
    capture_hashes = {
        key: base.tensor_sha256(value)
        for key, value in sorted(captures.items())
    }
    meta = {
        "target_started_ns": started_ns,
        "target_completed_ns": completed_ns,
        "target_calls": int(block_length),
        "target_sha256": base.canonical_sha256(tokens),
        "step_logits_sha256": step_logits_sha256,
        "final_cache_sha256": base.cache_sha256(cache),
        "final_cache_bytes": base.cache_nbytes(cache),
        "incremental_capture_sha256": base.canonical_sha256(capture_hashes),
        "incremental_capture_hashes": capture_hashes,
        "incremental_capture_count_per_operator": int(block_length),
    }
    return tuple(tokens), captures, meta


def _patched_generate_ar_target(
    model: Any, prefill: Any, *, block_length: int
) -> tuple[tuple[int, ...], dict[str, Any]]:
    tokens, captures, meta = _capture_incremental_reference(
        model, prefill, block_length=block_length
    )
    _PENDING[prefill.case_id] = {
        "tokens": tokens,
        "captures": captures,
        "meta": meta,
    }
    return tokens, meta


def _patched_capture_block(
    model: Any,
    prefill: Any,
    block_tokens: Sequence[int],
    *,
    layer_indices: Sequence[int],
    roles: Sequence[str],
):
    pending = _PENDING.get(prefill.case_id)
    if pending is None:
        return _ORIGINAL_CAPTURE_BLOCK(
            model,
            prefill,
            block_tokens,
            layer_indices=layer_indices,
            roles=roles,
        )
    tokens = tuple(map(int, block_tokens))
    if tokens != pending["tokens"]:
        raise RuntimeError("incremental target tokens changed before span analysis")
    if tuple(map(int, layer_indices)) != _LAYER_INDICES or tuple(map(str, roles)) != _ROLES:
        raise RuntimeError("incremental capture population changed")
    _PENDING.pop(prefill.case_id)
    meta = pending["meta"]
    pseudo_capture_meta = {
        "started_ns": meta["target_started_ns"],
        "completed_ns": meta["target_completed_ns"],
        "wall_ns": meta["target_completed_ns"] - meta["target_started_ns"],
        "input_sha256": base.canonical_sha256(tokens),
        "proposal_sha256": base.canonical_sha256(tokens),
        "logits_sha256": base.canonical_sha256(meta["step_logits_sha256"]),
        "cache_sha256": meta["final_cache_sha256"],
        "cache_bytes": meta["final_cache_bytes"],
        "incremental_capture_sha256": meta["incremental_capture_sha256"],
        "reference_abi": "official_incremental_decode",
    }
    return pending["captures"], tokens, pseudo_capture_meta


def _rewrite_incremental_result(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    result["mechanism_fingerprint"] = result["mechanism_fingerprint"].replace(
        "true-teacher-forced-oracle", "true-incremental-reference"
    )
    for case in result["case_rows"]:
        passed = bool(case.pop("true_teacher_forcing_exact"))
        case["incremental_reference_capture_complete"] = passed
        case["incremental_reference_capture_sha256"] = case["true_capture"].pop(
            "incremental_capture_sha256"
        )
        case["incremental_reference_capture"] = case.pop("true_capture")
        case.pop("true_proposal_sha256", None)
    result["timing"]["official_block_calls"] = len(result["case_rows"])
    result["provenance"]["measured"] = [
        "DEV-W guessed complete-block inputs and true incremental-reference operator-input blocks, exact AR tokens/caches, and modular ranks on frozen coordinate restrictions"
    ]
    result["provenance"]["granted"] = [
        "true incremental reference block, exact coefficients, basis discovery, routing, nonlinear recomputation, coefficient application, and native-rounding repair"
    ]
    core_keys = (
        "schema",
        "mechanism_fingerprint",
        "config_sha256",
        "checkpoint",
        "block_length",
        "seed_mode",
        "layer_indices",
        "operator_roles",
        "rank_sketch_columns",
        "rank_primes",
        "controls",
        "case_rows",
        "span_rows",
        "build_aggregate",
        "holdout_aggregate",
        "target_projection",
        "gates",
        "authoritative_decision",
    )
    core = {key: result[key] for key in core_keys}
    result["deterministic_core_sha256"] = base.canonical_sha256(core)
    base.write_json(output_dir / "artifacts/deterministic_core.json", core)
    base.write_json(output_dir / "raw/case_rows.json", result["case_rows"])
    base.write_json(output_dir / "result.json", result)
    base.write_checksums(
        output_dir,
        [
            "artifacts/checkpoint_identity.json",
            "artifacts/deterministic_core.json",
            "raw/span_rows.json",
            "raw/case_rows.json",
            "result.json",
        ],
    )
    return result


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    global _LAYER_INDICES, _ROLES
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _LAYER_INDICES = tuple(map(int, config["layer_indices"]))
    _ROLES = tuple(map(str, config["operator_roles"]))
    base.generate_ar_target = _patched_generate_ar_target
    base.capture_block = _patched_capture_block
    result = base.execute(config_path, output_dir)
    if _PENDING:
        raise RuntimeError(f"unconsumed incremental captures: {sorted(_PENDING)}")
    return _rewrite_incremental_result(result, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    execute(args.config, args.output_dir)


if __name__ == "__main__":
    main()
