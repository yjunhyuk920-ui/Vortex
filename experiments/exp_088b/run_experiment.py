from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.cross_layer_program_sharing import (  # noqa: E402
    LLAMA405_IDEAL_BF16_BYTES,
    TARGET_HOT_BYTES_PER_TOKEN,
    build_checkpoint_static_page_plan,
    first_tensor_mismatch,
    gray_kept_masks,
    optimistic_405b_projection,
    select_minimal_shared_program,
    tensor_bytes,
    tensor_sha256,
)
from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    PINNED_SAFETENSORS,
    PINNED_TORCH,
    PINNED_TRANSFORMERS,
)


@dataclass
class CapturedTransition:
    state_id: str
    split: str
    family: str
    prompt_sha256: str
    token_count: int
    prefix_length: int
    decode_token_id: int
    next_token_id: int
    hidden_input: torch.Tensor
    attention_mask: torch.Tensor | None
    position_ids: torch.Tensor | None
    cache_position: torch.Tensor | None
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None
    prefix_keys: dict[int, torch.Tensor]
    prefix_values: dict[int, torch.Tensor]
    reference_tensors: tuple[torch.Tensor, ...]

    def trace(self) -> dict[str, Any]:
        names = (
            "pair_output_hidden_state",
            "first_layer_complete_k_cache",
            "first_layer_complete_v_cache",
            "second_layer_complete_k_cache",
            "second_layer_complete_v_cache",
        )
        return {
            "state_id": self.state_id,
            "split": self.split,
            "family": self.family,
            "prompt_sha256": self.prompt_sha256,
            "token_count": self.token_count,
            "prefix_length": self.prefix_length,
            "decode_token_id": self.decode_token_id,
            "next_token_id": self.next_token_id,
            "hidden_input_sha256": tensor_sha256(self.hidden_input),
            "hidden_input_shape": list(self.hidden_input.shape),
            "transition_tensors": [
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": tensor_sha256(tensor),
                    "raw_bytes": len(tensor_bytes(tensor)),
                }
                for name, tensor in zip(names, self.reference_tensors)
            ],
        }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system() == "Darwin" else value * 1024
    except Exception:
        return None


def verify_runtime() -> dict[str, Any]:
    import safetensors

    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    expected = {
        "torch": PINNED_TORCH,
        "transformers": PINNED_TRANSFORMERS,
        "safetensors": PINNED_SAFETENSORS,
    }
    mismatch = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if expected[key] != actual[key]
    }
    if mismatch:
        raise RuntimeError(f"runtime pin mismatch: {mismatch}")
    return actual


def clone_optional(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_optional(item) for item in value)
    if isinstance(value, list):
        return [clone_optional(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_optional(item) for key, item in value.items()}
    return value


def clone_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    cloned = DynamicCache()
    cloned.key_cache = list(cache.key_cache)
    cloned.value_cache = list(cache.value_cache)
    cloned._seen_tokens = int(cache._seen_tokens)
    return cloned


def target_only_cache(state: CapturedTransition, layer_indices: tuple[int, int]) -> DynamicCache:
    cache = DynamicCache()
    maximum = max(layer_indices)
    cache.key_cache = [[] for _ in range(maximum + 1)]
    cache.value_cache = [[] for _ in range(maximum + 1)]
    for layer_index in layer_indices:
        cache.key_cache[layer_index] = state.prefix_keys[layer_index]
        cache.value_cache[layer_index] = state.prefix_values[layer_index]
    cache._seen_tokens = int(state.prefix_length)
    return cache


def load_model(config: dict[str, Any]) -> tuple[Any, Any, int]:
    if config["model_id"] != DEV_MODEL_ID or config["revision"] != DEV_REVISION:
        raise RuntimeError(
            f"frozen DEV-W mismatch: {config['model_id']}@{config['revision']} != "
            f"{DEV_MODEL_ID}@{DEV_REVISION}"
        )
    start = time.perf_counter_ns()
    model = LlamaForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        low_cpu_mem_usage=False,
    )
    model.eval()
    resolved = str(getattr(model.config, "_commit_hash", "") or "")
    if resolved and resolved != config["revision"]:
        raise RuntimeError(f"resolved checkpoint mismatch: {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["revision"], use_fast=True
    )
    return model, tokenizer, time.perf_counter_ns() - start


def capture_transition(
    model: Any,
    tokenizer: Any,
    state_spec: dict[str, Any],
    *,
    layer_indices: tuple[int, int],
) -> CapturedTransition:
    first_index, second_index = layer_indices
    input_ids = tokenizer(state_spec["prompt"], return_tensors="pt")["input_ids"]
    if int(input_ids.shape[1]) < 2:
        raise RuntimeError(f"state {state_spec['id']} has fewer than two tokens")
    prefix_ids = input_ids[:, :-1]
    decode_id = input_ids[:, -1:]

    prefix_cache = DynamicCache()
    with torch.inference_mode():
        model(
            input_ids=prefix_ids,
            past_key_values=prefix_cache,
            use_cache=True,
            return_dict=True,
        )

    decode_cache = clone_dynamic_cache(prefix_cache)
    captured: dict[str, Any] = {}

    def first_pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if "hidden_input" in captured:
            raise RuntimeError("first selected layer executed more than once")
        captured["hidden_input"] = args[0].detach().clone()
        for key in (
            "attention_mask",
            "position_ids",
            "cache_position",
            "position_embeddings",
        ):
            captured[key] = clone_optional(kwargs.get(key))

    def second_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        if "pair_output" in captured:
            raise RuntimeError("second selected layer executed more than once")
        captured["pair_output"] = output[0].detach().clone()

    handles = [
        model.model.layers[first_index].register_forward_pre_hook(
            first_pre_hook, with_kwargs=True
        ),
        model.model.layers[second_index].register_forward_hook(second_hook),
    ]
    try:
        with torch.inference_mode():
            output = model(
                input_ids=decode_id,
                past_key_values=decode_cache,
                use_cache=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    required = {
        "hidden_input",
        "attention_mask",
        "position_ids",
        "cache_position",
        "position_embeddings",
        "pair_output",
    }
    missing = sorted(required.difference(captured))
    if missing:
        raise RuntimeError(f"capture missing fields: {missing}")

    prefix_keys = {
        index: prefix_cache.key_cache[index].detach().clone() for index in layer_indices
    }
    prefix_values = {
        index: prefix_cache.value_cache[index].detach().clone() for index in layer_indices
    }
    reference = (
        captured["pair_output"],
        decode_cache.key_cache[first_index].detach().clone(),
        decode_cache.value_cache[first_index].detach().clone(),
        decode_cache.key_cache[second_index].detach().clone(),
        decode_cache.value_cache[second_index].detach().clone(),
    )
    next_token = int(torch.argmax(output.logits[0, -1]).item())
    return CapturedTransition(
        state_id=str(state_spec["id"]),
        split=str(state_spec["split"]),
        family=str(state_spec["family"]),
        prompt_sha256=sha256_bytes(state_spec["prompt"].encode("utf-8")),
        token_count=int(input_ids.shape[1]),
        prefix_length=int(prefix_ids.shape[1]),
        decode_token_id=int(decode_id.item()),
        next_token_id=next_token,
        hidden_input=captured["hidden_input"],
        attention_mask=captured["attention_mask"],
        position_ids=captured["position_ids"],
        cache_position=captured["cache_position"],
        position_embeddings=captured["position_embeddings"],
        prefix_keys=prefix_keys,
        prefix_values=prefix_values,
        reference_tensors=reference,
    )


def replay_transition(
    model: Any,
    state: CapturedTransition,
    *,
    layer_indices: tuple[int, int],
) -> tuple[torch.Tensor, ...]:
    cache = target_only_cache(state, layer_indices)
    hidden = state.hidden_input
    common = {
        "attention_mask": state.attention_mask,
        "position_ids": state.position_ids,
        "past_key_value": cache,
        "output_attentions": False,
        "use_cache": True,
        "cache_position": state.cache_position,
        "position_embeddings": state.position_embeddings,
    }
    with torch.inference_mode():
        for layer_index in layer_indices:
            hidden = model.model.layers[layer_index](hidden, **common)[0]
    first_index, second_index = layer_indices
    return (
        hidden.detach().clone(),
        cache.key_cache[first_index].detach().clone(),
        cache.value_cache[first_index].detach().clone(),
        cache.key_cache[second_index].detach().clone(),
        cache.value_cache[second_index].detach().clone(),
    )


def selected_window_sha256(model: Any, layer_indices: tuple[int, int]) -> str:
    digest = hashlib.sha256()
    for layer_index in layer_indices:
        layer = model.model.layers[layer_index]
        for module_name, module in sorted(layer.named_modules(), key=lambda item: item[0]):
            if not isinstance(module, torch.nn.Linear):
                continue
            digest.update(f"layers.{layer_index}.{module_name}.weight".encode("utf-8"))
            digest.update(str(tuple(module.weight.shape)).encode("utf-8"))
            digest.update(str(module.weight.dtype).encode("utf-8"))
            digest.update(tensor_bytes(module.weight))
    return digest.hexdigest()


def perturbation_control(reference: tuple[torch.Tensor, ...]) -> bool:
    candidate = [tensor.detach().clone() for tensor in reference]
    flat = candidate[0].view(torch.uint8).reshape(-1)
    flat[0] ^= 0x01
    return first_tensor_mismatch(reference, tuple(candidate)) is not None


def evaluate_state(
    model: Any, state: CapturedTransition, layer_indices: tuple[int, int]
) -> tuple[bool, dict[str, Any] | None, int]:
    start = time.perf_counter_ns()
    candidate = replay_transition(model, state, layer_indices=layer_indices)
    elapsed = time.perf_counter_ns() - start
    mismatch = first_tensor_mismatch(state.reference_tensors, candidate)
    return mismatch is None, mismatch, elapsed


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def write_checksums(output_dir: Path, relative_paths: Iterable[str]) -> None:
    lines = []
    for relative in sorted(set(relative_paths)):
        path = output_dir / relative
        lines.append(f"{sha256_file(path)}  {relative}")
    (output_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def validate_config(config: dict[str, Any]) -> tuple[int, int]:
    if config["schema"] != "exp-088b-oracle-cross-layer-program-sharing-v1":
        raise RuntimeError("unexpected config schema")
    if not config.get("frozen_before_results"):
        raise RuntimeError("config is not frozen before results")
    layers = tuple(int(value) for value in config["layer_indices"])
    if len(layers) != 2 or layers[1] != layers[0] + 1:
        raise RuntimeError("Gate requires exactly two adjacent complete decoder layers")
    if int(config["page_group_count"]) != 8:
        raise RuntimeError("preregistered exhaustive language requires eight page families")
    states = config["states"]
    build = [row for row in states if row["split"] == "build"]
    holdout = [row for row in states if row["split"] == "holdout"]
    if len(build) < int(config["minimum_distinct_build_states"]):
        raise RuntimeError("insufficient build states")
    if len(holdout) < int(config["minimum_distinct_holdout_states"]):
        raise RuntimeError("insufficient holdout states")
    ids = [str(row["id"]) for row in states]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate state IDs")
    if int(config["ideal_405b_bf16_checkpoint_bytes"]) != LLAMA405_IDEAL_BF16_BYTES:
        raise RuntimeError("405B byte baseline mismatch")
    if int(config["target_405b_hot_bytes_per_token"]) != TARGET_HOT_BYTES_PER_TOKEN:
        raise RuntimeError("405B hot-byte target mismatch")
    return layers  # type: ignore[return-value]


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    layer_indices = validate_config(config)
    runtime = verify_runtime()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    load_start_rss = rss_bytes()
    model, tokenizer, model_load_ns = load_model(config)
    if max(layer_indices) >= len(model.model.layers):
        raise RuntimeError("selected layer window outside checkpoint")

    plan = build_checkpoint_static_page_plan(
        [(index, model.model.layers[index]) for index in layer_indices],
        group_count=int(config["page_group_count"]),
        tile_rows=int(config["tile_rows"]),
        tile_cols=int(config["tile_cols"]),
    )
    if any(
        tuple(sorted(layers)) != tuple(sorted(layer_indices))
        for layers in plan.group_layers
    ):
        raise RuntimeError("every page family must span both selected layers")

    checkpoint_sha_before = selected_window_sha256(model, layer_indices)
    page_plan = plan.manifest()
    page_plan["manifest_sha256"] = plan.manifest_sha256()
    write_json(output_dir / "artifacts/page_plan.json", page_plan)

    build_specs = [row for row in config["states"] if row["split"] == "build"]
    holdout_specs = [row for row in config["states"] if row["split"] == "holdout"]
    build_states = [
        capture_transition(model, tokenizer, row, layer_indices=layer_indices)
        for row in build_specs
    ]
    build_hashes = [tensor_sha256(state.hidden_input) for state in build_states]
    distinct_build = len(set(build_hashes)) == len(build_hashes)
    perturbation_detected = all(
        perturbation_control(state.reference_tensors) for state in build_states
    )

    masks = gray_kept_masks(plan.group_count)
    exact_masks_by_state: dict[str, list[int]] = {
        state.state_id: [] for state in build_states
    }
    mask_rows: list[dict[str, Any]] = []
    total_pair_replays = 0
    total_pair_layer_calls = 0
    build_search_start = time.perf_counter_ns()
    with plan.executor() as executor:
        for ordinal, mask in enumerate(masks):
            executor.apply(mask)
            program = plan.program(mask)
            exact_state_ids: list[str] = []
            first_mismatches: dict[str, Any] = {}
            replay_ns: list[int] = []
            for state in build_states:
                exact, mismatch, elapsed = evaluate_state(model, state, layer_indices)
                total_pair_replays += 1
                total_pair_layer_calls += len(layer_indices)
                replay_ns.append(elapsed)
                if exact:
                    exact_masks_by_state[state.state_id].append(mask)
                    exact_state_ids.append(state.state_id)
                elif len(first_mismatches) < 2:
                    first_mismatches[state.state_id] = mismatch
            mask_rows.append(
                {
                    "gray_ordinal": ordinal,
                    "kept_mask": mask,
                    "program_sha256": program.program_sha256,
                    "kept_groups": list(program.kept_groups),
                    "retained_linear_bytes": program.retained_linear_bytes,
                    "retained_fraction": program.retained_fraction,
                    "exact_build_state_ids": exact_state_ids,
                    "exact_build_state_count": len(exact_state_ids),
                    "all_build_states_exact": len(exact_state_ids)
                    == len(build_states),
                    "pair_replay_ns_min": min(replay_ns),
                    "pair_replay_ns_max": max(replay_ns),
                    "first_mismatches": first_mismatches,
                }
            )
            if ordinal % 16 == 0:
                print(
                    f"build oracle {ordinal + 1}/{len(masks)} mask={mask:#04x} "
                    f"exact={len(exact_state_ids)}/{len(build_states)}",
                    flush=True,
                )

        selection = select_minimal_shared_program(
            exact_masks_by_state,
            [state.state_id for state in build_states],
            group_count=plan.group_count,
            group_bytes=plan.group_bytes,
        )
        selected_program = plan.program(selection.selected_mask)

        # Capture holdout references only after the program is selected from
        # build states. No holdout target can influence the selected program.
        executor.apply(plan.full_mask)
        holdout_states = [
            capture_transition(model, tokenizer, row, layer_indices=layer_indices)
            for row in holdout_specs
        ]
        holdout_hashes = [tensor_sha256(state.hidden_input) for state in holdout_states]
        distinct_holdout = len(set(holdout_hashes)) == len(holdout_hashes)
        disjoint_splits = set(build_hashes).isdisjoint(holdout_hashes)
        perturbation_detected = perturbation_detected and all(
            perturbation_control(state.reference_tensors) for state in holdout_states
        )

        holdout_masks: list[int] = []
        for mask in (plan.full_mask, selection.selected_mask, 0):
            if mask not in holdout_masks:
                holdout_masks.append(mask)
        holdout_rows: list[dict[str, Any]] = []
        for mask in holdout_masks:
            executor.apply(mask)
            exact_state_ids: list[str] = []
            mismatches: dict[str, Any] = {}
            replay_ns: list[int] = []
            for state in holdout_states:
                exact, mismatch, elapsed = evaluate_state(model, state, layer_indices)
                total_pair_replays += 1
                total_pair_layer_calls += len(layer_indices)
                replay_ns.append(elapsed)
                if exact:
                    exact_state_ids.append(state.state_id)
                else:
                    mismatches[state.state_id] = mismatch
            holdout_rows.append(
                {
                    "kept_mask": mask,
                    "program_sha256": plan.program(mask).program_sha256,
                    "role": (
                        "full_reference_control"
                        if mask == plan.full_mask
                        else "selected_build_program"
                        if mask == selection.selected_mask
                        else "empty_negative_control"
                    ),
                    "exact_holdout_state_ids": exact_state_ids,
                    "exact_holdout_state_count": len(exact_state_ids),
                    "all_holdout_states_exact": len(exact_state_ids)
                    == len(holdout_states),
                    "pair_replay_ns_min": min(replay_ns),
                    "pair_replay_ns_max": max(replay_ns),
                    "first_mismatches": mismatches,
                }
            )
        executor.apply(plan.full_mask)

    build_search_ns = time.perf_counter_ns() - build_search_start
    checkpoint_sha_after = selected_window_sha256(model, layer_indices)

    full_build_exact = all(
        plan.full_mask in exact_masks_by_state[state.state_id]
        for state in build_states
    )
    selected_holdout_row = next(
        row for row in holdout_rows if row["kept_mask"] == selection.selected_mask
    )
    full_holdout_row = next(
        row for row in holdout_rows if row["kept_mask"] == plan.full_mask
    )
    empty_holdout_row = next(row for row in holdout_rows if row["kept_mask"] == 0)

    input_integrity_passed = (
        distinct_build
        and distinct_holdout
        and disjoint_splits
        and perturbation_detected
        and checkpoint_sha_before == checkpoint_sha_after
        and full_build_exact
        and bool(full_holdout_row["all_holdout_states_exact"])
        and len(mask_rows) == (1 << plan.group_count)
        and all(
            len(exact_masks_by_state[state.state_id]) >= 1
            for state in build_states
        )
    )
    useful_nontrivial_program = selection.selected_mask != plan.full_mask
    holdout_shared_exact = bool(selected_holdout_row["all_holdout_states_exact"])
    retained_threshold_passed = (
        selected_program.retained_fraction
        <= float(config["promotion_max_retained_fraction"])
    )
    oracle_program_sharing_gate_passed = (
        input_integrity_passed
        and useful_nontrivial_program
        and holdout_shared_exact
        and retained_threshold_passed
        and len(holdout_states)
        >= int(config["minimum_distinct_holdout_states"])
    )
    projection_405b = optimistic_405b_projection(
        selected_program.retained_fraction
    )

    if not input_integrity_passed:
        authoritative_decision = "INVALID_ORACLE_PROGRAM_SHARING_CONTROL_FAILURE"
    elif oracle_program_sharing_gate_passed:
        authoritative_decision = (
            "PROMOTE_CROSS_LAYER_PAGE_PROGRAM_TO_FULL_LAYER_128_STEP_SELECTOR_GATE"
        )
    else:
        authoritative_decision = (
            "REJECT_CROSS_LAYER_PAGE_MASK_PROGRAM_SHARING_AS_405B_CORE"
        )

    traces = [state.trace() for state in build_states + holdout_states]
    per_state_minimal: dict[str, Any] = {}
    for state in build_states:
        candidates = sorted(
            exact_masks_by_state[state.state_id],
            key=lambda mask: (
                plan.program(mask).retained_linear_bytes,
                int(mask).bit_count(),
                int(mask),
            ),
        )
        best = plan.program(candidates[0])
        per_state_minimal[state.state_id] = {
            "kept_mask": best.kept_mask,
            "program_sha256": best.program_sha256,
            "retained_fraction": best.retained_fraction,
            "exact_program_count": len(candidates),
        }

    deterministic_core = {
        "schema": config["schema"],
        "mechanism_fingerprint": "checkpoint-static/two-adjacent-complete-layers/linear-64x64-tiles/eight-cross-layer-page-families/exhaustive-bitmask/final-hidden-plus-local-KV-bit-exact/build-intersection/untouched-holdout",
        "config_sha256": sha256_file(config_path),
        "checkpoint": {
            "model_id": config["model_id"],
            "revision": config["revision"],
            "resolved_revision": str(
                getattr(model.config, "_commit_hash", "") or ""
            ),
            "selected_window_linear_sha256": checkpoint_sha_before,
            "layer_indices": list(layer_indices),
        },
        "page_plan": {
            "manifest_sha256": plan.manifest_sha256(),
            "group_count": plan.group_count,
            "tile_rows": plan.tile_rows,
            "tile_cols": plan.tile_cols,
            "tile_count": len(plan.descriptors),
            "parameter_names": list(plan.parameter_names),
            "group_bytes": list(plan.group_bytes),
            "group_layers": [list(value) for value in plan.group_layers],
            "total_linear_bytes": plan.total_linear_bytes,
            "total_linear_elements": plan.total_linear_elements,
        },
        "oracle_contract": {
            "search_space_size": 1 << plan.group_count,
            "search_is_exhaustive": True,
            "selection_uses_only_build_states": True,
            "holdout_targets_observed_after_selection": True,
            "future_generated_tokens_used_for_selection": False,
            "program_contains_target_output_bytes": False,
            "program_contains_state_hash_or_state_id": False,
            "program_runtime_selector": "none; one selected static mask is applied to every holdout state",
            "candidate_evaluator": "official two-layer graph with omitted checkpoint tiles zeroed; dense evaluator is charged as oracle search and is not a runtime speed claim",
            "nonseparable_exactness": "only final pair hidden state and both layers' complete post-step K/V caches are compared; no individual page or first-layer equality is required",
        },
        "state_traces": traces,
        "build_mask_rows": mask_rows,
        "build_exact_masks_by_state": exact_masks_by_state,
        "per_state_minimal_program": per_state_minimal,
        "shared_build_selection": {
            "build_state_ids": list(selection.build_state_ids),
            "shared_exact_program_count": len(selection.shared_exact_masks),
            "selected": {
                "kept_mask": selected_program.kept_mask,
                "kept_groups": list(selected_program.kept_groups),
                "omitted_groups": list(selected_program.omitted_groups),
                "program_sha256": selected_program.program_sha256,
                "retained_linear_bytes": selected_program.retained_linear_bytes,
                "total_linear_bytes": selected_program.total_linear_bytes,
                "retained_fraction": selected_program.retained_fraction,
            },
        },
        "holdout_rows": holdout_rows,
        "controls": {
            "distinct_build_states": distinct_build,
            "distinct_holdout_states": distinct_holdout,
            "build_holdout_state_hashes_disjoint": disjoint_splits,
            "one_bit_perturbation_detected": perturbation_detected,
            "full_mask_exact_on_all_build_states": full_build_exact,
            "full_mask_exact_on_all_holdout_states": full_holdout_row[
                "all_holdout_states_exact"
            ],
            "empty_mask_exact_holdout_state_count": empty_holdout_row[
                "exact_holdout_state_count"
            ],
            "checkpoint_restored_bit_exact": checkpoint_sha_before
            == checkpoint_sha_after,
        },
        "gates": {
            "integrity_passed": input_integrity_passed,
            "useful_nontrivial_program": useful_nontrivial_program,
            "selected_program_exact_on_all_unseen_holdout_states": holdout_shared_exact,
            "promotion_retained_fraction_limit": float(
                config["promotion_max_retained_fraction"]
            ),
            "retained_fraction_gate_passed": retained_threshold_passed,
            "oracle_program_sharing_gate_passed": oracle_program_sharing_gate_passed,
        },
        "optimistic_405b_projection": projection_405b,
        "authoritative_decision": authoritative_decision,
    }

    result = {
        **deterministic_core,
        "provenance": {
            "measured": [
                "DEV-W page plan, exact program masks, pair hidden/KV equality, state uniqueness, exhaustive build-mask outcomes, holdout outcomes, wall time, RSS"
            ],
            "derived": [
                "retained checkpoint fraction inside the selected two-layer linear window"
            ],
            "projected": [
                "optimistic 405B hot bytes and 100 GB/s I/O time assuming measured retained fraction generalizes to all checkpoint linear pages"
            ],
            "unverified": [
                "TARGET-W 405B behavior, actual sparse kernel equality/performance, target GPU VRAM, p50/p95 latency, 128-step full-model continuation"
            ],
        },
        "runtime": runtime,
        "timing": {
            "model_load_ns": model_load_ns,
            "oracle_and_holdout_ns": build_search_ns,
            "total_pair_replays": total_pair_replays,
            "total_pair_layer_calls": total_pair_layer_calls,
            "reference_full_model_calls": len(build_states) * 2
            + len(holdout_states) * 2,
            "peak_rss_bytes": rss_bytes(),
            "rss_before_model_load_bytes": load_start_rss,
        },
    }

    write_json(output_dir / "artifacts/deterministic_core.json", deterministic_core)
    write_json(output_dir / "raw/build_mask_matrix.json", mask_rows)
    write_json(output_dir / "raw/state_traces.json", traces)
    write_json(output_dir / "result.json", result)
    write_checksums(
        output_dir,
        [
            "artifacts/page_plan.json",
            "artifacts/deterministic_core.json",
            "raw/build_mask_matrix.json",
            "raw/state_traces.json",
            "result.json",
        ],
    )
    print(
        json.dumps(
            {
                "authoritative_decision": authoritative_decision,
                "selected_mask": selected_program.kept_mask,
                "retained_fraction": selected_program.retained_fraction,
                "holdout_shared_exact": holdout_shared_exact,
                "oracle_program_sharing_gate_passed": oracle_program_sharing_gate_passed,
                "optimistic_405b_hot_bytes": projection_405b[
                    "ideal_hot_bytes_per_token"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    execute(args.config, args.output_dir)


if __name__ == "__main__":
    main()
