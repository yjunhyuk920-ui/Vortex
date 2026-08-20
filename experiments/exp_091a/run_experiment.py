from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.exact_jacobi_fixed_point import (  # noqa: E402
    build_seed,
    canonical_sha256,
    changed_positions,
    exhaustive_toy_prefix_control,
    future_mutation_control,
    jacobi_input,
    logical_weight_fraction,
    longest_common_prefix,
    required_tokens_per_sweep,
    run_triangular_toy,
    select_seed_mode,
    self_consistent_prefix,
    stable_argmax_rows,
    tensor_sha256,
)
from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    PINNED_SAFETENSORS,
    PINNED_TORCH,
    PINNED_TRANSFORMERS,
    config_dict,
    model_tensor_sha256,
)
from vortex_runtime.prefix_state_bisimulation import cache_nbytes, cache_sha256  # noqa: E402


@dataclass
class PrefillState:
    case_id: str
    family: str
    prompt_sha256: str
    prompt_ids: tuple[int, ...]
    boundary_token: int
    prefix_cache: tuple[Any, ...]
    prefill_logits_sha256: str
    prefix_cache_sha256: str
    prefix_cache_bytes: int
    prefill_target_calls: int


@dataclass
class Trajectory:
    mode: str
    seed_tokens: tuple[int, ...]
    seed_manifest: dict[str, Any]
    iterations: list[dict[str, Any]]
    completed_ns: int
    trajectory_sha256: str
    block_sweeps: int
    processed_token_positions: int


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def write_checksums(output_dir: Path, relative_paths: Iterable[str]) -> None:
    lines: list[str] = []
    for relative in sorted(set(relative_paths)):
        lines.append(f"{sha256_file(output_dir / relative)}  {relative}")
    (output_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def verify_runtime() -> dict[str, Any]:
    import safetensors

    actual = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
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


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "exp-091a-exact-jacobi-fixed-point-v1":
        raise RuntimeError("unexpected config schema")
    if not config.get("frozen_before_results"):
        raise RuntimeError("experiment was not frozen before results")
    if config.get("model_id") != DEV_MODEL_ID or config.get("revision") != DEV_REVISION:
        raise RuntimeError("frozen DEV-W identity mismatch")
    if int(config["block_length"]) != 128:
        raise RuntimeError("EXP-091A freezes K=128")
    if int(config["maximum_iterations"]) != 8:
        raise RuntimeError("EXP-091A freezes eight Jacobi sweeps")
    expected_modes = [
        "boundary_repeat",
        "prompt_suffix_cycle_16",
        "prompt_longest_ngram_8",
    ]
    if list(config["seed_modes"]) != expected_modes:
        raise RuntimeError("seed language/order changed")

    target = config["target_projection"]
    no_compression = required_tokens_per_sweep(
        float(target["whole_model_fraction_limit"])
    )
    favorable = required_tokens_per_sweep(
        float(target["whole_model_fraction_limit"]),
        compression_ratio=float(target["checkpoint_compression_ratio_favorable"]),
    )
    if no_compression != int(target["no_compression_required_tokens_per_sweep"]):
        raise RuntimeError("no-compression token threshold mismatch")
    if favorable != int(target["favorable_required_tokens_per_sweep"]):
        raise RuntimeError("favorable token threshold mismatch")

    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("case population is empty")
    ids = [str(case["id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case IDs")
    build = [case for case in cases if case["split"] == "build"]
    holdout = [case for case in cases if case["split"] == "holdout"]
    if len(build) < int(config["success"]["minimum_build_cases"]):
        raise RuntimeError("insufficient build cases")
    if len(holdout) < int(config["success"]["minimum_holdout_cases"]):
        raise RuntimeError("insufficient holdout cases")
    build_prompts = {sha256_bytes(case["prompt"].encode()) for case in build}
    holdout_prompts = {sha256_bytes(case["prompt"].encode()) for case in holdout}
    if build_prompts.intersection(holdout_prompts):
        raise RuntimeError("build and holdout prompts overlap")


def load_model(config: dict[str, Any]) -> tuple[Any, Any, int]:
    start = time.perf_counter_ns()
    model = LlamaForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        low_cpu_mem_usage=False,
    ).eval()
    resolved = str(getattr(model.config, "_commit_hash", "") or "")
    if resolved and resolved != config["revision"]:
        raise RuntimeError(f"resolved checkpoint mismatch: {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["revision"], use_fast=True
    )
    return model, tokenizer, time.perf_counter_ns() - start


def legacy_cache(past_key_values: Any) -> tuple[Any, ...]:
    if isinstance(past_key_values, DynamicCache):
        return tuple(past_key_values.to_legacy_cache())
    if hasattr(past_key_values, "to_legacy_cache"):
        return tuple(past_key_values.to_legacy_cache())
    if isinstance(past_key_values, (tuple, list)):
        return tuple(past_key_values)
    raise RuntimeError(f"unsupported cache type: {type(past_key_values).__name__}")


def fresh_dynamic_cache(prefix_cache: tuple[Any, ...]) -> DynamicCache:
    return DynamicCache.from_legacy_cache(prefix_cache)


def checkpoint_identity(model: Any, tokenizer: Any, config: dict[str, Any]) -> dict[str, Any]:
    tokenizer_payload = {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0)),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
    }
    return {
        "model_id": config["model_id"],
        "requested_revision": config["revision"],
        "resolved_revision": str(getattr(model.config, "_commit_hash", "") or ""),
        "model_class": type(model).__name__,
        "config_sha256": canonical_sha256(config_dict(model.config)),
        "checkpoint_tensor_sha256": model_tensor_sha256(model),
        "tokenizer": {**tokenizer_payload, "sha256": canonical_sha256(tokenizer_payload)},
    }


def prefill_case(model: Any, tokenizer: Any, case: Mapping[str, Any]) -> PrefillState:
    input_ids = tokenizer(case["prompt"], return_tensors="pt")["input_ids"]
    if int(input_ids.shape[1]) < 1:
        raise RuntimeError(f"empty prompt tokenization: {case['id']}")
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
    logits = output.logits[:, -1, :].detach().to(torch.float32).cpu()
    boundary = int(stable_argmax_rows(logits)[0].item())
    prefix = legacy_cache(output.past_key_values)
    return PrefillState(
        case_id=str(case["id"]),
        family=str(case["family"]),
        prompt_sha256=sha256_bytes(case["prompt"].encode("utf-8")),
        prompt_ids=tuple(int(value) for value in input_ids[0].tolist()),
        boundary_token=boundary,
        prefix_cache=prefix,
        prefill_logits_sha256=tensor_sha256(logits),
        prefix_cache_sha256=cache_sha256(prefix),
        prefix_cache_bytes=cache_nbytes(prefix),
        prefill_target_calls=1,
    )


def jacobi_block_sweep(
    model: Any,
    prefill: PrefillState,
    guess: Sequence[int],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    block_input = jacobi_input(prefill.boundary_token, guess)
    cache = fresh_dynamic_cache(prefill.prefix_cache)
    start = time.perf_counter_ns()
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([list(block_input)], dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
    wall_ns = time.perf_counter_ns() - start
    proposals_tensor = stable_argmax_rows(output.logits)[0]
    proposal = tuple(int(value) for value in proposals_tensor.tolist())
    if len(proposal) != len(guess):
        raise RuntimeError("Jacobi proposal length mismatch")
    return proposal, {
        "wall_ns": wall_ns,
        "input_sha256": canonical_sha256(block_input),
        "logits_sha256": tensor_sha256(output.logits.detach().to(torch.float32).cpu()),
        "proposal_sha256": canonical_sha256(proposal),
        "post_block_cache_sha256": cache_sha256(output.past_key_values),
        "post_block_cache_bytes": cache_nbytes(output.past_key_values),
    }


def run_trajectory(
    model: Any,
    prefill: PrefillState,
    *,
    mode: str,
    block_length: int,
    maximum_iterations: int,
) -> Trajectory:
    seed, manifest = build_seed(
        mode,
        prompt_ids=prefill.prompt_ids,
        boundary_token=prefill.boundary_token,
        block_length=block_length,
    )
    guess = seed
    iterations: list[dict[str, Any]] = []
    for sweep in range(1, int(maximum_iterations) + 1):
        proposal, trace = jacobi_block_sweep(model, prefill, guess)
        accepted = self_consistent_prefix(guess, proposal)
        changed = changed_positions(guess, proposal)
        iterations.append(
            {
                "sweep": sweep,
                "input_guess": list(guess),
                "proposal": list(proposal),
                "input_guess_sha256": canonical_sha256(guess),
                "proposal_sha256": canonical_sha256(proposal),
                "self_consistent_prefix": accepted,
                "changed_positions": changed,
                "fixed_point": changed == 0,
                **trace,
            }
        )
        guess = proposal
        if changed == 0:
            break
    completed_ns = time.perf_counter_ns()
    trajectory_core = {
        "mode": mode,
        "seed": list(seed),
        "seed_manifest": manifest.to_dict(),
        "iterations": iterations,
    }
    return Trajectory(
        mode=mode,
        seed_tokens=seed,
        seed_manifest=manifest.to_dict(),
        iterations=iterations,
        completed_ns=completed_ns,
        trajectory_sha256=canonical_sha256(trajectory_core),
        block_sweeps=len(iterations),
        processed_token_positions=len(iterations) * int(block_length),
    )


def generate_ar_target(
    model: Any,
    prefill: PrefillState,
    *,
    block_length: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    target_start_ns = time.perf_counter_ns()
    cache: Any = fresh_dynamic_cache(prefill.prefix_cache)
    current = int(prefill.boundary_token)
    tokens: list[int] = []
    step_hashes: list[str] = []
    for _ in range(int(block_length)):
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([[current]], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        logits = output.logits[:, -1, :].detach().to(torch.float32).cpu()
        token = int(stable_argmax_rows(logits)[0].item())
        tokens.append(token)
        step_hashes.append(tensor_sha256(logits))
        current = token
        cache = output.past_key_values
    return tuple(tokens), {
        "target_start_ns": target_start_ns,
        "target_calls": int(block_length),
        "step_logits_sha256": step_hashes,
        "target_sha256": canonical_sha256(tokens),
        "final_cache_sha256": cache_sha256(cache),
        "final_cache_bytes": cache_nbytes(cache),
    }


def finalize_trajectory(
    prefill: PrefillState,
    trajectory: Trajectory,
    target: Sequence[int],
    target_meta: Mapping[str, Any],
    *,
    fraction_limit: float,
    compression_ratio: float,
) -> dict[str, Any]:
    target_tokens = tuple(int(value) for value in target)
    rows: list[dict[str, Any]] = []
    prefix_sound = True
    fixed_point_sound = True
    first_proposal_token_exact = True
    for iteration in trajectory.iterations:
        guess = tuple(int(value) for value in iteration["input_guess"])
        proposal = tuple(int(value) for value in iteration["proposal"])
        accepted = int(iteration["self_consistent_prefix"])
        guess_target_prefix = longest_common_prefix(guess, target_tokens)
        proposal_target_prefix = longest_common_prefix(proposal, target_tokens)
        this_prefix_sound = guess[:accepted] == target_tokens[:accepted]
        this_fixed_sound = not bool(iteration["fixed_point"]) or guess == target_tokens
        this_first_exact = proposal[0] == target_tokens[0]
        prefix_sound = prefix_sound and this_prefix_sound
        fixed_point_sound = fixed_point_sound and this_fixed_sound
        first_proposal_token_exact = first_proposal_token_exact and this_first_exact
        rows.append(
            {
                **iteration,
                "guess_target_prefix": guess_target_prefix,
                "proposal_target_prefix": proposal_target_prefix,
                "self_consistent_prefix_sound": this_prefix_sound,
                "fixed_point_equals_ar_target": this_fixed_sound,
                "first_proposal_token_exact": this_first_exact,
                "logical_fraction_no_compression": logical_weight_fraction(
                    int(iteration["sweep"]), accepted
                ),
                "logical_fraction_favorable_compression": logical_weight_fraction(
                    int(iteration["sweep"]),
                    accepted,
                    compression_ratio=compression_ratio,
                ),
            }
        )

    first_accepted = int(rows[0]["self_consistent_prefix"])
    best_row = sorted(
        rows,
        key=lambda row: (
            -int(row["self_consistent_prefix"]),
            int(row["sweep"]),
        ),
    )[0]
    fixed_sweeps = [int(row["sweep"]) for row in rows if row["fixed_point"]]
    fixed_point_sweep = min(fixed_sweeps) if fixed_sweeps else None
    return {
        "case_id": prefill.case_id,
        "family": prefill.family,
        "mode": trajectory.mode,
        "prompt_sha256": prefill.prompt_sha256,
        "prompt_token_count": len(prefill.prompt_ids),
        "boundary_token": prefill.boundary_token,
        "seed_manifest": trajectory.seed_manifest,
        "seed_tokens": list(trajectory.seed_tokens),
        "seed_sha256": canonical_sha256(trajectory.seed_tokens),
        "trajectory_sha256": trajectory.trajectory_sha256,
        "trajectory_completed_before_target": trajectory.completed_ns
        <= int(target_meta["target_start_ns"]),
        "target_tokens": list(target_tokens),
        "target_sha256": target_meta["target_sha256"],
        "iterations": rows,
        "first_sweep_accepted": first_accepted,
        "best_accepted": int(best_row["self_consistent_prefix"]),
        "best_accepted_sweep": int(best_row["sweep"]),
        "fixed_point_sweep": fixed_point_sweep,
        "prefix_theorem_control_passed": prefix_sound,
        "fixed_point_control_passed": fixed_point_sound,
        "first_proposal_token_control_passed": first_proposal_token_exact,
        "first_sweep_fraction_no_compression": logical_weight_fraction(1, first_accepted),
        "first_sweep_fraction_favorable_compression": logical_weight_fraction(
            1, first_accepted, compression_ratio=compression_ratio
        ),
        "first_sweep_target_passed_no_compression": logical_weight_fraction(
            1, first_accepted
        )
        <= fraction_limit,
        "first_sweep_target_passed_favorable_compression": logical_weight_fraction(
            1, first_accepted, compression_ratio=compression_ratio
        )
        <= fraction_limit,
        "block_sweeps": trajectory.block_sweeps,
        "processed_token_positions": trajectory.processed_token_positions,
    }


def run_case_modes(
    model: Any,
    tokenizer: Any,
    case: Mapping[str, Any],
    *,
    modes: Sequence[str],
    block_length: int,
    maximum_iterations: int,
    fraction_limit: float,
    compression_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prefill = prefill_case(model, tokenizer, case)
    trajectories = [
        run_trajectory(
            model,
            prefill,
            mode=str(mode),
            block_length=block_length,
            maximum_iterations=maximum_iterations,
        )
        for mode in modes
    ]
    # Every trajectory is complete and hashed before the delayed AR control starts.
    target, target_meta = generate_ar_target(
        model, prefill, block_length=block_length
    )
    rows = [
        finalize_trajectory(
            prefill,
            trajectory,
            target,
            target_meta,
            fraction_limit=fraction_limit,
            compression_ratio=compression_ratio,
        )
        for trajectory in trajectories
    ]
    trace = {
        "case_id": prefill.case_id,
        "family": prefill.family,
        "prompt_sha256": prefill.prompt_sha256,
        "prompt_token_count": len(prefill.prompt_ids),
        "boundary_token": prefill.boundary_token,
        "prefill_logits_sha256": prefill.prefill_logits_sha256,
        "prefix_cache_sha256": prefill.prefix_cache_sha256,
        "prefix_cache_bytes": prefill.prefix_cache_bytes,
        "target_sha256": target_meta["target_sha256"],
        "target_final_cache_sha256": target_meta["final_cache_sha256"],
        "target_final_cache_bytes": target_meta["final_cache_bytes"],
        "prefill_target_calls": prefill.prefill_target_calls,
        "ar_target_calls": target_meta["target_calls"],
        "trajectory_modes": list(map(str, modes)),
        "all_trajectories_precede_target": all(
            trajectory.completed_ns <= int(target_meta["target_start_ns"])
            for trajectory in trajectories
        ),
    }
    return rows, trace


def aggregate_rows(rows: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row["mode"]) == str(mode)]
    if not selected:
        raise RuntimeError(f"no rows for mode {mode}")
    first = [int(row["first_sweep_accepted"]) for row in selected]
    best = [int(row["best_accepted"]) for row in selected]
    fixed = [row["fixed_point_sweep"] for row in selected]
    return {
        "mode": str(mode),
        "case_count": len(selected),
        "first_sweep_accepted": first,
        "first_sweep_min": min(first),
        "first_sweep_p50": float(np.percentile(first, 50, method="linear")),
        "first_sweep_max": max(first),
        "first_sweep_total": sum(first),
        "best_accepted": best,
        "best_min": min(best),
        "best_p50": float(np.percentile(best, 50, method="linear")),
        "best_max": max(best),
        "fixed_point_sweeps": fixed,
        "all_trajectories_precede_target": all(
            bool(row["trajectory_completed_before_target"]) for row in selected
        ),
        "prefix_theorem_controls": all(
            bool(row["prefix_theorem_control_passed"]) for row in selected
        ),
        "fixed_point_controls": all(
            bool(row["fixed_point_control_passed"]) for row in selected
        ),
        "first_proposal_controls": all(
            bool(row["first_proposal_token_control_passed"]) for row in selected
        ),
    }


def model_free_controls(config: Mapping[str, Any]) -> dict[str, Any]:
    exhaustive = exhaustive_toy_prefix_control(8)
    future = future_mutation_control()
    toy = run_triangular_toy(
        [101, 102, 103, 104, 105], [1, 2, 3, 4, 5], maximum_iterations=6
    )
    target = config["target_projection"]
    no_compression = required_tokens_per_sweep(
        float(target["whole_model_fraction_limit"])
    )
    favorable = required_tokens_per_sweep(
        float(target["whole_model_fraction_limit"]),
        compression_ratio=float(target["checkpoint_compression_ratio_favorable"]),
    )
    controls = {
        "exhaustive_prefix_theorem": exhaustive,
        "future_mutation": future,
        "triangular_positive_control": toy,
        "triangular_fixed_point_sweep": next(
            (int(row["sweep"]) for row in toy if row["fixed_point"]), None
        ),
        "required_no_compression": no_compression,
        "required_favorable": favorable,
        "two_sweep_no_compression_fraction": logical_weight_fraction(2, 128),
        "two_sweep_favorable_fraction": logical_weight_fraction(
            2,
            128,
            compression_ratio=float(target["checkpoint_compression_ratio_favorable"]),
        ),
    }
    controls["passed"] = all(
        [
            bool(exhaustive["passed"]),
            bool(future["passed"]),
            controls["triangular_fixed_point_sweep"] == 6,
            no_compression == int(target["no_compression_required_tokens_per_sweep"]),
            favorable == int(target["favorable_required_tokens_per_sweep"]),
            controls["two_sweep_no_compression_fraction"]
            > float(target["whole_model_fraction_limit"]),
            controls["two_sweep_favorable_fraction"]
            > float(target["whole_model_fraction_limit"]),
        ]
    )
    return controls


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    runtime = verify_runtime()
    torch.manual_seed(0)
    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.set_num_interop_threads(1)

    model, tokenizer, model_load_ns = load_model(config)
    identity = checkpoint_identity(model, tokenizer, config)
    controls = model_free_controls(config)
    target_projection = config["target_projection"]
    fraction_limit = float(target_projection["whole_model_fraction_limit"])
    compression_ratio = float(target_projection["checkpoint_compression_ratio_favorable"])
    block_length = int(config["block_length"])
    maximum_iterations = int(config["maximum_iterations"])
    modes = list(map(str, config["seed_modes"]))

    build_cases = [case for case in config["cases"] if case["split"] == "build"]
    holdout_cases = [case for case in config["cases"] if case["split"] == "holdout"]
    build_rows: list[dict[str, Any]] = []
    case_traces: list[dict[str, Any]] = []
    run_start = time.perf_counter_ns()

    for case in build_cases:
        rows, trace = run_case_modes(
            model,
            tokenizer,
            case,
            modes=modes,
            block_length=block_length,
            maximum_iterations=maximum_iterations,
            fraction_limit=fraction_limit,
            compression_ratio=compression_ratio,
        )
        build_rows.extend(rows)
        case_traces.append({**trace, "split": "build"})
        print(
            f"build {case['id']}: "
            + ", ".join(
                f"{row['mode']}={row['first_sweep_accepted']}/{row['best_accepted']}"
                for row in rows
            ),
            flush=True,
        )

    selection = select_seed_mode(build_rows, modes)
    selection_frozen_ns = time.perf_counter_ns()
    selected_mode = selection.mode
    holdout_rows: list[dict[str, Any]] = []
    holdout_started_after_selection = True

    for case in holdout_cases:
        before = time.perf_counter_ns()
        rows, trace = run_case_modes(
            model,
            tokenizer,
            case,
            modes=[selected_mode],
            block_length=block_length,
            maximum_iterations=maximum_iterations,
            fraction_limit=fraction_limit,
            compression_ratio=compression_ratio,
        )
        holdout_started_after_selection = (
            holdout_started_after_selection and before >= selection_frozen_ns
        )
        holdout_rows.extend(rows)
        case_traces.append({**trace, "split": "holdout"})
        print(
            f"holdout {case['id']}: {selected_mode}="
            f"{rows[0]['first_sweep_accepted']}/{rows[0]['best_accepted']}",
            flush=True,
        )

    experiment_ns = time.perf_counter_ns() - run_start
    build_aggregates = {mode: aggregate_rows(build_rows, mode) for mode in modes}
    holdout_aggregate = aggregate_rows(holdout_rows, selected_mode)
    holdout_min_first = int(holdout_aggregate["first_sweep_min"])
    no_compression_required = int(
        target_projection["no_compression_required_tokens_per_sweep"]
    )
    favorable_required = int(target_projection["favorable_required_tokens_per_sweep"])

    integrity_failures: list[str] = []
    if not bool(controls["passed"]):
        integrity_failures.append("model_free_controls")
    if not all(
        bool(row["trajectory_completed_before_target"])
        for row in build_rows + holdout_rows
    ):
        integrity_failures.append("trajectory_target_order")
    if not all(
        bool(row["prefix_theorem_control_passed"])
        for row in build_rows + holdout_rows
    ):
        integrity_failures.append("self_consistent_prefix_theorem")
    if not all(
        bool(row["fixed_point_control_passed"])
        for row in build_rows + holdout_rows
    ):
        integrity_failures.append("fixed_point_ar_equality")
    if not all(
        bool(row["first_proposal_token_control_passed"])
        for row in build_rows + holdout_rows
    ):
        integrity_failures.append("first_proposal_token")
    if not holdout_started_after_selection:
        integrity_failures.append("holdout_before_selection")
    if identity["resolved_revision"] not in {"", config["revision"]}:
        integrity_failures.append("checkpoint_revision")
    if any(
        len(row["seed_tokens"]) != block_length for row in build_rows + holdout_rows
    ):
        integrity_failures.append("seed_length")

    integrity_passed = len(integrity_failures) <= int(
        config["success"]["maximum_integrity_failures"]
    )
    no_compression_signal = (
        integrity_passed and holdout_min_first >= no_compression_required
    )
    favorable_signal = (
        integrity_passed
        and not no_compression_signal
        and holdout_min_first >= favorable_required
    )

    if not integrity_passed:
        decision = "INVALID_EXACT_JACOBI_CONTROL_FAILURE"
    elif no_compression_signal:
        decision = "PROMOTE_ONE_SWEEP_JACOBI_PREFIX_TO_EXACT_BLOCK_EXECUTOR_GATE"
    elif favorable_signal:
        decision = "PROMOTE_COMPRESSED_ONE_SWEEP_JACOBI_PREFIX_TO_BLOCK_EXECUTOR_GATE"
    else:
        decision = "REJECT_NATIVE_JACOBI_FIXED_POINT_AS_405B_CORE"

    total_block_sweeps = sum(
        int(row["block_sweeps"]) for row in build_rows + holdout_rows
    )
    total_block_positions = sum(
        int(row["processed_token_positions"]) for row in build_rows + holdout_rows
    )
    total_prefill_calls = sum(int(trace["prefill_target_calls"]) for trace in case_traces)
    total_ar_calls = sum(int(trace["ar_target_calls"]) for trace in case_traces)
    first_fraction = logical_weight_fraction(1, holdout_min_first)
    first_fraction_favorable = logical_weight_fraction(
        1, holdout_min_first, compression_ratio=compression_ratio
    )

    deterministic_core = {
        "schema": config["schema"],
        "mechanism_fingerprint": "official-target-jacobi/128-token-causal-block/three-frozen-causal-seeds/eight-sweeps/self-consistent-exact-prefix/build-selection/untouched-holdout",
        "config_sha256": sha256_file(config_path),
        "checkpoint": identity,
        "controls": controls,
        "build_rows": build_rows,
        "build_aggregates": build_aggregates,
        "selection": selection.to_dict(),
        "holdout_rows": holdout_rows,
        "holdout_aggregate": holdout_aggregate,
        "case_traces": case_traces,
        "target_equation": {
            **target_projection,
            "observed_holdout_minimum_first_sweep_accepted": holdout_min_first,
            "observed_first_sweep_fraction_no_compression": first_fraction,
            "observed_first_sweep_fraction_favorable_compression": first_fraction_favorable,
            "two_sweeps_full_block_fraction_no_compression": logical_weight_fraction(
                2, block_length
            ),
            "two_sweeps_full_block_fraction_favorable_compression": logical_weight_fraction(
                2, block_length, compression_ratio=compression_ratio
            ),
            "target_arithmetic_per_block_sweep_fraction": 1.0,
        },
        "resource_trace": {
            "prefill_target_calls": total_prefill_calls,
            "delayed_ar_control_calls": total_ar_calls,
            "jacobi_block_sweeps": total_block_sweeps,
            "jacobi_processed_token_positions": total_block_positions,
            "block_length": block_length,
        },
        "gates": {
            "integrity_passed": integrity_passed,
            "integrity_failures": integrity_failures,
            "selected_mode": selected_mode,
            "holdout_started_after_selection": holdout_started_after_selection,
            "holdout_minimum_first_sweep_accepted": holdout_min_first,
            "no_compression_signal": no_compression_signal,
            "favorable_compression_signal": favorable_signal,
            "target_arithmetic_reduced": False,
        },
        "authoritative_decision": decision,
    }

    result = {
        **deterministic_core,
        "experiment": "EXP-091A",
        "name": "exact_jacobi_fixed_point_gate",
        "source_sha": os.getenv("GITHUB_SHA", "LOCAL_UNVERIFIED"),
        "evidence_level": "E3_CAUSAL_HELDOUT_EXACT_PREFIX_SIGNAL",
        "runtime": runtime,
        "timing": {
            "model_load_ns": model_load_ns,
            "experiment_ns": experiment_ns,
            "selection_frozen_ns": selection_frozen_ns,
        },
        "deterministic_core_sha256": canonical_sha256(deterministic_core),
        "claim_boundary": {
            "official_public_checkpoint_loaded": True,
            "unchanged_target_block_operator": True,
            "trajectory_completed_before_ar_control": integrity_passed,
            "self_consistent_prefix_is_exact": integrity_passed,
            "target_405b_checkpoint": "NOT_TESTED",
            "target_8gib_residency": "NOT_TESTED",
            "maximum_context_kv": "NOT_TESTED",
            "physical_checkpoint_stream": "NOT_TESTED",
            "same_machine_4b_p50_p95": "NOT_TESTED",
        },
        "provenance": {
            "MEASURED": [
                "DEV-W seed trajectories, block proposals, self-consistent exact prefixes, fixed points, delayed AR controls, call and position counts"
            ],
            "DERIVED": [
                "exact-prefix soundness from causality and measured equality; checkpoint fractions from observed committed tokens"
            ],
            "PROJECTED": [
                "405B logical checkpoint weight fraction if DEV-W first-sweep acceptance generalized"
            ],
            "UNVERIFIED": [
                "TARGET-W acceptance, physical streamed block executor, KV/storage/PCIe/HBM costs and latency"
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "artifacts/checkpoint_identity.json", identity)
    write_json(output_dir / "artifacts/deterministic_core.json", deterministic_core)
    write_json(output_dir / "raw/build_rows.json", build_rows)
    write_json(output_dir / "raw/holdout_rows.json", holdout_rows)
    write_json(output_dir / "raw/case_traces.json", case_traces)
    write_json(output_dir / "result.json", result)
    write_checksums(
        output_dir,
        [
            "artifacts/checkpoint_identity.json",
            "artifacts/deterministic_core.json",
            "raw/build_rows.json",
            "raw/holdout_rows.json",
            "raw/case_traces.json",
            "result.json",
        ],
    )
    print(
        json.dumps(
            {
                "authoritative_decision": decision,
                "selected_mode": selected_mode,
                "build_minimum_first_sweep": selection.minimum_first_sweep_accepted,
                "holdout_first_sweep_accepted": holdout_aggregate[
                    "first_sweep_accepted"
                ],
                "holdout_best_accepted": holdout_aggregate["best_accepted"],
                "holdout_fixed_point_sweeps": holdout_aggregate[
                    "fixed_point_sweeps"
                ],
                "required_no_compression": no_compression_required,
                "required_favorable": favorable_required,
                "observed_fraction_no_compression": first_fraction,
                "target_arithmetic_fraction_per_sweep": 1.0,
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
