from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    PINNED_SAFETENSORS,
    PINNED_TORCH,
    PINNED_TRANSFORMERS,
    config_dict,
    model_tensor_sha256,
)
from vortex_runtime.prefix_state_bisimulation import (  # noqa: E402
    PrefixBisimulationError,
    PrefixCompiledState,
    PrefixRNGState,
    _flatten_cache,
    append_exact_token,
    cache_nbytes,
    cache_sha256,
    first_byte_mismatch,
    initialize_prefix_state,
    replay_prefix,
    select_token,
    sha256_json,
    tensor_bytes,
    tensor_sha256,
)


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
    if config.get("schema") != "exp-089a-prefix-state-bisimulation-v1":
        raise RuntimeError("unexpected config schema")
    if not config.get("frozen_before_results"):
        raise RuntimeError("experiment was not frozen before results")
    if config.get("model_id") != DEV_MODEL_ID or config.get("revision") != DEV_REVISION:
        raise RuntimeError("frozen DEV-W identity mismatch")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("case population is empty")
    case_ids = [str(row["id"]) for row in cases]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("duplicate case IDs")
    total = sum(int(row["steps"]) for row in cases)
    if total < int(config["success"]["minimum_total_transitions"]):
        raise RuntimeError("configured transition population is below the frozen minimum")
    for row in cases:
        if row["mode"] not in {"greedy", "sample"}:
            raise RuntimeError(f"unsupported case mode: {row['mode']}")
        if int(row["steps"]) <= 0:
            raise RuntimeError("case steps must be positive")
        if not str(row["prompt"]):
            raise RuntimeError("case prompt is empty")


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


def rng_for_case(case: dict[str, Any]) -> PrefixRNGState:
    if case["mode"] == "greedy":
        return PrefixRNGState.greedy()
    return PrefixRNGState.sampled(
        int(case["seed"]),
        temperature=float(case["temperature"]),
        top_k=int(case["top_k"]),
    )


def compare_caches(reference: Any, replay: Any) -> dict[str, Any] | None:
    left = _flatten_cache(reference)
    right = _flatten_cache(replay)
    if len(left) != len(right):
        return {
            "kind": "tensor_count",
            "reference": len(left),
            "replay": len(right),
        }
    for index, ((left_path, left_tensor), (right_path, right_tensor)) in enumerate(
        zip(left, right)
    ):
        if left_path != right_path:
            return {
                "kind": "path",
                "tensor_index": index,
                "reference": left_path,
                "replay": right_path,
            }
        if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
            return {
                "kind": "metadata",
                "tensor_index": index,
                "path": left_path,
                "reference_shape": list(left_tensor.shape),
                "replay_shape": list(right_tensor.shape),
                "reference_dtype": str(left_tensor.dtype),
                "replay_dtype": str(right_tensor.dtype),
            }
        mismatch = first_byte_mismatch(tensor_bytes(left_tensor), tensor_bytes(right_tensor))
        if mismatch is not None:
            return {
                "kind": "bytes",
                "tensor_index": index,
                "path": left_path,
                **mismatch,
            }
    return None


def tokenizer_manifest(tokenizer: Any) -> dict[str, Any]:
    payload = {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0)),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "init_kwargs": getattr(tokenizer, "init_kwargs", {}),
    }
    return {**payload, "sha256": sha256_json(payload)}


def compile_identity(model: Any, tokenizer: Any, config: dict[str, Any]) -> dict[str, Any]:
    config_payload = config_dict(model.config)
    return {
        "format": "prefix-state-bisimulation-identity-v1",
        "model_id": config["model_id"],
        "requested_revision": config["revision"],
        "resolved_revision": str(getattr(model.config, "_commit_hash", "") or ""),
        "model_class": type(model).__name__,
        "dtype": config["dtype"],
        "attention_implementation": config["attention_implementation"],
        "config_sha256": sha256_json(config_payload),
        "checkpoint_tensor_sha256": model_tensor_sha256(model),
        "tokenizer": tokenizer_manifest(tokenizer),
    }


def reference_prefill(model: Any, prompt_ids: Sequence[int]) -> tuple[torch.Tensor, Any]:
    input_ids = torch.tensor([list(map(int, prompt_ids))], dtype=torch.long)
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return output.logits[:, -1, :].detach().contiguous().cpu(), output.past_key_values


def reference_advance(model: Any, token: int, past: Any) -> tuple[torch.Tensor, Any]:
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[int(token)]], dtype=torch.long),
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
    return output.logits[:, -1, :].detach().contiguous().cpu(), output.past_key_values


def run_case(
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt_ids = tuple(
        int(value)
        for value in tokenizer(case["prompt"], return_tensors="pt")["input_ids"][0].tolist()
    )
    initial_rng = rng_for_case(case)
    compiled = initialize_prefix_state(prompt_ids, initial_rng)
    reference_rng = initial_rng
    reference_logits, reference_cache = reference_prefill(model, prompt_ids)
    reference_target_calls = 1

    rows: list[dict[str, Any]] = []
    generated_tokens: list[int] = []
    token_mismatches = 0
    logits_byte_mismatches = 0
    cache_mismatches = 0
    rng_mismatches = 0
    replay_target_calls = 0
    replayed_positions = 0

    for step in range(int(case["steps"])):
        replay = replay_prefix(model, compiled)
        replay_target_calls += int(replay.target_calls)
        replayed_positions += int(replay.replayed_token_positions)

        logits_mismatch = first_byte_mismatch(
            tensor_bytes(reference_logits), tensor_bytes(replay.logits)
        )
        cache_mismatch = compare_caches(reference_cache, replay.past_key_values)
        if logits_mismatch is not None:
            logits_byte_mismatches += 1
        if cache_mismatch is not None:
            cache_mismatches += 1

        reference_token, next_reference_rng = select_token(reference_logits, reference_rng)
        replay_token, next_replay_rng = select_token(replay.logits, compiled.rng_state)
        token_equal = reference_token == replay_token
        rng_equal = next_reference_rng.digest() == next_replay_rng.digest()
        if not token_equal:
            token_mismatches += 1
        if not rng_equal:
            rng_mismatches += 1

        rows.append(
            {
                "case_id": case["id"],
                "family": case["family"],
                "mode": case["mode"],
                "step": step,
                "prefix_sha256": compiled.digest(),
                "prefix_length": compiled.position,
                "reference_logits_sha256": tensor_sha256(reference_logits),
                "replay_logits_sha256": replay.logits_sha256,
                "logits_first_mismatch": logits_mismatch,
                "reference_cache_sha256": cache_sha256(reference_cache),
                "replay_cache_sha256": replay.cache_sha256,
                "cache_first_mismatch": cache_mismatch,
                "cache_bytes": cache_nbytes(reference_cache),
                "reference_token": reference_token,
                "replay_token": replay_token,
                "token_equal": token_equal,
                "reference_rng_before_sha256": reference_rng.digest(),
                "replay_rng_before_sha256": compiled.rng_state.digest(),
                "reference_rng_after_sha256": next_reference_rng.digest(),
                "replay_rng_after_sha256": next_replay_rng.digest(),
                "rng_equal": rng_equal,
                "replay_target_calls": replay.target_calls,
                "replayed_token_positions": replay.replayed_token_positions,
            }
        )

        if not token_equal or logits_mismatch is not None or cache_mismatch is not None or not rng_equal:
            raise RuntimeError(
                f"prefix-state witness mismatch in {case['id']} step {step}: "
                f"token={token_equal} logits={logits_mismatch} cache={cache_mismatch} rng={rng_equal}"
            )

        generated_tokens.append(reference_token)
        compiled = append_exact_token(compiled, replay_token, next_replay_rng)
        reference_rng = next_reference_rng
        reference_logits, reference_cache = reference_advance(
            model, reference_token, reference_cache
        )
        reference_target_calls += 1

    # Terminal successor-state witness after the last selected token. This closes
    # transition preservation for the final configured decision as well.
    terminal = replay_prefix(model, compiled)
    replay_target_calls += int(terminal.target_calls)
    replayed_positions += int(terminal.replayed_token_positions)
    terminal_logits_mismatch = first_byte_mismatch(
        tensor_bytes(reference_logits), tensor_bytes(terminal.logits)
    )
    terminal_cache_mismatch = compare_caches(reference_cache, terminal.past_key_values)
    terminal_rng_equal = reference_rng.digest() == compiled.rng_state.digest()
    if terminal_logits_mismatch is not None:
        logits_byte_mismatches += 1
    if terminal_cache_mismatch is not None:
        cache_mismatches += 1
    if not terminal_rng_equal:
        rng_mismatches += 1
    if terminal_logits_mismatch is not None or terminal_cache_mismatch is not None or not terminal_rng_equal:
        raise RuntimeError(f"terminal successor-state mismatch in {case['id']}")

    summary = {
        "case_id": case["id"],
        "family": case["family"],
        "mode": case["mode"],
        "prompt_sha256": sha256_bytes(case["prompt"].encode("utf-8")),
        "prompt_token_count": len(prompt_ids),
        "steps": int(case["steps"]),
        "generated_tokens": generated_tokens,
        "generated_sha256": sha256_json(generated_tokens),
        "token_mismatches": token_mismatches,
        "logits_byte_mismatches": logits_byte_mismatches,
        "cache_digest_mismatches": cache_mismatches,
        "rng_mismatches": rng_mismatches,
        "terminal_logits_sha256": terminal.logits_sha256,
        "terminal_cache_sha256": terminal.cache_sha256,
        "terminal_rng_sha256": compiled.rng_state.digest(),
        "terminal_prefix_sha256": compiled.digest(),
        "reference_target_calls": reference_target_calls,
        "replay_target_calls": replay_target_calls,
        "replayed_token_positions": replayed_positions,
        "terminal_cache_bytes": terminal.cache_bytes,
    }
    return summary, rows


def controls() -> dict[str, Any]:
    greedy = PrefixRNGState.greedy()
    base = initialize_prefix_state([1, 2, 3], greedy)
    changed = append_exact_token(base, 4, greedy)
    prefix_mutation_detected = base.digest() != changed.digest()

    sampled = PrefixRNGState.sampled(1234, temperature=0.8, top_k=4)
    logits = torch.tensor([0.1, 0.2, 1.7, -0.4, 0.9], dtype=torch.float32)
    token_a, next_a = select_token(logits, sampled)
    token_b, next_b = select_token(logits.clone(), sampled)
    rng_replay_exact = token_a == token_b and next_a.digest() == next_b.digest()

    payload = tensor_bytes(torch.tensor([1.0, -2.0], dtype=torch.float32))
    altered = bytearray(payload)
    altered[0] ^= 0x01
    byte_fault_detected = first_byte_mismatch(payload, bytes(altered)) is not None

    empty_prompt_failed = False
    invalid_temperature_failed = False
    try:
        initialize_prefix_state([], greedy)
    except PrefixBisimulationError:
        empty_prompt_failed = True
    try:
        PrefixRNGState.sampled(1, temperature=0.0)
    except PrefixBisimulationError:
        invalid_temperature_failed = True

    passed = all(
        [
            prefix_mutation_detected,
            rng_replay_exact,
            byte_fault_detected,
            empty_prompt_failed,
            invalid_temperature_failed,
        ]
    )
    return {
        "prefix_mutation_detected": prefix_mutation_detected,
        "rng_replay_exact": rng_replay_exact,
        "sampled_token": token_a,
        "byte_fault_detected": byte_fault_detected,
        "empty_prompt_failed_closed": empty_prompt_failed,
        "invalid_temperature_failed_closed": invalid_temperature_failed,
        "passed": passed,
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    runtime = verify_runtime()
    torch.manual_seed(0)
    torch.set_num_threads(int(config["torch_num_threads"]))
    model, tokenizer, load_wall_ns = load_model(config)

    compile_start = time.perf_counter_ns()
    identity = compile_identity(model, tokenizer, config)
    compile_wall_ns = time.perf_counter_ns() - compile_start
    control_rows = controls()

    case_summaries: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    run_start = time.perf_counter_ns()
    for case in config["cases"]:
        summary, rows = run_case(model, tokenizer, case)
        case_summaries.append(summary)
        transition_rows.extend(rows)
        print(
            f"{case['id']}: transitions={summary['steps']} "
            f"token/logits/cache/rng mismatches="
            f"{summary['token_mismatches']}/{summary['logits_byte_mismatches']}/"
            f"{summary['cache_digest_mismatches']}/{summary['rng_mismatches']}",
            flush=True,
        )
    run_wall_ns = time.perf_counter_ns() - run_start

    totals = {
        "transitions": sum(int(row["steps"]) for row in case_summaries),
        "token_mismatches": sum(int(row["token_mismatches"]) for row in case_summaries),
        "logits_byte_mismatches": sum(
            int(row["logits_byte_mismatches"]) for row in case_summaries
        ),
        "cache_digest_mismatches": sum(
            int(row["cache_digest_mismatches"]) for row in case_summaries
        ),
        "rng_mismatches": sum(int(row["rng_mismatches"]) for row in case_summaries),
        "reference_target_calls": sum(
            int(row["reference_target_calls"]) for row in case_summaries
        ),
        "replay_target_calls": sum(
            int(row["replay_target_calls"]) for row in case_summaries
        ),
        "replayed_token_positions": sum(
            int(row["replayed_token_positions"]) for row in case_summaries
        ),
    }
    success = config["success"]
    gates = {
        "runtime_identity": identity["resolved_revision"] in {"", config["revision"]},
        "controls": bool(control_rows["passed"]),
        "minimum_transitions": totals["transitions"]
        >= int(success["minimum_total_transitions"]),
        "tokens": totals["token_mismatches"]
        <= int(success["maximum_token_mismatches"]),
        "logits": totals["logits_byte_mismatches"]
        <= int(success["maximum_logits_byte_mismatches"]),
        "cache": totals["cache_digest_mismatches"]
        <= int(success["maximum_cache_digest_mismatches"]),
        "rng": totals["rng_mismatches"] <= int(success["maximum_rng_mismatches"]),
    }
    accepted = all(gates.values())
    decision = (
        "ACCEPT_PREFIX_STATE_BISIMULATION_WITNESS_FOR_TOKEN_DECISION_RESEARCH"
        if accepted
        else "INVALID_PREFIX_STATE_BISIMULATION_WITNESS"
    )

    deterministic_core = {
        "schema": config["schema"],
        "config_sha256": sha256_file(config_path),
        "identity": identity,
        "cases": case_summaries,
        "totals": totals,
        "controls": control_rows,
        "gates": gates,
        "authoritative_decision": decision,
    }
    result = {
        **deterministic_core,
        "experiment": "EXP-089A",
        "name": "prefix_state_bisimulation_gate",
        "source_sha": os.getenv("GITHUB_SHA", "LOCAL_UNVERIFIED"),
        "evidence_level": "E2_STATE_WITNESS",
        "runtime": runtime,
        "timing": {
            "model_load_ns": load_wall_ns,
            "identity_compile_ns": compile_wall_ns,
            "case_run_ns": run_wall_ns,
        },
        "deterministic_core_sha256": sha256_bytes(
            json.dumps(
                deterministic_core,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ),
        "claim_boundary": {
            "official_checkpoint_loaded": True,
            "official_incremental_reference_executed": True,
            "prefix_replay_executed": True,
            "raw_logits_compared": True,
            "raw_cache_compared": True,
            "rng_bytes_compared": True,
            "replay_is_fast_executor": False,
            "dense_fallback_for_final_core": "PROHIBITED",
            "exact_token_decision_program": "NOT_TESTED",
            "128_step_per_case_performance_gate": "NOT_TESTED",
            "405b": "NOT_TESTED",
            "8gib_gpu": "NOT_TESTED",
            "physical_latency": "NOT_TESTED",
        },
        "provenance": {
            "MEASURED": [
                "DEV-W reference/replay logits bytes, cache bytes, RNG bytes, tokens, call counts, replayed positions"
            ],
            "DERIVED": [
                "prefix-state bisimulation acceptance from the frozen byte-equality obligations"
            ],
            "PROJECTED": [],
            "UNVERIFIED": [
                "cheap exact token-decision compiler, TARGET-W, target VRAM and latency"
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "artifacts/checkpoint_identity.json", identity)
    write_json(output_dir / "artifacts/deterministic_core.json", deterministic_core)
    write_json(output_dir / "raw/case_summaries.json", case_summaries)
    write_json(output_dir / "raw/transition_rows.json", transition_rows)
    write_json(output_dir / "result.json", result)
    write_checksums(
        output_dir,
        [
            "artifacts/checkpoint_identity.json",
            "artifacts/deterministic_core.json",
            "raw/case_summaries.json",
            "raw/transition_rows.json",
            "result.json",
        ],
    )
    print(
        json.dumps(
            {
                "authoritative_decision": decision,
                "transitions": totals["transitions"],
                "token_mismatches": totals["token_mismatches"],
                "logits_byte_mismatches": totals["logits_byte_mismatches"],
                "cache_digest_mismatches": totals["cache_digest_mismatches"],
                "rng_mismatches": totals["rng_mismatches"],
                "reference_target_calls": totals["reference_target_calls"],
                "replay_target_calls": totals["replay_target_calls"],
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
