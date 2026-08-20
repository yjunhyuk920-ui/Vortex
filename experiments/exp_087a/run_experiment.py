from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

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
)
from vortex_runtime.quadratic_bf16_residual_generator import (  # noqa: E402
    P50_WHOLE_MODEL_FRACTION,
    bf16_words,
    exact_report,
    fit_library,
    synthetic_quadratic_control,
    target_resources,
    tensor_sha256,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def runtime_manifest() -> dict[str, str]:
    import safetensors

    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
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
        raise RuntimeError(f"pinned runtime mismatch: {mismatch}")
    return actual


def load_model(config: dict[str, Any]) -> tuple[Any, Any, int]:
    if config["model_id"] != DEV_MODEL_ID or config["revision"] != DEV_REVISION:
        raise RuntimeError("frozen DEV-W identity mismatch")
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


def capture_causal_pairs(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_index: int,
    positions: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    inputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []

    def pre_hook(_module: Any, arguments: tuple[torch.Tensor, ...]) -> None:
        inputs.append(arguments[0][:, -1, :].detach().cpu().contiguous())

    def post_hook(
        _module: Any,
        _arguments: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        outputs.append(output[:, -1, :].detach().cpu().contiguous())

    mlp = model.model.layers[int(layer_index)].mlp
    pre_handle = mlp.register_forward_pre_hook(pre_hook)
    post_handle = mlp.register_forward_hook(post_hook)
    generated: list[int] = []
    try:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            result = model(input_ids=input_ids, use_cache=True, return_dict=True)
            past = result.past_key_values
            next_token = int(result.logits[0, -1].argmax().item())
            generated.append(next_token)
            for _ in range(1, int(positions)):
                result = model(
                    input_ids=torch.tensor([[next_token]], dtype=torch.long),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = result.past_key_values
                next_token = int(result.logits[0, -1].argmax().item())
                generated.append(next_token)
    finally:
        pre_handle.remove()
        post_handle.remove()
    if len(inputs) != positions or len(outputs) != positions:
        raise RuntimeError(
            f"causal capture mismatch: inputs={len(inputs)} outputs={len(outputs)} expected={positions}"
        )
    return (
        torch.cat(inputs, dim=0).to(torch.bfloat16),
        torch.cat(outputs, dim=0).to(torch.bfloat16),
        generated,
    )


def residual_diagnostics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    residual = torch.bitwise_xor(bf16_words(reference), bf16_words(candidate))
    hashes = [
        tensor_sha256(row.contiguous())
        for row in residual
    ]
    unique_per_coordinate = [
        int(torch.unique(residual[:, column]).numel())
        for column in range(residual.shape[1])
    ]
    ordered = sorted(unique_per_coordinate)
    p50 = ordered[(len(ordered) - 1) // 2]
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "unique_residual_vector_fraction": len(set(hashes)) / max(1, residual.shape[0]),
        "zero_word_fraction": float((residual == 0).float().mean().item()),
        "unique_words_per_coordinate_p50": int(p50),
        "unique_words_per_coordinate_p95": int(p95),
    }


def query_record(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": query["mode"],
        "selected": query["selected"],
        "use_counts": query["use_counts"],
        "report": query["report"],
        "candidate_sha256": tensor_sha256(query["candidate"]),
    }


def program_manifest(program: Any) -> dict[str, Any]:
    return {
        "build_count": program.build_count,
        "context_bits": program.context_bits,
        "feature_count": program.feature_count,
        "feature_rank": program.feature_rank,
        "fingerprint": program.fingerprint,
        "input_mean_sha256": tensor_sha256(program.input_mean),
        "context_basis_sha256": tensor_sha256(program.context_basis),
        "baseline_words_sha256": tensor_sha256(program.baseline_words),
        "coefficient_words_sha256": tensor_sha256(program.coefficient_words),
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = runtime_manifest()
    torch.manual_seed(int(config["seed"]))
    torch.set_num_threads(int(config["torch_num_threads"]))
    model, tokenizer, load_wall_ns = load_model(config)

    build_x: list[torch.Tensor] = []
    build_y: list[torch.Tensor] = []
    evaluation_x: list[torch.Tensor] = []
    evaluation_y: list[torch.Tensor] = []
    trace_rows: list[dict[str, Any]] = []
    capture_start = time.perf_counter_ns()
    for prompt_spec in config["prompts"]:
        x, y, tokens = capture_causal_pairs(
            model,
            tokenizer,
            prompt_spec["prompt"],
            int(config["layer_index"]),
            int(config["decode_positions_per_prompt"]),
        )
        destination_x = build_x if prompt_spec["split"] == "build" else evaluation_x
        destination_y = build_y if prompt_spec["split"] == "build" else evaluation_y
        destination_x.append(x)
        destination_y.append(y)
        trace_rows.append(
            {
                "id": prompt_spec["id"],
                "family": prompt_spec["family"],
                "split": prompt_spec["split"],
                "prompt_sha256": sha256_bytes(prompt_spec["prompt"].encode()),
                "state_count": int(x.shape[0]),
                "input_sha256": tensor_sha256(x),
                "output_sha256": tensor_sha256(y),
                "generated_tokens": tokens,
            }
        )
    capture_wall_ns = time.perf_counter_ns() - capture_start
    x_build = torch.cat(build_x, dim=0)
    y_build = torch.cat(build_y, dim=0)
    x_evaluation = torch.cat(evaluation_x, dim=0)
    y_evaluation = torch.cat(evaluation_y, dim=0)

    control = synthetic_quadratic_control(int(config["seed"]) + 17)
    compile_start = time.perf_counter_ns()
    library, assignments, build_program_exact = fit_library(
        x_build,
        y_build,
        program_count=int(config["program_count"]),
        context_bits=int(config["context_bits"]),
        context_pool=int(config["context_pool"]),
        seed=int(config["seed"]),
    )
    compile_wall_ns = time.perf_counter_ns() - compile_start

    raw_queries = {
        "build_oracle": library.query(x_build, y_build, oracle=True),
        "build_router": library.query(x_build, y_build, oracle=False),
        "evaluation_oracle": library.query(x_evaluation, y_evaluation, oracle=True),
        "evaluation_router": library.query(x_evaluation, y_evaluation, oracle=False),
    }
    queries = {name: query_record(value) for name, value in raw_queries.items()}
    resources = target_resources(
        program_count=int(config["program_count"]),
        context_bits=int(config["context_bits"]),
    )
    evaluation_used = [
        count
        for count in raw_queries["evaluation_oracle"]["use_counts"]
        if count > 0
    ]

    integrity_passed = bool(control["consistent"]) and control["mismatches"] == 0
    build_passed = (
        all(build_program_exact)
        and queries["build_oracle"]["report"]["vector_exact_fraction"]
        >= float(config["success"]["build_vector_exact_fraction_min"])
    )
    resource_passed = (
        resources["sidecar_bytes"] <= int(config["target_sidecar_limit_bytes"])
        and resources["compiled_mlp_operation_fraction"]
        <= float(config["success"]["compiled_mlp_operation_fraction_max"])
    )
    oracle_passed = (
        queries["evaluation_oracle"]["report"]["vector_exact_fraction"]
        >= float(config["success"]["oracle_vector_exact_fraction_min"])
        and bool(evaluation_used)
        and min(evaluation_used)
        >= int(config["success"]["minimum_evaluation_states_per_used_program"])
    )
    router_passed = (
        queries["evaluation_router"]["report"]["vector_exact_fraction"]
        >= float(config["success"]["router_vector_exact_fraction_min"])
    )

    if not integrity_passed:
        decision = "INVALID_QUADRATIC_BF16_RESIDUAL_GENERATOR_CONTROL_FAILURE"
    elif not build_passed:
        decision = "REJECT_QUADRATIC_BF16_RESIDUAL_GENERATOR_BUILD_GATE"
    elif not resource_passed:
        decision = "REJECT_QUADRATIC_BF16_RESIDUAL_GENERATOR_RESOURCE_GATE"
    elif not oracle_passed:
        decision = "REJECT_QUADRATIC_BF16_RESIDUAL_GENERATOR_AT_ORACLE_GATE"
    elif not router_passed:
        decision = "PROMOTE_QUADRATIC_BF16_RESIDUAL_LIBRARY_TO_CAUSAL_ROUTER_GATE"
    else:
        decision = "PROMOTE_QUADRATIC_BF16_RESIDUAL_GENERATOR_TO_COMPLETE_MLP_GATE"

    manifest = {
        "format": "quadratic-bf16-residual-generator-v1",
        "mechanism": "cross_weight_context_conditioned_gf2_quadratic_bf16_word_residual_generator",
        "library_fingerprint": library.fingerprint,
        "program_count": len(library.programs),
        "context_bits": int(config["context_bits"]),
        "programs": [program_manifest(program) for program in library.programs],
        "partition_sha256": tensor_sha256(assignments),
        "forbidden_shortcuts": [
            "exact state key",
            "prefix or KV lookup table",
            "target forward inside query",
            "dense MLP opcode inside query",
            "row/page fallback",
        ],
    }
    core = {
        "config_sha256": file_sha256(config_path),
        "manifest": manifest,
        "trace_rows": trace_rows,
        "queries": queries,
        "resources": resources,
        "controls": {"synthetic_quadratic": control},
        "gates": {
            "integrity": integrity_passed,
            "build": build_passed,
            "resource": resource_passed,
            "oracle": oracle_passed,
            "router": router_passed,
        },
        "decision": decision,
    }
    result = {
        "experiment": "EXP-087A",
        "name": "quadratic_bf16_residual_generator_gate",
        "source_sha": os.getenv("GITHUB_SHA", "LOCAL_UNVERIFIED"),
        "evidence_level": "E1",
        "runtime": runtime,
        "load_wall_ns": load_wall_ns,
        "capture_wall_ns": capture_wall_ns,
        "compile_wall_ns": compile_wall_ns,
        "config_sha256": file_sha256(config_path),
        "manifest": manifest,
        "MEASURED": {
            "build_state_count": int(x_build.shape[0]),
            "evaluation_state_count": int(x_evaluation.shape[0]),
            "build_assignment_counts": [
                int(value)
                for value in torch.bincount(
                    assignments, minlength=int(config["program_count"])
                ).tolist()
            ],
            "build_program_exact": build_program_exact,
            "queries": queries,
            "residual_diagnostics": {
                "oracle": residual_diagnostics(
                    y_evaluation, raw_queries["evaluation_oracle"]["candidate"]
                ),
                "router": residual_diagnostics(
                    y_evaluation, raw_queries["evaluation_router"]["candidate"]
                ),
            },
            "controls": {"synthetic_quadratic": control},
        },
        "DERIVED": {
            "target_resources": resources,
            "minimum_service_tokens_to_amortize_build_transitions": math.ceil(
                int(x_build.shape[0]) / P50_WHOLE_MODEL_FRACTION
            ),
        },
        "gates": core["gates"],
        "authoritative_decision": decision,
        "deterministic_core_sha256": sha256_bytes(canonical_json(core)),
        "claim_boundary": {
            "official_checkpoint_loaded": True,
            "actual_causal_states": True,
            "disjoint_build_evaluation_prompts": True,
            "checkpoint_weights_modified": False,
            "state_lookup_table": False,
            "oracle_deployable": False,
            "complete_mlp_replaced": False,
            "complete_layer": "NOT_TESTED",
            "successor_state": "NOT_TESTED",
            "405b": "NOT_TESTED",
            "8gib_gpu": "NOT_TESTED",
            "physical_latency": "NOT_TESTED",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "artifacts").mkdir(exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "raw/trace_rows.json").write_text(
        json.dumps(trace_rows, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "raw/query_rows.json").write_text(
        json.dumps(queries, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "artifacts/compiler_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "artifacts/deterministic_core.json").write_bytes(canonical_json(core))
    paths = [
        "result.json",
        "raw/trace_rows.json",
        "raw/query_rows.json",
        "artifacts/compiler_manifest.json",
        "artifacts/deterministic_core.json",
    ]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(f"{file_sha256(output_dir / path)}  {path}" for path in paths) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args.config, args.output_dir)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "experiment": "EXP-087A",
            "source_sha": os.getenv("GITHUB_SHA", "LOCAL_UNVERIFIED"),
            "authoritative_decision": "INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION",
            "error": f"{type(error).__name__}: {error}",
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
