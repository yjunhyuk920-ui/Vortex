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
from vortex_runtime.functional_microprogram_compiler import (  # noqa: E402
    P50,
    fit_library,
    fit_library_from_partition,
    residual_diagnostics,
    target_resources,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fsha(path: Path) -> str:
    return sha(path.read_bytes())


def tsha(tensor: torch.Tensor) -> str:
    return sha(
        tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    )


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def runtime() -> dict[str, str]:
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
        key: [expected[key], actual[key]]
        for key in expected
        if expected[key] != actual[key]
    }
    if mismatch:
        raise RuntimeError(f"pin mismatch: {mismatch}")
    return actual


def load(config: dict[str, Any]) -> tuple[Any, Any, int]:
    if config["model_id"] != DEV_MODEL_ID or config["revision"] != DEV_REVISION:
        raise RuntimeError("DEV-W mismatch")
    start = time.perf_counter_ns()
    model = LlamaForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
    ).eval()
    resolved = str(getattr(model.config, "_commit_hash", "") or "")
    if resolved and resolved != config["revision"]:
        raise RuntimeError(f"resolved checkpoint mismatch {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["revision"], use_fast=True
    )
    return model, tokenizer, time.perf_counter_ns() - start


def capture(
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
        _module: Any, _arguments: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        outputs.append(output[:, -1, :].detach().cpu().contiguous())

    mlp = model.model.layers[layer_index].mlp
    pre_handle = mlp.register_forward_pre_hook(pre_hook)
    post_handle = mlp.register_forward_hook(post_hook)
    tokens: list[int] = []
    try:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            result = model(input_ids=input_ids, use_cache=True, return_dict=True)
            past = result.past_key_values
            next_token = int(result.logits[0, -1].argmax())
            tokens.append(next_token)
            for _ in range(1, positions):
                result = model(
                    input_ids=torch.tensor([[next_token]]),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = result.past_key_values
                next_token = int(result.logits[0, -1].argmax())
                tokens.append(next_token)
    finally:
        pre_handle.remove()
        post_handle.remove()
    if len(inputs) != positions or len(outputs) != positions:
        raise RuntimeError("capture count mismatch")
    return (
        torch.cat(inputs).bfloat16(),
        torch.cat(outputs).bfloat16(),
        tokens,
    )


def query_json(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": query["mode"],
        "precision": query["precision"],
        "selected": query["selected"],
        "use_counts": query["use_counts"],
        "exact_report": query["report"],
        "candidate_sha256": tsha(query["candidate"]),
    }


def library_manifest(library: Any) -> dict[str, Any]:
    return {
        "rank_cap": library.rank_cap,
        "fingerprint": library.fingerprint,
        "programs": [
            {
                "rank": program.rank,
                "build_count": program.build_count,
                "fingerprint": program.fingerprint,
            }
            for program in library.programs
        ],
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    runtime_info = runtime()
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["torch_num_threads"])
    model, tokenizer, load_ns = load(config)

    build_inputs: list[torch.Tensor] = []
    build_outputs: list[torch.Tensor] = []
    evaluation_inputs: list[torch.Tensor] = []
    evaluation_outputs: list[torch.Tensor] = []
    traces: list[dict[str, Any]] = []
    capture_start = time.perf_counter_ns()
    for specification in config["prompts"]:
        x, y, tokens = capture(
            model,
            tokenizer,
            specification["prompt"],
            config["layer_index"],
            config["decode_positions_per_prompt"],
        )
        if specification["split"] == "build":
            build_inputs.append(x)
            build_outputs.append(y)
        else:
            evaluation_inputs.append(x)
            evaluation_outputs.append(y)
        traces.append(
            {
                "id": specification["id"],
                "family": specification["family"],
                "split": specification["split"],
                "prompt_sha256": sha(specification["prompt"].encode()),
                "states": int(x.shape[0]),
                "input_sha256": tsha(x),
                "output_sha256": tsha(y),
                "tokens": tokens,
            }
        )
    capture_ns = time.perf_counter_ns() - capture_start
    x_build, y_build, x_evaluation, y_evaluation = map(
        torch.cat,
        (build_inputs, build_outputs, evaluation_inputs, evaluation_outputs),
    )

    compile_start = time.perf_counter_ns()
    candidate, assignments, centroids = fit_library(
        x_build,
        y_build,
        config["clusters"],
        config["rank_cap"],
    )
    # Control only: retain the same frozen partition but allow each cluster its
    # complete empirical rank. This verifies the SVD/interpolation/query path
    # without changing the rank-32 scientific candidate.
    control = fit_library_from_partition(
        x_build,
        y_build,
        assignments,
        centroids,
        rank_cap=int(x_build.shape[0]),
    )
    compile_ns = time.perf_counter_ns() - compile_start

    raw_queries = {
        "control_build_oracle_fp64": control.query(
            x_build, y_build, "fp64", True
        ),
        "control_build_router_fp64": control.query(
            x_build, y_build, "fp64", False
        ),
        "candidate_build_oracle_fp64": candidate.query(
            x_build, y_build, "fp64", True
        ),
        "candidate_build_router_fp32": candidate.query(
            x_build, y_build, "fp32", False
        ),
        "evaluation_oracle_fp64": candidate.query(
            x_evaluation, y_evaluation, "fp64", True
        ),
        "evaluation_oracle_fp32": candidate.query(
            x_evaluation, y_evaluation, "fp32", True
        ),
        "evaluation_router_fp32": candidate.query(
            x_evaluation, y_evaluation, "fp32", False
        ),
    }
    queries = {name: query_json(value) for name, value in raw_queries.items()}
    resources = target_resources(
        config["clusters"], config["rank_cap"], config["sidecar_scalar_bytes"]
    )
    build_counts = [
        int(value)
        for value in torch.bincount(
            assignments, minlength=config["clusters"]
        ).tolist()
    ]
    used = [
        value
        for value in raw_queries["evaluation_oracle_fp32"]["use_counts"]
        if value > 0
    ]

    integrity = (
        queries["control_build_oracle_fp64"]["exact_report"]
        ["vector_exact_fraction"]
        == 1.0
        and queries["control_build_router_fp64"]["exact_report"]
        ["vector_exact_fraction"]
        == 1.0
    )
    resource = (
        resources["sidecar_bytes"] <= config["target_sidecar_limit_bytes"]
        and resources["compiled_mlp_operation_fraction"]
        <= config["success"]["compiled_mlp_operation_fraction_max"]
    )
    oracle = (
        queries["evaluation_oracle_fp32"]["exact_report"]
        ["vector_exact_fraction"]
        == config["success"]["oracle_vector_exact_fraction_min"]
        and bool(used)
        and min(used)
        >= config["success"]["minimum_evaluation_states_per_used_program"]
    )
    router = (
        queries["evaluation_router_fp32"]["exact_report"]
        ["vector_exact_fraction"]
        == config["success"]["router_vector_exact_fraction_min"]
    )

    if not integrity:
        decision = "INVALID_CAUSAL_FUNCTIONAL_MICROPROGRAM_NUMERICAL_CONTROL_FAILURE"
    elif not resource:
        decision = "REJECT_CAUSAL_FUNCTIONAL_MICROPROGRAM_RESOURCE_GATE"
    elif not oracle:
        decision = "REJECT_LOW_RANK_FUNCTIONAL_MICROPROGRAM_LIBRARY_AT_ORACLE_GATE"
    elif not router:
        decision = "PROMOTE_FUNCTIONAL_MICROPROGRAM_LIBRARY_TO_CAUSAL_ROUTER_GATE"
    else:
        decision = "PROMOTE_FUNCTIONAL_MICROPROGRAM_TO_COMPLETE_MLP_REPLACEMENT_GATE"

    manifest = {
        "mechanism": "checkpoint_static_low_rank_affine_whole_swiglu_microprogram_library",
        "clusters": config["clusters"],
        "candidate": library_manifest(candidate),
        "full_rank_numerical_control": library_manifest(control),
        "control_partition_sha256": tsha(assignments),
    }
    core = {
        "config_sha256": fsha(config_path),
        "manifest": manifest,
        "traces": traces,
        "queries": queries,
        "resources": resources,
        "gates": {
            "integrity": integrity,
            "resource": resource,
            "oracle": oracle,
            "router": router,
        },
        "decision": decision,
    }
    result = {
        "experiment": "EXP-086A",
        "name": "causal_functional_microprogram_sharing_gate",
        "source_sha": os.getenv("GITHUB_SHA", "LOCAL_UNVERIFIED"),
        "evidence_level": "E1",
        "runtime": runtime_info,
        "load_wall_ns": load_ns,
        "capture_wall_ns": capture_ns,
        "compile_wall_ns": compile_ns,
        "config_sha256": fsha(config_path),
        "manifest": manifest,
        "MEASURED": {
            "build_state_count": int(x_build.shape[0]),
            "evaluation_state_count": int(x_evaluation.shape[0]),
            "build_assignment_counts": build_counts,
            "queries": queries,
            "residual": {
                "oracle": residual_diagnostics(
                    y_evaluation,
                    raw_queries["evaluation_oracle_fp32"]["candidate"],
                ),
                "router": residual_diagnostics(
                    y_evaluation,
                    raw_queries["evaluation_router_fp32"]["candidate"],
                ),
            },
        },
        "DERIVED": {
            "target_resources": resources,
            "minimum_service_tokens_to_amortize_build_forwards": math.ceil(
                x_build.shape[0] / P50
            ),
        },
        "gates": core["gates"],
        "authoritative_decision": decision,
        "deterministic_core_sha256": sha(canonical(core)),
        "control_correction": {
            "previous_source_sha": "376700afc276cb30b5eece61d3c1fb7e2e5b6ccd",
            "previous_classification": "INVALID_CAUSAL_FUNCTIONAL_MICROPROGRAM_NUMERICAL_CONTROL_FAILURE",
            "correction": "separate full-rank interpolation integrity control from the frozen rank-32 candidate",
            "candidate_population_or_rank_changed": False,
        },
        "claim_boundary": {
            "official_checkpoint_loaded": True,
            "actual_causal_states": True,
            "disjoint_build_evaluation_prompts": True,
            "state_lookup_table": False,
            "oracle_deployable": False,
            "complete_mlp_replaced": False,
            "complete_layer": "NOT_TESTED",
            "405b": "NOT_TESTED",
            "8gib_gpu": "NOT_TESTED",
            "latency": "NOT_TESTED",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "artifacts").mkdir(exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    (output_dir / "raw/trace_rows.json").write_text(
        json.dumps(traces, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    (output_dir / "raw/query_rows.json").write_text(
        json.dumps(queries, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    (output_dir / "artifacts/library_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "artifacts/deterministic_core.json").write_bytes(
        canonical(core)
    )
    paths = [
        "result.json",
        "raw/trace_rows.json",
        "raw/query_rows.json",
        "artifacts/library_manifest.json",
        "artifacts/deterministic_core.json",
    ]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(f"{fsha(output_dir / path)}  {path}" for path in paths) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = execute(arguments.config, arguments.output_dir)
    except Exception as error:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "experiment": "EXP-086A",
            "source_sha": os.getenv("GITHUB_SHA", "LOCAL_UNVERIFIED"),
            "authoritative_decision": "INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION",
            "error": f"{type(error).__name__}: {error}",
        }
        (arguments.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
