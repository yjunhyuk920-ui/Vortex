from __future__ import annotations

import argparse
import hashlib
import json
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
from vortex_runtime.joint_exact_swiglu_compiler import (  # noqa: E402
    LLAMA405_MLP_PARAMETER_SHARE,
    P50_WHOLE_MODEL_FRACTION,
    P95_WHOLE_MODEL_FRACTION,
    aggregate_query_rows,
    compile_joint_exact_swiglu,
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_runtime() -> dict[str, str]:
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
    mismatch = {key: [expected[key], actual[key]] for key in expected if expected[key] != actual[key]}
    if mismatch:
        raise RuntimeError(f"runtime pin mismatch: {mismatch}")
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
    )
    model.eval()
    resolved = str(getattr(model.config, "_commit_hash", "") or "")
    if resolved and resolved != config["revision"]:
        raise RuntimeError(f"resolved checkpoint mismatch {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["revision"], use_fast=True
    )
    return model, tokenizer, time.perf_counter_ns() - start


def capture_states(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    layer_index: int,
    positions: int,
) -> tuple[list[torch.Tensor], list[int]]:
    captured: list[torch.Tensor] = []

    def hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        hidden = inputs[0]
        captured.append(hidden[:, -1, :].detach().contiguous().cpu())

    layer = model.model.layers[int(layer_index)]
    handle = layer.mlp.register_forward_pre_hook(hook)
    tokens: list[int] = []
    try:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            output = model(input_ids=input_ids, use_cache=True, return_dict=True)
            past = output.past_key_values
            next_token = int(torch.argmax(output.logits[0, -1]).item())
            tokens.append(next_token)
            for _ in range(1, int(positions)):
                token_tensor = torch.tensor([[next_token]], dtype=torch.long)
                output = model(
                    input_ids=token_tensor,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = output.past_key_values
                next_token = int(torch.argmax(output.logits[0, -1]).item())
                tokens.append(next_token)
    finally:
        handle.remove()
    if len(captured) != int(positions):
        raise RuntimeError(f"captured {len(captured)} states, expected {positions}")
    return captured, tokens


def project_target_metadata_bytes(config: dict[str, Any]) -> int:
    # Llama-3.1-405B registered shape: H=16384, I=53248, 126 layers.
    target_hidden = 16_384
    target_intermediate = 53_248
    target_layers = 126
    pages = target_intermediate // int(config["page_channels"])
    return pages * target_hidden * 8 * target_layers + pages * 16 * target_layers


def write_checksums(output_dir: Path, relative_paths: list[str]) -> None:
    lines = []
    for relative in sorted(relative_paths):
        path = output_dir / relative
        lines.append(f"{sha256_file(path)}  {relative}")
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = verify_runtime()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model, tokenizer, load_ns = load_model(config)
    layer = model.model.layers[int(config["layer_index"])]
    compile_start = time.perf_counter_ns()
    compiler = compile_joint_exact_swiglu(
        layer.mlp, page_channels=int(config["page_channels"])
    )
    compile_ns = time.perf_counter_ns() - compile_start

    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for prompt_spec in config["prompts"]:
        states, tokens = capture_states(
            model,
            tokenizer,
            prompt_spec["prompt"],
            layer_index=int(config["layer_index"]),
            positions=int(config["decode_positions_per_prompt"]),
        )
        trace_rows.append(
            {
                "prompt_id": prompt_spec["id"],
                "family": prompt_spec["family"],
                "prompt_sha256": sha256_bytes(prompt_spec["prompt"].encode()),
                "tokens": tokens,
                "state_count": len(states),
            }
        )
        for position, activation in enumerate(states):
            for result in (
                compiler.query_oracle_greedy(activation),
                compiler.query_sound_metadata(activation),
            ):
                row = result.to_dict()
                row.update(
                    {
                        "prompt_id": prompt_spec["id"],
                        "family": prompt_spec["family"],
                        "decode_position": position,
                        "activation_sha256": sha256_bytes(
                            activation.view(torch.uint8).numpy().tobytes()
                        ),
                        "activation_l2": float(
                            torch.linalg.vector_norm(activation.float()).item()
                        ),
                    }
                )
                rows.append(row)

    oracle = aggregate_query_rows(rows, mode="oracle_exact_contribution_greedy")
    sound = aggregate_query_rows(rows, mode="sound_metadata_bound")
    projected_metadata = project_target_metadata_bytes(config)
    manifest = compiler.manifest()

    integrity_passed = all(
        aggregate["false_accepts"] == 0
        and aggregate["candidate_mismatches"] == 0
        and aggregate["gate_up_row_split_mismatches"] == 0
        and aggregate["bound_violations"] == 0
        for aggregate in (oracle, sound)
    )
    oracle_passed = (
        oracle["fallback_rate"] == 0.0
        and oracle["whole_model_fraction_p50"] <= P50_WHOLE_MODEL_FRACTION
        and oracle["whole_model_fraction_p95"] <= P95_WHOLE_MODEL_FRACTION
        and oracle["whole_model_operation_fraction_p50"] <= P50_WHOLE_MODEL_FRACTION
        and oracle["whole_model_operation_fraction_p95"] <= P95_WHOLE_MODEL_FRACTION
    )
    sound_passed = (
        sound["fallback_rate"] == 0.0
        and sound["whole_model_fraction_p50"] <= P50_WHOLE_MODEL_FRACTION
        and sound["whole_model_fraction_p95"] <= P95_WHOLE_MODEL_FRACTION
        and sound["whole_model_operation_fraction_p50"] <= P50_WHOLE_MODEL_FRACTION
        and sound["whole_model_operation_fraction_p95"] <= P95_WHOLE_MODEL_FRACTION
    )
    metadata_passed = projected_metadata <= int(config["target_hot_metadata_limit_bytes"])

    if not integrity_passed:
        verdict = "INVALID_JOINT_EXACT_SWIGLU_COMPILER_CONTROL_FAILURE"
    elif oracle_passed and sound_passed and metadata_passed:
        verdict = "PROMOTE_JOINT_EXACT_SWIGLU_COMPILER_TO_COMPLETE_LAYER_GATE"
    else:
        verdict = "REJECT_PAGE_SEPARABLE_JOINT_EXACT_SWIGLU_REFINEMENT_AS_CORE"

    deterministic_core = {
        "schema": config["schema"],
        "config_sha256": sha256_file(config_path),
        "checkpoint": {
            "model_id": config["model_id"],
            "revision": config["revision"],
        },
        "compiler_manifest": manifest,
        "trace_rows": trace_rows,
        "query_rows": rows,
        "aggregates": {"oracle": oracle, "sound": sound},
        "projected_target_metadata_bytes": projected_metadata,
        "gates": {
            "integrity_passed": integrity_passed,
            "oracle_passed": oracle_passed,
            "sound_passed": sound_passed,
            "metadata_passed": metadata_passed,
        },
        "verdict": verdict,
    }
    core_hash = sha256_bytes(canonical_bytes(deterministic_core))
    result = {
        "experiment": "EXP-085A",
        "name": "joint_exact_swiglu_compiler_gate",
        "evidence_level": "E1",
        "phase": ["A-structure", "B-reference", "C-small-real-checkpoint-observation"],
        "source_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNVERIFIED"),
        "runtime": runtime,
        "load_wall_ns": load_ns,
        "compile_wall_ns": compile_ns,
        "config_sha256": sha256_file(config_path),
        "compiler_manifest": manifest,
        "measured_case_count": len(rows),
        "measured_activation_count": len(rows) // 2,
        "aggregates": {"oracle": oracle, "sound": sound},
        "projected": {
            "target_mlp_parameter_share": LLAMA405_MLP_PARAMETER_SHARE,
            "target_hot_metadata_bytes": projected_metadata,
            "target_hot_metadata_gib": projected_metadata / (1024**3),
            "p50_whole_model_fraction": P50_WHOLE_MODEL_FRACTION,
            "p95_whole_model_fraction": P95_WHOLE_MODEL_FRACTION,
        },
        "gates": deterministic_core["gates"],
        "authoritative_decision": verdict,
        "deterministic_core_sha256": core_hash,
        "claim_boundary": {
            "complete_real_mlp_replaced": False,
            "official_checkpoint_loaded": True,
            "actual_checkpoint_activations": True,
            "oracle_selector_deployable": False,
            "sound_reference_deployable": True,
            "exactness": "fail-closed; unresolved rows use unchanged native MLP",
            "405b_execution": "NOT_TESTED",
            "8gib_gpu": "NOT_TESTED",
            "physical_latency": "NOT_TESTED",
            "complete_layer": "NOT_TESTED",
        },
        "unverified": [
            "CUDA lowering and physical page I/O",
            "complete Transformer-layer replacement",
            "128-step candidate successor-state equality",
            "405B model execution",
            "8 GiB peak VRAM",
            "same-machine 4B p50/p95 target",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    artifacts_dir = output_dir / "artifacts"
    raw_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (raw_dir / "query_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    (raw_dir / "trace_rows.json").write_text(
        json.dumps(trace_rows, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (artifacts_dir / "compiler_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artifacts_dir / "deterministic_core.json").write_bytes(canonical_bytes(deterministic_core))
    write_checksums(
        output_dir,
        [
            "result.json",
            "raw/query_rows.jsonl",
            "raw/trace_rows.json",
            "artifacts/compiler_manifest.json",
            "artifacts/deterministic_core.json",
        ],
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args.config, args.output_dir)
    except Exception as exc:  # fail closed while preserving an infrastructure artifact
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "experiment": "EXP-085A",
            "source_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNVERIFIED"),
            "authoritative_decision": "INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION",
            "error": f"{type(exc).__name__}: {exc}",
            "claim_boundary": {
                "mechanism_scientifically_rejected": False,
                "official_checkpoint_result": "NOT_COMPLETED",
            },
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output_dir / "checksums.sha256").write_text(
            f"{sha256_file(args.output_dir / 'result.json')}  result.json\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
