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

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.exact_state_axis_lifting import (  # noqa: E402
    LLAMA405_MLP_PARAMETER_SHARE,
    P50_WHOLE_MODEL_FRACTION,
    P95_WHOLE_MODEL_FRACTION,
    analyze_mlp_block,
    canonical_sha256,
)
from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    PINNED_SAFETENSORS,
    PINNED_TORCH,
    PINNED_TRANSFORMERS,
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    mismatch = {
        key: [expected[key], actual[key]]
        for key in expected
        if expected[key] != actual[key]
    }
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


def capture_exact_mlp_streams(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    layer_index: int,
    positions: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], int]:
    mlp_inputs: list[torch.Tensor] = []
    down_inputs: list[torch.Tensor] = []

    def mlp_hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        hidden = inputs[0]
        mlp_inputs.append(hidden[:, -1, :].detach().contiguous().cpu())

    def down_hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        hidden = inputs[0]
        down_inputs.append(hidden[:, -1, :].detach().contiguous().cpu())

    layer = model.model.layers[int(layer_index)]
    handles = [
        layer.mlp.register_forward_pre_hook(mlp_hook),
        layer.mlp.down_proj.register_forward_pre_hook(down_hook),
    ]
    tokens: list[int] = []
    target_calls = 0
    try:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            output = model(input_ids=input_ids, use_cache=True, return_dict=True)
            target_calls += 1
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
                target_calls += 1
                past = output.past_key_values
                next_token = int(torch.argmax(output.logits[0, -1]).item())
                tokens.append(next_token)
    finally:
        for handle in handles:
            handle.remove()

    if len(mlp_inputs) != int(positions) or len(down_inputs) != int(positions):
        raise RuntimeError(
            f"captured mlp/down {len(mlp_inputs)}/{len(down_inputs)}, expected {positions}"
        )
    x = torch.cat(mlp_inputs, dim=0)
    z = torch.cat(down_inputs, dim=0)
    if x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
        raise RuntimeError(f"unexpected captured dtype {x.dtype}/{z.dtype}")
    return x, z, tokens, target_calls


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise RuntimeError("cannot aggregate empty population")
    return float(
        np.percentile(np.asarray(values, dtype=np.float64), q, method="linear")
    )


def aggregate(rows: list[dict[str, Any]], *, mode: str, block_size: int) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["mode"] == mode and int(row["block_size"]) == int(block_size)
    ]
    if not selected:
        raise RuntimeError(f"empty aggregate for {mode}/{block_size}")
    keys = (
        "projected_whole_model_operation_fraction",
        "projected_whole_model_weight_fraction",
        "projected_joint_fraction",
        "input_residual_nonzero_fraction",
        "down_residual_nonzero_fraction",
        "input_atom_union_operation_fraction",
        "down_atom_union_operation_fraction",
    )
    result: dict[str, Any] = {
        "mode": mode,
        "block_size": int(block_size),
        "count": len(selected),
        "reconstruction_mismatches": sum(
            int(row["reconstruction_mismatches"]) for row in selected
        ),
    }
    for key in keys:
        values = [float(row[key]) for row in selected]
        result[f"{key}_p05"] = percentile(values, 5)
        result[f"{key}_p50"] = percentile(values, 50)
        result[f"{key}_p95"] = percentile(values, 95)
        result[f"{key}_max"] = max(values)
    families: dict[str, Any] = {}
    for family in sorted({str(row["family"]) for row in selected}):
        family_rows = [row for row in selected if row["family"] == family]
        values = [float(row["projected_joint_fraction"]) for row in family_rows]
        families[family] = {
            "count": len(family_rows),
            "joint_p50": percentile(values, 50),
            "joint_p95": percentile(values, 95),
            "joint_max": max(values),
        }
    result["families"] = families
    return result


def write_checksums(output_dir: Path, relative_paths: list[str]) -> None:
    lines = []
    for relative in sorted(relative_paths):
        path = output_dir / relative
        lines.append(f"{sha256_file(path)}  {relative}")
    (output_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = verify_runtime()
    if config["p50_whole_model_fraction"] != P50_WHOLE_MODEL_FRACTION:
        raise RuntimeError("p50 target mismatch")
    if config["p95_whole_model_fraction"] != P95_WHOLE_MODEL_FRACTION:
        raise RuntimeError("p95 target mismatch")
    if max(map(int, config["block_sizes"])) != int(
        config["decode_positions_per_prompt"]
    ):
        raise RuntimeError("largest block must equal frozen decode length")

    torch.manual_seed(0)
    torch.set_num_threads(2)
    model, tokenizer, load_ns = load_model(config)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    target_calls = 0
    analysis_start = time.perf_counter_ns()

    for prompt_spec in config["prompts"]:
        x, z, tokens, calls = capture_exact_mlp_streams(
            model,
            tokenizer,
            prompt_spec["prompt"],
            layer_index=int(config["layer_index"]),
            positions=int(config["decode_positions_per_prompt"]),
        )
        target_calls += calls
        traces.append(
            {
                "prompt_id": prompt_spec["id"],
                "family": prompt_spec["family"],
                "prompt_sha256": sha256_bytes(prompt_spec["prompt"].encode()),
                "tokens": tokens,
                "mlp_input_sha256": sha256_bytes(
                    x.view(torch.uint8).numpy().tobytes()
                ),
                "down_input_sha256": sha256_bytes(
                    z.view(torch.uint8).numpy().tobytes()
                ),
                "state_count": int(x.shape[0]),
                "mlp_width": int(x.shape[1]),
                "down_width": int(z.shape[1]),
            }
        )
        for block_size in map(int, config["block_sizes"]):
            if x.shape[0] % block_size != 0:
                raise RuntimeError(
                    f"state count {x.shape[0]} not divisible by block {block_size}"
                )
            for block_start in range(0, int(x.shape[0]), block_size):
                block_x = x[block_start : block_start + block_size]
                block_z = z[block_start : block_start + block_size]
                for mode in config["predictor_modes"]:
                    analysis = analyze_mlp_block(block_x, block_z, mode=mode)
                    row = analysis.to_dict()
                    row.update(
                        {
                            "prompt_id": prompt_spec["id"],
                            "family": prompt_spec["family"],
                            "block_start": block_start,
                            "block_stop": block_start + block_size,
                            "future_target_tokens_used": True,
                            "selector_uses_current_exact_activation": mode
                            == "coordinate",
                            "coordinate_mixed_parent_application_granted_free": mode
                            == "coordinate",
                            "input_residual_nonzero_fraction": analysis.input.residual_nonzero_fraction,
                            "down_residual_nonzero_fraction": analysis.down_input.residual_nonzero_fraction,
                            "input_atom_union_operation_fraction": analysis.input.atom_union_operation_fraction,
                            "down_atom_union_operation_fraction": analysis.down_input.atom_union_operation_fraction,
                            "reconstruction_mismatches": (
                                analysis.input.reconstruction_mismatches
                                + analysis.down_input.reconstruction_mismatches
                            ),
                        }
                    )
                    rows.append(row)

    analysis_ns = time.perf_counter_ns() - analysis_start
    aggregates = {
        f"{mode}_k{block_size}": aggregate(
            rows, mode=mode, block_size=int(block_size)
        )
        for mode in config["predictor_modes"]
        for block_size in config["block_sizes"]
    }
    decisive = aggregates[
        f"coordinate_k{max(map(int, config['block_sizes']))}"
    ]
    integrity_passed = (
        sum(int(row["reconstruction_mismatches"]) for row in rows) == 0
        and len(traces) == len(config["prompts"])
        and target_calls
        == len(config["prompts"])
        * int(config["decode_positions_per_prompt"])
    )
    oracle_passed = (
        decisive["projected_whole_model_operation_fraction_p50"]
        <= P50_WHOLE_MODEL_FRACTION
        and decisive["projected_whole_model_operation_fraction_p95"]
        <= P95_WHOLE_MODEL_FRACTION
        and decisive["projected_whole_model_weight_fraction_p50"]
        <= P50_WHOLE_MODEL_FRACTION
        and decisive["projected_whole_model_weight_fraction_p95"]
        <= P95_WHOLE_MODEL_FRACTION
        and decisive["projected_joint_fraction_p50"]
        <= P50_WHOLE_MODEL_FRACTION
        and decisive["projected_joint_fraction_p95"]
        <= P95_WHOLE_MODEL_FRACTION
        and all(
            family["joint_p95"] <= P95_WHOLE_MODEL_FRACTION
            for family in decisive["families"].values()
        )
    )

    if not integrity_passed:
        verdict = "INVALID_EXACT_STATE_AXIS_LIFTING_CONTROL_FAILURE"
    elif oracle_passed:
        verdict = "PROMOTE_EXACT_STATE_AXIS_LIFTING_TO_CAUSAL_VECTOR_PROGRAM_GATE"
    else:
        verdict = "REJECT_EXACT_STATE_AXIS_LIFTING_MICROPROGRAM_AS_CORE"

    deterministic_core = {
        "schema": config["schema"],
        "config_sha256": sha256_file(config_path),
        "checkpoint": {
            "model_id": config["model_id"],
            "revision": config["revision"],
        },
        "trace_rows": traces,
        "query_rows": rows,
        "aggregates": aggregates,
        "decisive_aggregate": decisive,
        "gates": {
            "integrity_passed": integrity_passed,
            "oracle_passed": oracle_passed,
        },
        "verdict": verdict,
    }
    result = {
        "experiment": "EXP-086A",
        "name": "exact_state_axis_lifting_microprogram_gate",
        "evidence_level": "E1",
        "phase": [
            "A-structure",
            "B-reference",
            "C-small-real-checkpoint-observation",
        ],
        "source_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNVERIFIED"),
        "runtime": runtime,
        "load_wall_ns": load_ns,
        "analysis_wall_ns": analysis_ns,
        "config_sha256": sha256_file(config_path),
        "target_forward_calls": target_calls,
        "measured_prompt_count": len(traces),
        "measured_block_rows": len(rows),
        "aggregates": aggregates,
        "decisive_aggregate": decisive,
        "projected": {
            "target_mlp_parameter_share": LLAMA405_MLP_PARAMETER_SHARE,
            "p50_whole_model_fraction": P50_WHOLE_MODEL_FRACTION,
            "p95_whole_model_fraction": P95_WHOLE_MODEL_FRACTION,
            "all_non_mlp_work_granted_free": True,
            "selector_metadata_granted_free": True,
            "coordinate_mixed_parent_application_granted_free": True,
            "signed_atom_broadcast_and_reconstruction_granted_free": True,
        },
        "gates": deterministic_core["gates"],
        "authoritative_decision": verdict,
        "deterministic_core_sha256": canonical_sha256(deterministic_core),
        "claim_boundary": {
            "official_checkpoint_loaded": True,
            "actual_checkpoint_activations": True,
            "future_target_tokens_used": True,
            "oracle_deployable": False,
            "native_fp32_reduction_order_preserved": False,
            "complete_real_mlp_replaced": False,
            "complete_layer": "NOT_TESTED",
            "405b_execution": "NOT_TESTED",
            "8gib_gpu": "NOT_TESTED",
            "physical_latency": "NOT_TESTED",
        },
        "unverified": [
            "native-order BF16/FP32 packed lifting semantics",
            "causal drafter producing the exact activation block",
            "candidate-tree acceptance and rollback",
            "complete Transformer-layer successor-state equality",
            "CUDA lowering and physical bandwidth",
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
    with (raw_dir / "block_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            )
    (raw_dir / "trace_rows.json").write_text(
        json.dumps(traces, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (artifacts_dir / "deterministic_core.json").write_bytes(
        canonical_bytes(deterministic_core)
    )
    write_checksums(
        output_dir,
        [
            "result.json",
            "raw/block_rows.jsonl",
            "raw/trace_rows.json",
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
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "experiment": "EXP-086A",
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
            f"{sha256_file(args.output_dir / 'result.json')}  result.json\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
