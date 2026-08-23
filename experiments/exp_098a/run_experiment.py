from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import transformers
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_094a.run_experiment import checkpoint_identity, load_model, verify_runtime, write_json  # noqa: E402
from vortex_runtime.causal_suffix_action_cache import quantize_rowwise_symmetric_int4  # noqa: E402


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "exp-098a-q4-rounding-cell-oracle-v1":
        raise RuntimeError("unexpected config schema")
    if not bool(config.get("frozen_before_results")):
        raise RuntimeError("configuration was not frozen")
    if [int(x) for x in config["layer_indices"]] != [0, 15, 29]:
        raise RuntimeError("layer population changed")
    expected_roles = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if list(config["projection_roles"]) != expected_roles:
        raise RuntimeError("projection population changed")
    if int(config["decode_positions_per_prompt"]) != 4:
        raise RuntimeError("decode population changed")
    if int(config["q4"]["minimum"]) != -8 or int(config["q4"]["maximum"]) != 7:
        raise RuntimeError("Q4 range changed")
    if len(config["prompts"]) != 6:
        raise RuntimeError("prompt population changed")
    if {str(p["split"]) for p in config["prompts"]} != {"build", "holdout"}:
        raise RuntimeError("build/holdout split missing")


def projection_modules(layer: Any) -> dict[str, Any]:
    return {
        "q_proj": layer.self_attn.q_proj,
        "k_proj": layer.self_attn.k_proj,
        "v_proj": layer.self_attn.v_proj,
        "o_proj": layer.self_attn.o_proj,
        "gate_proj": layer.mlp.gate_proj,
        "up_proj": layer.mlp.up_proj,
        "down_proj": layer.mlp.down_proj,
    }


def int16_words(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().to(torch.bfloat16).view(torch.int16)


def exact_word_match(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lhs = int16_words(left)
    rhs = int16_words(right)
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"word-shape mismatch {tuple(lhs.shape)} != {tuple(rhs.shape)}")
    return lhs == rhs


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise RuntimeError("empty percentile population")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    runtime = verify_runtime()
    torch.manual_seed(0)
    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.set_num_interop_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_load_ns = load_model(config)
    identity = checkpoint_identity(model, tokenizer, config)

    modules: dict[tuple[int, str], Any] = {}
    qweights: dict[tuple[int, str], torch.Tensor] = {}
    manifests: dict[str, Any] = {}
    for layer_index in config["layer_indices"]:
        layer = model.model.layers[int(layer_index)]
        by_role = projection_modules(layer)
        for role in config["projection_roles"]:
            module = by_role[str(role)]
            key = (int(layer_index), str(role))
            modules[key] = module
            q4 = quantize_rowwise_symmetric_int4(
                module.weight,
                minimum=int(config["q4"]["minimum"]),
                maximum=int(config["q4"]["maximum"]),
            )
            qweights[key] = q4.dequantized_weight_bf16
            manifests[f"L{layer_index}:{role}"] = q4.manifest.to_dict()

    rows: list[dict[str, Any]] = []
    context: dict[str, Any] = {}
    handles = []

    def make_hook(layer_index: int, role: str, module: Any):
        key = (layer_index, role)

        def hook(_module: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if "prompt_id" not in context:
                raise RuntimeError("hook fired without prompt context")
            x = inputs[0]
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("linear projection output is not a tensor")
            bias = module.bias
            exact_replay = F.linear(x, module.weight, bias)
            if not bool(exact_word_match(exact_replay, output).all().item()):
                rows.append(
                    {
                        "prompt_id": context["prompt_id"],
                        "split": context["split"],
                        "family": context["family"],
                        "call_index": int(context["call_index"]),
                        "layer_index": layer_index,
                        "role": role,
                        "exact_replay_mismatch": True,
                    }
                )
                return

            predictor = F.linear(x.to(torch.bfloat16), qweights[key], bias)
            exact_last = output[:, -1, :].detach().contiguous().cpu().to(torch.bfloat16)
            predictor_last = predictor[:, -1, :].detach().contiguous().cpu().to(torch.bfloat16)
            match = exact_word_match(exact_last, predictor_last).reshape(-1)
            output_rows = int(match.numel())
            mismatched_rows = int((~match).sum().item())
            input_width = int(module.in_features)
            total_bf16_macs = output_rows * input_width
            repair_bf16_macs = mismatched_rows * input_width
            rows.append(
                {
                    "prompt_id": context["prompt_id"],
                    "split": context["split"],
                    "family": context["family"],
                    "call_index": int(context["call_index"]),
                    "layer_index": layer_index,
                    "role": role,
                    "input_width": input_width,
                    "output_rows": output_rows,
                    "matched_rows": output_rows - mismatched_rows,
                    "mismatched_rows": mismatched_rows,
                    "oracle_lock_fraction": (output_rows - mismatched_rows) / output_rows,
                    "oracle_repair_fraction": mismatched_rows / output_rows,
                    "total_bf16_macs": total_bf16_macs,
                    "repair_bf16_macs": repair_bf16_macs,
                    "exact_replay_mismatch": False,
                    "exact_output_sha256": tensor_sha256(exact_last),
                    "q4_output_sha256": tensor_sha256(predictor_last),
                }
            )

        return hook

    for (layer_index, role), module in modules.items():
        handles.append(module.register_forward_hook(make_hook(layer_index, role, module)))

    traces: list[dict[str, Any]] = []
    try:
        for prompt in config["prompts"]:
            context.update(
                prompt_id=str(prompt["id"]),
                split=str(prompt["split"]),
                family=str(prompt["family"]),
                call_index=0,
            )
            input_ids = tokenizer(str(prompt["prompt"]), return_tensors="pt")["input_ids"]
            tokens: list[int] = []
            with torch.inference_mode():
                output = model(input_ids=input_ids, use_cache=True, return_dict=True)
                past = output.past_key_values
                next_token = int(torch.argmax(output.logits[0, -1]).item())
                tokens.append(next_token)
                for call_index in range(1, int(config["decode_positions_per_prompt"])):
                    context["call_index"] = call_index
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
            traces.append(
                {
                    "prompt_id": str(prompt["id"]),
                    "split": str(prompt["split"]),
                    "family": str(prompt["family"]),
                    "prompt_sha256": hashlib.sha256(str(prompt["prompt"]).encode()).hexdigest(),
                    "tokens": tokens,
                }
            )
    finally:
        for handle in handles:
            handle.remove()

    replay_mismatches = sum(bool(row.get("exact_replay_mismatch")) for row in rows)
    valid_rows = [row for row in rows if not row.get("exact_replay_mismatch")]
    expected_rows = len(config["prompts"]) * int(config["decode_positions_per_prompt"]) * len(config["layer_indices"]) * len(config["projection_roles"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"captured {len(rows)} projection rows, expected {expected_rows}")

    grouped: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in valid_rows:
        key = (str(row["prompt_id"]), int(row["call_index"]), int(row["layer_index"]))
        agg = grouped.setdefault(
            key,
            {
                "prompt_id": row["prompt_id"],
                "split": row["split"],
                "family": row["family"],
                "call_index": row["call_index"],
                "layer_index": row["layer_index"],
                "repair_bf16_macs": 0,
                "total_bf16_macs": 0,
            },
        )
        agg["repair_bf16_macs"] += int(row["repair_bf16_macs"])
        agg["total_bf16_macs"] += int(row["total_bf16_macs"])
    layer_rows = []
    for agg in grouped.values():
        agg = dict(agg)
        agg["oracle_bf16_repair_fraction"] = agg["repair_bf16_macs"] / agg["total_bf16_macs"]
        layer_rows.append(agg)

    def summarize(split: str) -> dict[str, Any]:
        subset = [row for row in layer_rows if row["split"] == split]
        fractions = [float(row["oracle_bf16_repair_fraction"]) for row in subset]
        role_stats: dict[str, Any] = {}
        for role in config["projection_roles"]:
            role_rows = [row for row in valid_rows if row["split"] == split and row["role"] == role]
            rf = [float(row["oracle_repair_fraction"]) for row in role_rows]
            role_stats[str(role)] = {
                "count": len(rf),
                "repair_fraction_p50": percentile(rf, 50),
                "repair_fraction_p95": percentile(rf, 95),
                "repair_fraction_min": min(rf),
                "repair_fraction_max": max(rf),
            }
        return {
            "layer_state_count": len(subset),
            "oracle_bf16_repair_fraction_p50": percentile(fractions, 50),
            "oracle_bf16_repair_fraction_p95": percentile(fractions, 95),
            "oracle_bf16_repair_fraction_min": min(fractions),
            "oracle_bf16_repair_fraction_max": max(fractions),
            "role_stats": role_stats,
        }

    build = summarize("build")
    holdout = summarize("holdout")
    max_role_p95 = max(float(v["repair_fraction_p95"]) for v in holdout["role_stats"].values())
    gate = config["first_gate"]
    repair_gate = (
        float(holdout["oracle_bf16_repair_fraction_p50"]) <= float(gate["maximum_bf16_repair_fraction_p50"])
        and float(holdout["oracle_bf16_repair_fraction_p95"]) <= float(gate["maximum_bf16_repair_fraction_p95"])
        and max_role_p95 <= float(gate["maximum_role_repair_fraction_p95"])
    )

    integrity_failures: list[str] = []
    if replay_mismatches:
        integrity_failures.append(f"exact_replay_mismatches:{replay_mismatches}")
    if identity["resolved_revision"] not in ("", config["revision"]):
        integrity_failures.append("resolved_checkpoint_revision")

    if integrity_failures:
        decision = "INVALID_Q4_ROUNDING_CELL_ORACLE_CONTROL_FAILURE"
    elif repair_gate:
        decision = "SURVIVES_Q4_ORACLE_ROW_LOCK_REQUIRES_SOUND_ROUNDING_CELL_GATE"
    else:
        decision = "REJECT_Q4_ROUNDING_CELL_ROW_REPAIR_AS_10X_CORE"

    selected_parameter_count = sum(int(m.weight.numel()) for m in modules.values())
    selected_q4_bytes = sum(int(v["artifact_bytes"]) for v in manifests.values())
    deterministic_core = {
        "schema": config["schema"],
        "config_sha256": canonical_sha256(config),
        "checkpoint": identity,
        "trace_rows": traces,
        "selected_projection_parameter_count": selected_parameter_count,
        "selected_q4_artifact_bytes": selected_q4_bytes,
        "q4_manifests": manifests,
        "build": build,
        "holdout": holdout,
        "maximum_holdout_role_repair_fraction_p95": max_role_p95,
        "integrity_failures": integrity_failures,
        "authoritative_decision": decision,
    }
    result = {
        **deterministic_core,
        "deterministic_core_sha256": canonical_sha256(deterministic_core),
        "projection_rows": rows,
        "layer_state_rows": sorted(layer_rows, key=lambda r: (r["prompt_id"], r["call_index"], r["layer_index"])),
        "runtime": {
            **runtime,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "timing": {"model_load_ns": model_load_ns},
        "claim_boundary": {
            "perfect_row_match_oracle": True,
            "sound_rounding_cell_certificate": "NOT_CONSTRUCTED",
            "q4_predictor_physical_runtime": "NOT_TESTED",
            "target_405b": "NOT_TESTED",
            "physical_8gib": "NOT_TESTED",
            "same_machine_4b_latency": "NOT_TESTED",
        },
    }
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "raw/projection_rows.json", rows)
    write_json(output_dir / "raw/layer_state_rows.json", result["layer_state_rows"])
    write_json(output_dir / "artifacts/deterministic_core.json", deterministic_core)
    print(
        json.dumps(
            {
                "authoritative_decision": decision,
                "holdout_repair_p50": holdout["oracle_bf16_repair_fraction_p50"],
                "holdout_repair_p95": holdout["oracle_bf16_repair_fraction_p95"],
                "maximum_holdout_role_repair_p95": max_role_p95,
                "integrity_failures": integrity_failures,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
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
