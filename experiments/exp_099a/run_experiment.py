from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
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


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "exp-099a-dyadic-reassociation-lock-v1":
        raise RuntimeError("unexpected config schema")
    if not bool(config.get("frozen_before_results")):
        raise RuntimeError("configuration was not frozen")
    if [int(x) for x in config["layer_indices"]] != [0, 15, 29]:
        raise RuntimeError("layer population changed")
    roles = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if list(config["projection_roles"]) != roles:
        raise RuntimeError("projection population changed")
    if int(config["decode_positions_per_prompt"]) != 4:
        raise RuntimeError("decode population changed")
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


def exact_word_match(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lhs = left.detach().contiguous().to(torch.bfloat16).view(torch.int16)
    rhs = right.detach().contiguous().to(torch.bfloat16).view(torch.int16)
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"word-shape mismatch {tuple(lhs.shape)} != {tuple(rhs.shape)}")
    return lhs == rhs


def gamma(k: int, unit_roundoff: float) -> float:
    ku = float(k) * float(unit_roundoff)
    if not (0.0 <= ku < 1.0):
        raise RuntimeError("invalid gamma regime")
    return ku / (1.0 - ku)


def bf16_cell(candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cand = candidate.detach().contiguous().to(torch.bfloat16)
    pos_inf = torch.full_like(cand, float("inf"))
    neg_inf = torch.full_like(cand, float("-inf"))
    upper_neighbor = torch.nextafter(cand, pos_inf).to(torch.float64)
    lower_neighbor = torch.nextafter(cand, neg_inf).to(torch.float64)
    center = cand.to(torch.float64)
    return (center + lower_neighbor) * 0.5, (center + upper_neighbor) * 0.5


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
    for layer_index in config["layer_indices"]:
        by_role = projection_modules(model.model.layers[int(layer_index)])
        for role in config["projection_roles"]:
            modules[(int(layer_index), str(role))] = by_role[str(role)]

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    context: dict[str, Any] = {}
    handles = []

    def make_hook(layer_index: int, role: str, module: Any):
        def hook(_module: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if "prompt_id" not in context:
                raise RuntimeError("hook fired without prompt context")
            x_last = inputs[0][:, -1, :].detach().contiguous()
            official_last = output[:, -1, :].detach().contiguous()
            native_replay = F.linear(x_last, module.weight, module.bias)
            replay_ok = bool(exact_word_match(native_replay, official_last).all().item())
            if not replay_ok:
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

            x64 = x_last.to(torch.float64)
            w64 = module.weight.detach().to(torch.float64)
            b64 = None if module.bias is None else module.bias.detach().to(torch.float64)
            exact_real_approx = F.linear(x64, w64, b64)
            sum_abs = torch.matmul(x64.abs(), w64.abs().transpose(0, 1))
            if b64 is not None:
                sum_abs = sum_abs + b64.abs()

            n = int(module.in_features)
            # BF16 operands have <=8 significand bits; their product is exactly representable
            # in FP64. This gamma term conservatively encloses only the FP64 reductions.
            g64 = gamma(n + 2, 2.0 ** -53)
            fp64_error = sum_abs * g64
            exact_candidate = exact_real_approx.to(torch.bfloat16)
            cell_low, cell_high = bf16_cell(exact_candidate)
            exact_low = exact_real_approx - fp64_error
            exact_high = exact_real_approx + fp64_error
            exact_real_cell_certified = torch.isfinite(exact_real_approx) & (exact_low > cell_low) & (exact_high < cell_high)

            official_match = exact_word_match(exact_candidate, official_last)
            lock = exact_real_cell_certified & official_match

            # Secondary diagnostic: a deliberately generic FP32 dot-product envelope.
            # It is not used for promotion because the target CUDA/BLAS reduction tree is
            # not fixed by this CPU Gate. If it fits, however, it is strong evidence that
            # even reassociation cannot change the BF16 word.
            g32 = gamma(2 * n + 2, 2.0 ** -24)
            underflow_floor = float(n + 1) * (2.0 ** -149)
            generic_native_error = sum_abs * g32 + underflow_floor
            generic_low = exact_low - generic_native_error
            generic_high = exact_high + generic_native_error
            generic_fp32_cell_certified = (generic_low > cell_low) & (generic_high < cell_high)
            generic_fp32_validated = generic_fp32_cell_certified & official_match

            lock_flat = lock.reshape(-1)
            exact_cert_flat = exact_real_cell_certified.reshape(-1)
            generic_flat = generic_fp32_validated.reshape(-1)
            output_rows = int(lock_flat.numel())
            repaired_rows = int((~lock_flat).sum().item())
            input_width = int(module.in_features)
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
                    "exact_real_cell_certified_rows": int(exact_cert_flat.sum().item()),
                    "official_exact_real_match_rows": int(official_match.reshape(-1).sum().item()),
                    "locked_rows": output_rows - repaired_rows,
                    "repaired_rows": repaired_rows,
                    "native_order_repair_fraction": repaired_rows / output_rows,
                    "generic_fp32_certified_rows": int(generic_flat.sum().item()),
                    "generic_fp32_certified_fraction": float(generic_flat.to(torch.float64).mean().item()),
                    "repair_original_macs": repaired_rows * input_width,
                    "full_original_macs": output_rows * input_width,
                    "exact_replay_mismatch": False,
                    "official_output_sha256": tensor_sha256(official_last.to(torch.bfloat16)),
                    "exact_real_candidate_sha256": tensor_sha256(exact_candidate),
                }
            )

        return hook

    for (layer_index, role), module in modules.items():
        handles.append(module.register_forward_hook(make_hook(layer_index, role, module)))

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
                    output = model(input_ids=token_tensor, past_key_values=past, use_cache=True, return_dict=True)
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

    expected_rows = len(config["prompts"]) * int(config["decode_positions_per_prompt"]) * len(config["layer_indices"]) * len(config["projection_roles"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"captured {len(rows)} rows, expected {expected_rows}")
    replay_mismatches = sum(bool(row.get("exact_replay_mismatch")) for row in rows)
    valid_rows = [row for row in rows if not row.get("exact_replay_mismatch")]

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
                "repair_original_macs": 0,
                "full_original_macs": 0,
                "generic_fp32_certified_rows": 0,
                "total_rows": 0,
            },
        )
        agg["repair_original_macs"] += int(row["repair_original_macs"])
        agg["full_original_macs"] += int(row["full_original_macs"])
        agg["generic_fp32_certified_rows"] += int(row["generic_fp32_certified_rows"])
        agg["total_rows"] += int(row["output_rows"])

    layer_rows: list[dict[str, Any]] = []
    for agg in grouped.values():
        row = dict(agg)
        row["native_order_repair_fraction"] = row["repair_original_macs"] / row["full_original_macs"]
        row["generic_fp32_certified_fraction"] = row["generic_fp32_certified_rows"] / row["total_rows"]
        layer_rows.append(row)

    def summarize(split: str) -> dict[str, Any]:
        subset = [row for row in layer_rows if row["split"] == split]
        repair = [float(row["native_order_repair_fraction"]) for row in subset]
        generic = [float(row["generic_fp32_certified_fraction"]) for row in subset]
        role_stats: dict[str, Any] = {}
        for role in config["projection_roles"]:
            rr = [row for row in valid_rows if row["split"] == split and row["role"] == role]
            rf = [float(row["native_order_repair_fraction"]) for row in rr]
            exact_cert = [float(row["exact_real_cell_certified_rows"]) / float(row["output_rows"]) for row in rr]
            role_stats[str(role)] = {
                "count": len(rr),
                "repair_fraction_p50": percentile(rf, 50),
                "repair_fraction_p95": percentile(rf, 95),
                "exact_real_cell_certificate_p50": percentile(exact_cert, 50),
            }
        return {
            "layer_state_count": len(subset),
            "native_order_repair_fraction_p50": percentile(repair, 50),
            "native_order_repair_fraction_p95": percentile(repair, 95),
            "native_order_repair_fraction_min": min(repair),
            "native_order_repair_fraction_max": max(repair),
            "generic_fp32_certificate_fraction_p50": percentile(generic, 50),
            "generic_fp32_certificate_fraction_p95": percentile(generic, 95),
            "role_stats": role_stats,
        }

    build = summarize("build")
    holdout = summarize("holdout")
    max_role_p95 = max(float(v["repair_fraction_p95"]) for v in holdout["role_stats"].values())
    gate = config["first_gate"]
    passes = (
        float(holdout["native_order_repair_fraction_p50"]) <= float(gate["maximum_native_order_repair_fraction_p50"])
        and float(holdout["native_order_repair_fraction_p95"]) <= float(gate["maximum_native_order_repair_fraction_p95"])
        and max_role_p95 <= float(gate["maximum_role_repair_fraction_p95"])
    )

    integrity_failures: list[str] = []
    if replay_mismatches:
        integrity_failures.append(f"exact_replay_mismatches:{replay_mismatches}")
    if identity["resolved_revision"] not in ("", config["revision"]):
        integrity_failures.append("resolved_checkpoint_revision")

    if integrity_failures:
        decision = "INVALID_DYADIC_REASSOCIATION_LOCK_CONTROL_FAILURE"
    elif passes:
        decision = "PROMOTE_DYADIC_REASSOCIATION_LOCK_TO_EXPLICIT_RECTANGULAR_FMM_GATE"
    else:
        decision = "REJECT_REASSOCIATED_EXACT_DOT_NATIVE_ORDER_REPAIR_AS_10X_CORE"

    deterministic_core = {
        "schema": config["schema"],
        "config_sha256": canonical_sha256(config),
        "checkpoint": identity,
        "traces": traces,
        "build": build,
        "holdout": holdout,
        "maximum_holdout_role_repair_p95": max_role_p95,
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
            "fp64_exact_real_proxy_with_roundoff_enclosure": True,
            "official_match_selector": "PERFECT_ORACLE_FOR_FIRST_GATE",
            "explicit_fast_rectangular_algorithm": "NOT_CONSTRUCTED",
            "causal_future_block": "NOT_CONSTRUCTED",
            "target_405b": "NOT_TESTED",
            "physical_8gib": "NOT_TESTED",
            "same_machine_4b_latency": "NOT_TESTED"
        }
    }
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "raw/projection_rows.json", rows)
    write_json(output_dir / "raw/layer_state_rows.json", result["layer_state_rows"])
    write_json(output_dir / "artifacts/deterministic_core.json", deterministic_core)
    print(json.dumps({
        "authoritative_decision": decision,
        "holdout_repair_p50": holdout["native_order_repair_fraction_p50"],
        "holdout_repair_p95": holdout["native_order_repair_fraction_p95"],
        "generic_fp32_certificate_p50": holdout["generic_fp32_certificate_fraction_p50"],
        "maximum_holdout_role_repair_p95": max_role_p95,
        "integrity_failures": integrity_failures,
    }, indent=2, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    execute(args.config, args.output_dir)


if __name__ == "__main__":
    main()
