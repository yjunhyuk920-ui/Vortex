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

from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    PINNED_SAFETENSORS,
    PINNED_TORCH,
    PINNED_TRANSFORMERS,
)
from vortex_runtime.lossless_entropy_stationary import (  # noqa: E402
    P50_WHOLE_MODEL_FRACTION,
    P95_WHOLE_MODEL_FRACTION,
    block_linear_report,
    decode_tensor_zlib,
    encode_tensor_zlib,
    full_mlp_block_exactness,
    sha256_bytes,
    target_surface,
    tensor_bytes,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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
    bad = {
        key: [expected[key], actual[key]]
        for key in expected
        if expected[key] != actual[key]
    }
    if bad:
        raise RuntimeError(f"runtime pin mismatch: {bad}")
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


def capture_mlp_inputs(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    layer_index: int,
    positions: int,
) -> tuple[torch.Tensor, int]:
    rows: list[torch.Tensor] = []

    def hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        rows.append(inputs[0][:, -1, :].detach().contiguous().cpu())

    handle = model.model.layers[int(layer_index)].mlp.register_forward_pre_hook(hook)
    calls = 0
    try:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            output = model(input_ids=ids, use_cache=True, return_dict=True)
            calls += 1
            past = output.past_key_values
            token = int(torch.argmax(output.logits[0, -1]).item())
            for _ in range(1, int(positions)):
                output = model(
                    input_ids=torch.tensor([[token]], dtype=torch.long),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                calls += 1
                past = output.past_key_values
                token = int(torch.argmax(output.logits[0, -1]).item())
    finally:
        handle.remove()
    if len(rows) != int(positions):
        raise RuntimeError(f"captured {len(rows)} MLP inputs, expected {positions}")
    return torch.cat(rows, dim=0), calls


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise RuntimeError("cannot aggregate empty values")
    return float(
        np.percentile(np.asarray(values, dtype=np.float64), q, method="linear")
    )


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = verify_runtime()
    if config["p50_whole_model_fraction"] != P50_WHOLE_MODEL_FRACTION:
        raise RuntimeError("p50 target mismatch")
    if config["p95_whole_model_fraction"] != P95_WHOLE_MODEL_FRACTION:
        raise RuntimeError("p95 target mismatch")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    model, tokenizer, load_ns = load_model(config)
    mlp = model.model.layers[int(config["layer_index"])].mlp
    source_weights = {
        "gate_proj": mlp.gate_proj.weight.detach().cpu().contiguous(),
        "up_proj": mlp.up_proj.weight.detach().cpu().contiguous(),
        "down_proj": mlp.down_proj.weight.detach().cpu().contiguous(),
    }

    compression_rows: list[dict[str, Any]] = []
    decoded_weights: dict[str, torch.Tensor] = {}
    roundtrip_mismatches = 0
    total_raw = 0
    total_encoded = 0
    total_shannon = 0.0
    for name, weight in source_weights.items():
        encoded = encode_tensor_zlib(weight, tile_bytes=int(config["tile_bytes"]))
        decoded, decode_ns = decode_tensor_zlib(encoded, dtype=weight.dtype)
        mismatch = int(tensor_bytes(decoded) != tensor_bytes(weight))
        roundtrip_mismatches += mismatch
        decoded_weights[name] = decoded
        row = encoded.summary()
        row.update(
            {
                "name": name,
                "decode_ns": int(decode_ns),
                "roundtrip_mismatch": mismatch,
                "shape": list(weight.shape),
            }
        )
        compression_rows.append(row)
        total_raw += encoded.raw_bytes
        total_encoded += encoded.encoded_bytes
        total_shannon += encoded.shannon_word16_bytes

    global_zlib_ratio = total_raw / max(1, total_encoded)
    global_shannon_ratio = total_raw / max(1.0, total_shannon)
    prompt_rows: list[dict[str, Any]] = []
    target_calls = 0
    for spec in config["prompts"]:
        activations, calls = capture_mlp_inputs(
            model,
            tokenizer,
            spec["prompt"],
            layer_index=int(config["layer_index"]),
            positions=int(config["decode_positions_per_prompt"]),
        )
        target_calls += calls
        for block_size in map(int, config["block_sizes"]):
            for start in range(
                0, int(activations.shape[0]) - block_size + 1, block_size
            ):
                block = activations[start : start + block_size]
                gate_block = torch.nn.functional.linear(
                    block, decoded_weights["gate_proj"]
                )
                up_block = torch.nn.functional.linear(
                    block, decoded_weights["up_proj"]
                )
                down_input = mlp.act_fn(gate_block) * up_block
                projection_inputs = {
                    "gate_proj": block,
                    "up_proj": block,
                    "down_proj": down_input,
                }
                projection_reports = {
                    name: block_linear_report(
                        projection_inputs[name],
                        decoded_weights[name],
                        repeats=int(config["timing_repeats"]),
                    ).to_dict()
                    for name in ("gate_proj", "up_proj", "down_proj")
                }
                full_report = full_mlp_block_exactness(
                    block,
                    gate_weight=decoded_weights["gate_proj"],
                    up_weight=decoded_weights["up_proj"],
                    down_weight=decoded_weights["down_proj"],
                    act_fn=mlp.act_fn,
                )
                prompt_rows.append(
                    {
                        "prompt_id": spec["id"],
                        "family": spec["family"],
                        "block_size": block_size,
                        "block_start": start,
                        "activation_sha256": sha256_bytes(tensor_bytes(block)),
                        "projection_reports": projection_reports,
                        "full_mlp_report": full_report,
                    }
                )

    block_mismatches = sum(
        int(row["full_mlp_report"]["mismatch_count"])
        + sum(
            int(report["mismatch_count"])
            for report in row["projection_reports"].values()
        )
        for row in prompt_rows
    )
    aggregate_by_k: dict[str, Any] = {}
    for block_size in map(int, config["block_sizes"]):
        selected = [row for row in prompt_rows if row["block_size"] == block_size]
        full_exact = [
            float(row["full_mlp_report"]["vector_exact_fraction"])
            for row in selected
        ]
        block_speedups = [
            float(report["block_speedup"])
            for row in selected
            for report in row["projection_reports"].values()
        ]
        aggregate_by_k[str(block_size)] = {
            "count": len(selected),
            "full_mlp_vector_exact_p05": percentile(full_exact, 5),
            "full_mlp_vector_exact_p50": percentile(full_exact, 50),
            "full_mlp_vector_exact_min": min(full_exact),
            "projection_block_speedup_p05": percentile(block_speedups, 5),
            "projection_block_speedup_p50": percentile(block_speedups, 50),
            "projection_block_speedup_p95": percentile(block_speedups, 95),
            "zlib_weight_fraction": 1.0 / (global_zlib_ratio * block_size),
            "shannon_weight_fraction": 1.0 / (global_shannon_ratio * block_size),
        }

    ratios = sorted(
        {
            1.0,
            float(global_zlib_ratio),
            float(global_shannon_ratio),
            *map(float, config["external_compression_ratios"]),
        }
    )
    surface = [
        row.to_dict()
        for row in target_surface(
            compression_ratios=ratios, block_sizes=config["block_sizes"]
        )
    ]
    exactness_passed = roundtrip_mismatches == 0 and block_mismatches == 0
    physical_saving = total_encoded < total_raw
    k128 = aggregate_by_k[str(max(map(int, config["block_sizes"])))]
    io_screen_passed = (
        k128["zlib_weight_fraction"] <= P50_WHOLE_MODEL_FRACTION
    )
    if not exactness_passed:
        verdict = "REVISE_LOSSLESS_ENTROPY_STATIONARY_NATIVE_BLOCK_ABI_MISMATCH"
    elif not physical_saving:
        verdict = "REJECT_ZLIB_TILE_CODEC_RETAIN_ENTROPY_LOWER_BOUND"
    elif io_screen_passed:
        verdict = (
            "CONDITIONAL_LOSSLESS_ENTROPY_STATIONARY_PATH_REQUIRES_"
            "TARGET_BASELINE_AND_TARGET_ENTROPY"
        )
    else:
        verdict = (
            "REVISE_LOSSLESS_ENTROPY_STATIONARY_REQUIRES_LONGER_BLOCK_"
            "OR_STRONGER_CODEC"
        )

    core = {
        "schema": config["schema"],
        "checkpoint": {
            "model_id": config["model_id"],
            "revision": config["revision"],
        },
        "config_sha256": sha256_file(config_path),
        "compression_rows": compression_rows,
        "global": {
            "raw_bytes": total_raw,
            "encoded_bytes": total_encoded,
            "shannon_bytes": total_shannon,
            "zlib_ratio": global_zlib_ratio,
            "shannon_ratio": global_shannon_ratio,
        },
        "prompt_rows": prompt_rows,
        "aggregate_by_k": aggregate_by_k,
        "target_surface": surface,
        "gates": {
            "exactness_passed": exactness_passed,
            "physical_saving": physical_saving,
            "io_screen_passed": io_screen_passed,
        },
        "verdict": verdict,
    }
    result = {
        "experiment": "EXP-087A",
        "name": "lossless_entropy_stationary_perfect_block_gate",
        "evidence_level": "E1",
        "phase": [
            "A-resource",
            "B-reference",
            "C-small-real-checkpoint-observation",
        ],
        "source_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNVERIFIED"),
        "runtime": runtime,
        "load_wall_ns": load_ns,
        "target_forward_calls": target_calls,
        "measured_prompt_count": len(config["prompts"]),
        "measured_block_rows": len(prompt_rows),
        "compression_rows": compression_rows,
        "global_compression": core["global"],
        "aggregate_by_k": aggregate_by_k,
        "target_surface": surface,
        "roundtrip_mismatches": roundtrip_mismatches,
        "block_mismatches": block_mismatches,
        "gates": core["gates"],
        "authoritative_decision": verdict,
        "deterministic_core_sha256": canonical_sha256(core),
        "claim_boundary": {
            "official_checkpoint_loaded": True,
            "actual_checkpoint_weights": True,
            "actual_checkpoint_activations": True,
            "future_target_activations_used": True,
            "lossless_roundtrip": exactness_passed,
            "target_405b_entropy": "NOT_TESTED",
            "target_native_4b_baseline": "NOT_TESTED",
            "target_effective_fp32_throughput": "NOT_TESTED",
            "405b_execution": "NOT_TESTED",
            "8gib_gpu": "NOT_TESTED",
        },
        "external_assumptions": {
            "published_lossless_entropy_range": (
                "2x-10x effective entropy reduction; not measured here"
            ),
            "m5000_fp32_peak_tflop_s": 4.3,
            "target_pcie_ceiling_gib_s": 7.4506,
            "stage2_baseline_required": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = output_dir / "raw"
    artifacts = output_dir / "artifacts"
    raw.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (raw / "block_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in prompt_rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            )
    (raw / "compression_rows.json").write_text(
        json.dumps(compression_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (raw / "target_surface.json").write_text(
        json.dumps(surface, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artifacts / "deterministic_core.json").write_text(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ),
        encoding="utf-8",
    )
    checks = []
    for relative in (
        "result.json",
        "raw/block_rows.jsonl",
        "raw/compression_rows.json",
        "raw/target_surface.json",
        "artifacts/deterministic_core.json",
    ):
        checks.append(f"{sha256_file(output_dir / relative)}  {relative}")
    (output_dir / "checksums.sha256").write_text(
        "\n".join(checks) + "\n", encoding="utf-8"
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
            "experiment": "EXP-087A",
            "source_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNVERIFIED"),
            "authoritative_decision": (
                "INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION"
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
