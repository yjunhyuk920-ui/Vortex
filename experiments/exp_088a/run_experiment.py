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
from transformers import LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.cross_layer_template_code import (  # noqa: E402
    audit_cross_layer_templates,
)
from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    PINNED_SAFETENSORS,
    PINNED_TORCH,
    PINNED_TRANSFORMERS,
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def load_model(config: dict[str, Any]) -> tuple[Any, int]:
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
    if model.__class__.__name__ != "LlamaForCausalLM":
        raise RuntimeError("official loader class mismatch")
    return model, time.perf_counter_ns() - start


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = verify_runtime()
    torch.manual_seed(int(config["seed"]))
    torch.set_num_threads(int(config["torch_num_threads"]))
    model, load_ns = load_model(config)
    layers = list(model.model.layers)
    if config["layer_selection_rule"] != "all_decoder_layers" or len(layers) < 3:
        raise RuntimeError("invalid frozen layer selection")

    gate_weights = [layer.mlp.gate_proj.weight for layer in layers]
    up_weights = [layer.mlp.up_proj.weight for layer in layers]
    down_weights_transposed = [
        layer.mlp.down_proj.weight.transpose(0, 1).contiguous()
        for layer in layers
    ]
    audit_start = time.perf_counter_ns()
    audit = audit_cross_layer_templates(
        gate_weights=gate_weights,
        up_weights=up_weights,
        down_weights_transposed=down_weights_transposed,
        template_groups=int(config["template_groups"]),
    )
    audit_ns = time.perf_counter_ns() - audit_start
    aggregate = audit["aggregate"]
    integrity_passed = bool(aggregate["all_reconstructions_exact"])
    information_passed = (
        float(aggregate["total_information_fraction"])
        <= float(config["success"]["total_information_fraction_max"])
        and float(aggregate["effective_layer_fraction_p50"])
        <= float(config["success"]["effective_layer_fraction_p50_max"])
        and float(aggregate["effective_layer_fraction_p95"])
        <= float(config["success"]["effective_layer_fraction_p95_max"])
    )
    if not integrity_passed:
        decision = "INVALID_CROSS_LAYER_TEMPLATE_RECONSTRUCTION_FAILURE"
    elif information_passed:
        decision = "PROMOTE_CROSS_LAYER_TEMPLATE_CODE_TO_FUSED_QUERY_GATE"
    else:
        decision = "REJECT_CROSS_LAYER_TEMPLATE_AND_RECURRENCE_CODE_AS_COLD_QUERY_CORE"

    core = {
        "schema": config["schema"],
        "config_sha256": file_sha256(config_path),
        "checkpoint": {
            "model_id": config["model_id"],
            "revision": config["revision"],
        },
        "template_groups": int(config["template_groups"]),
        "audit": audit,
        "gates": {
            "integrity_passed": integrity_passed,
            "information_passed": information_passed,
        },
        "decision": decision,
    }
    result = {
        "experiment": "EXP-088A",
        "name": "cross_layer_template_and_recurrence_code_gate",
        "source_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNVERIFIED"),
        "evidence_level": "E1",
        "phase": [
            "A-structure",
            "B-reference",
            "C-small-real-checkpoint-weight-observation",
        ],
        "runtime": runtime,
        "config_sha256": file_sha256(config_path),
        "load_wall_ns": load_ns,
        "audit_wall_ns": audit_ns,
        "MEASURED": {
            "official_checkpoint_loaded": True,
            "model_forward_calls": 0,
            "decoder_layer_count": len(layers),
            "role_rows": audit["role_rows"],
        },
        "DERIVED": {
            "aggregate": aggregate,
            "success_thresholds": config["success"],
        },
        "gates": core["gates"],
        "authoritative_decision": decision,
        "deterministic_core_sha256": digest(canonical(core)),
        "claim_boundary": {
            "actual_checkpoint_weights": True,
            "all_decoder_mlp_layers": True,
            "model_forward_calls": 0,
            "role_wise_oracle_generator_choice": True,
            "ideal_entropy_coder": True,
            "complete_mlp_replaced": False,
            "dense_arithmetic_reduced": "NOT_TESTED",
            "complete_transformer_layer": "NOT_TESTED",
            "successor_state": "NOT_TESTED",
            "405b_execution": "NOT_TESTED",
            "8gib_gpu": "NOT_TESTED",
            "physical_latency": "NOT_TESTED",
        },
        "UNVERIFIED": [
            "generator sublinear in both matrix area and layer count",
            "cross-attention/MLP shared generator",
            "fused exact query without reconstructing weight words",
            "complete Transformer transition and successor-state equality",
            "405B same-machine 4B-class p50/p95",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "artifacts").mkdir(exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "raw/role_rows.json").write_text(
        json.dumps(audit["role_rows"], indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "artifacts/deterministic_core.json").write_bytes(canonical(core))
    paths = ["result.json", "raw/role_rows.json", "artifacts/deterministic_core.json"]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(f"{file_sha256(output_dir / path)}  {path}" for path in paths)
        + "\n",
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
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "experiment": "EXP-088A",
            "source_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNVERIFIED"),
            "authoritative_decision": "INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION",
            "error": f"{type(exc).__name__}: {exc}",
            "claim_boundary": {"mechanism_scientifically_rejected": False},
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
