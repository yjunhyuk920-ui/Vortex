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
from torch import nn
from transformers import AutoTokenizer, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.exact_swiglu_global_compiler import (  # noqa: E402
    FINAL_P50_TARGET_FRACTION,
    MECHANISM_FINGERPRINT,
    ExactGlobalSwiGLUProgram,
    aggregate_plans,
    analyze_swiglu,
    probe_native_abi,
)
from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    PINNED_SAFETENSORS,
    PINNED_TORCH,
    PINNED_TRANSFORMERS,
    environment_manifest,
    module_parameter_bytes,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    import safetensors

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
    }


def verify_pins() -> None:
    actual = package_versions()
    expected = {
        "torch": PINNED_TORCH,
        "transformers": PINNED_TRANSFORMERS,
        "safetensors": PINNED_SAFETENSORS,
    }
    bad = {key: {"expected": expected[key], "actual": actual[key]} for key in expected if expected[key] != actual[key]}
    if bad:
        raise RuntimeError(f"pinned runtime mismatch: {bad}")


def load_dev() -> tuple[Any, Any, int]:
    started = time.perf_counter_ns()
    model = LlamaForCausalLM.from_pretrained(
        DEV_MODEL_ID,
        revision=DEV_REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
    )
    model.eval()
    setattr(model, "_vortex_revision", DEV_REVISION)
    if model.__class__.__name__ != "LlamaForCausalLM" or model.config.model_type != "llama":
        raise RuntimeError("official Llama loader ABI mismatch")
    resolved = str(getattr(model.config, "_commit_hash", "") or "")
    if resolved and resolved != DEV_REVISION:
        raise RuntimeError(f"resolved revision mismatch: {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(DEV_MODEL_ID, revision=DEV_REVISION, use_fast=True)
    return model, tokenizer, time.perf_counter_ns() - started


def render(spec: dict[str, Any]) -> str:
    if "prompt" in spec:
        return str(spec["prompt"])
    return str(spec.get("prompt_repeat", "")) * int(spec.get("repeat_count", 1)) + str(spec.get("suffix", ""))


def capture_layer0_mlp_inputs(model: Any, tokenizer: Any, workload: dict[str, Any]) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("MLP pre-hook did not receive hidden states")
        captured.append(args[0].detach().cpu())

    handle = model.model.layers[0].mlp.register_forward_pre_hook(hook)
    try:
        for spec in workload["workloads"]:
            ids = tokenizer(render(spec), return_tensors="pt")["input_ids"]
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("no actual checkpoint MLP input captured")
    return torch.cat([value.reshape(-1, value.shape[-1]) for value in captured], dim=0)


def adversarial_inputs(hidden: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260819)
    rows = [
        torch.zeros(1, hidden, dtype=dtype),
        torch.ones(1, hidden, dtype=dtype),
        -torch.ones(1, hidden, dtype=dtype),
        torch.full((1, hidden), 2.0**-7, dtype=dtype),
        torch.full((1, hidden), -(2.0**-7), dtype=dtype),
    ]
    for scale in (2.0**-8, 0.125, 1.0, 8.0):
        rows.append((torch.randn(16, hidden, generator=generator, dtype=torch.float32) * scale).to(dtype))
    basis_indices = torch.linspace(0, hidden - 1, steps=min(32, hidden)).round().to(torch.long)
    basis = torch.zeros(len(basis_indices), hidden, dtype=dtype)
    basis[torch.arange(len(basis_indices)), basis_indices] = 1
    rows.extend((basis, -basis))
    return torch.cat(rows, dim=0)


def structured_control() -> dict[str, Any]:
    class ControlMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(8, 12, bias=False)
            self.up_proj = nn.Linear(8, 12, bias=False)
            self.down_proj = nn.Linear(12, 8, bias=False)
            self.act_fn = nn.SiLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    torch.manual_seed(20260819)
    module = ControlMLP().eval()
    with torch.no_grad():
        module.gate_proj.weight[1].copy_(module.gate_proj.weight[0])
        module.up_proj.weight[1].copy_(module.up_proj.weight[0])
        module.gate_proj.weight[3].copy_(-module.gate_proj.weight[2])
        module.up_proj.weight[3].copy_(module.up_proj.weight[2])
        module.down_proj.weight[:, 5].zero_()
    plan = analyze_swiglu(module)
    passed = (
        plan.duplicate_gate_up_atoms >= 1
        and plan.antipodal_gate_pairs >= 1
        and plan.signed_gate_up_pairs >= 1
        and plan.zero_down_columns == 1
        and plan.optimistic_algebraic_operation_fraction < 1.0
    )
    return {"passed": passed, "plan": plan.to_dict(include_groups=False)}


def random_control() -> dict[str, Any]:
    class ControlMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(8, 12, bias=False)
            self.up_proj = nn.Linear(8, 12, bias=False)
            self.down_proj = nn.Linear(12, 8, bias=False)
            self.act_fn = nn.SiLU()

    torch.manual_seed(777)
    plan = analyze_swiglu(ControlMLP().eval())
    passed = (
        plan.duplicate_gate_rows == 0
        and plan.duplicate_up_rows == 0
        and plan.duplicate_gate_up_atoms == 0
        and plan.optimistic_algebraic_operation_fraction == 1.0
    )
    return {"passed": passed, "plan": plan.to_dict(include_groups=False)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workload", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_path = Path(args.workload)
    workload = json.loads(workload_path.read_text())
    if not workload.get("frozen_before_results"):
        raise RuntimeError("workload must be frozen before the compiler result")

    verify_pins()
    model, tokenizer, load_ns = load_dev()
    plans = [analyze_swiglu(layer.mlp) for layer in model.model.layers]
    aggregate = aggregate_plans(plans)
    actual_inputs = capture_layer0_mlp_inputs(model, tokenizer, workload)
    adversarial = adversarial_inputs(model.config.hidden_size, actual_inputs.dtype)
    probe_inputs = torch.cat((actual_inputs, adversarial), dim=0)
    reference_mlp = model.model.layers[0].mlp
    reference_program = ExactGlobalSwiGLUProgram(reference_mlp, mode="reference_order").eval()
    joint_program = ExactGlobalSwiGLUProgram(reference_mlp, mode="joint_gate_up").eval()
    reference_probe = probe_native_abi(reference_mlp, reference_program, probe_inputs, mode="reference_order")
    joint_probe = probe_native_abi(reference_mlp, joint_program, probe_inputs, mode="joint_gate_up")
    controls = {
        "structured_positive": structured_control(),
        "dense_random_negative": random_control(),
    }
    controls_passed = all(bool(value["passed"]) for value in controls.values()) and reference_probe.exact

    if not controls_passed:
        verdict = "INVALID_EXACT_SWIGLU_GLOBAL_COMPILER_CONTROL_FAILURE"
    elif not aggregate["structural_gate_passed"]:
        verdict = "REJECT_EXACT_SWIGLU_EQUIVALENCE_CSE_AS_CORE"
    elif not joint_probe.exact:
        verdict = "REJECT_JOINT_GATE_UP_NATIVE_ABI_PATH"
    else:
        verdict = "PROMOTE_EXACT_SWIGLU_GLOBAL_COMPILER_TO_OPERATION_REPLACEMENT_GATE"

    result = {
        "schema": "exact-swiglu-global-compiler-result-v1",
        "experiment": "EXP-085A",
        "name": "exact_swiglu_global_compiler_gate",
        "mechanism": MECHANISM_FINGERPRINT,
        "github": {
            "repository": os.getenv("GITHUB_REPOSITORY"),
            "sha": os.getenv("GITHUB_SHA"),
            "ref": os.getenv("GITHUB_REF"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
        },
        "pins": {
            "DEV-W": {"model_id": DEV_MODEL_ID, "revision": DEV_REVISION},
            "runtime": package_versions(),
        },
        "environment": environment_manifest(),
        "workload_sha256": sha256_file(workload_path),
        "official_loader": "transformers.LlamaForCausalLM.from_pretrained",
        "official_checkpoint_loaded": True,
        "reference_model_parameter_bytes": module_parameter_bytes(model),
        "reference_load_ns": load_ns,
        "architecture": {
            "layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "activation": model.config.hidden_act,
            "dtype": str(model.config.torch_dtype),
        },
        "MEASURED": {
            "layers_audited": len(plans),
            "actual_prompt_input_rows": int(actual_inputs.shape[0]),
            "adversarial_input_rows": int(adversarial.shape[0]),
            "reference_abi_probe": reference_probe.to_dict(),
            "joint_gate_up_abi_probe": joint_probe.to_dict(),
            "controls": controls,
            "layer_plans": [plan.to_dict(include_groups=False) for plan in plans],
        },
        "DERIVED": {
            "aggregate": aggregate,
            "target_fraction": FINAL_P50_TARGET_FRACTION,
            "distance_to_target": aggregate["optimistic_algebraic_operation_fraction"] / FINAL_P50_TARGET_FRACTION,
            "decision": verdict,
        },
        "authoritative_decision": verdict,
        "highest_evidence": "E1_actual_public_checkpoint_structural_and_native_ABI_gate",
        "claim_boundary": {
            "whole_swiglu_compiler": "IMPLEMENTED_AND_EXECUTED",
            "complete_transformer_layer_replacement": "NOT_AUTHORIZED_UNLESS_STRUCTURAL_GATE_PASSES",
            "successor_state": "NOT_TESTED_IN_THIS_PROOF_FIRST_GATE",
            "405B": "PROJECTED_SHAPE_RATIO_ONLY_NOT_EXECUTED",
            "8_GiB": "NOT_TESTED",
            "wall_clock_speedup": "NOT_CLAIMED",
            "E6_E7": "NOT_ACHIEVED",
        },
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["deterministic_core_sha256"] = hashlib.sha256(canonical).hexdigest()
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (output_dir / "run.log").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "aggregate": aggregate,
                "reference_probe": reference_probe.to_dict(),
                "joint_probe": joint_probe.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "layers": len(plans),
                "optimistic_fraction": aggregate["optimistic_algebraic_operation_fraction"],
                "target_fraction": FINAL_P50_TARGET_FRACTION,
                "joint_gate_up_abi_exact": joint_probe.exact,
            },
            indent=2,
        )
    )
    if not controls_passed:
        raise RuntimeError("compiler controls failed")


if __name__ == "__main__":
    main()
