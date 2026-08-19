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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fixed_public_dynamic_executor.run_real_checkpoint import (  # noqa: E402
    choose_prompt,
    digest_file,
    hf_identity,
    load_dev,
    official_128,
    packages,
    render,
    transition,
    verify_pins,
)
from vortex_runtime.exact_swiglu_program import (  # noqa: E402
    DEFAULT_FIBER_TILE_ROWS,
    DEFAULT_OUTPUT_TILE_ROWS,
    MECHANISM,
    TARGET_EQUIVALENT_FRACTION,
    compile_checkpoint_global_swiglu,
)
from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    TARGET_MODEL_ID,
    TARGET_REVISION,
    environment_manifest,
    module_parameter_bytes,
    percentile_ns,
)
from vortex_runtime.fixed_public_dynamic_executor import (  # noqa: E402
    make_reference_artifact,
    physical_gate,
)
from vortex_runtime.fixed_public_dynamic_inductor import refresh_artifact_bytes  # noqa: E402


def _sha256_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _run_compiler(
    reference: Any,
    tokenizer: Any,
    workload: dict[str, Any],
    output_dir: Path,
    official_tokens: list[int],
) -> dict[str, Any]:
    candidate, _, candidate_load_ns = load_dev()
    artifact_dir = output_dir / "compiler_artifact"
    compile_start = time.perf_counter_ns()
    artifact = compile_checkpoint_global_swiglu(
        candidate,
        artifact_dir=artifact_dir,
        layer_index=0,
        fiber_tile_rows=DEFAULT_FIBER_TILE_ROWS,
        output_tile_rows=DEFAULT_OUTPUT_TILE_ROWS,
        verify_integrity=True,
        run_fiber_tree_screen=True,
    )
    compile_outer_ns = time.perf_counter_ns() - compile_start
    refresh_artifact_bytes(artifact)
    manifest = json.loads(Path(artifact.manifest_path).read_text())

    reference_artifact = make_reference_artifact(reference)
    runs: list[dict[str, Any]] = []
    prompt_selection_evidence: dict[str, Any] = {}
    for index, spec in enumerate(workload["workloads"]):
        prompt, selection = choose_prompt(reference, tokenizer, spec)
        prompt_selection_evidence[spec["id"]] = selection if selection else None
        row = transition(
            reference_artifact,
            artifact,
            tokenizer,
            prompt,
            spec,
            int(workload["decode_steps"]),
            official_tokens if index == 0 else None,
        )
        row["id"] = spec["id"]
        row["category"] = spec["category"]
        runs.append(row)
        if not row["exact_transition_pass"]:
            break

    baseline_latency = [value for row in runs for value in row["reference_latency_ns"]]
    candidate_latency = [value for row in runs for value in row["candidate_latency_ns"]]
    records = getattr(getattr(artifact.runtime_layer, "mlp", None), "records", {})
    peak_hot = max((record.raw_bytes for record in records.values()), default=0)
    gate = physical_gate(
        baseline_p50_ns=percentile_ns(baseline_latency, 0.5),
        baseline_p95_ns=percentile_ns(baseline_latency, 0.95),
        candidate_p50_ns=percentile_ns(candidate_latency, 0.5),
        candidate_p95_ns=percentile_ns(candidate_latency, 0.95),
        reference_layer_parameter_bytes=artifact.replaced_layer_reference_parameter_bytes,
        artifact_bytes=artifact.artifact_bytes,
        candidate_layer_resident_parameter_bytes=artifact.compiled_layer_resident_parameter_bytes,
        peak_projection_hot_bytes=peak_hot,
    )

    exact = (
        len(runs) == len(workload["workloads"])
        and all(row["exact_transition_pass"] for row in runs)
    )
    complete_steps = sum(int(row["steps_completed"]) for row in runs)
    expected_steps = len(workload["workloads"]) * int(workload["decode_steps"])
    continuous = bool(exact and complete_steps == expected_steps and expected_steps >= 128)
    full_layer = bool(artifact.layer_index == 0 and artifact.runtime_layer is not None)
    physical = bool(exact and full_layer and continuous and gate["passed"])

    audit = manifest["compiler_audit"]
    fiber_screen = audit["q4_nonlinear_fiber_tree_screen"]
    exact_identity_fraction = float(audit["identity_cse_optimistic_coefficient_fraction"])
    q4_fiber_tree_fraction = (
        float(fiber_screen["certified_fraction"])
        if fiber_screen.get("status") == "EXECUTED"
        else None
    )

    # The initial rule set intentionally contains no rewrite that can be called
    # a target-scale BF16 core.  Identity CSE is exact but usually dense; the
    # tree screen is Q4-surrogate evidence only.
    implemented_target_scale_native_rewrite = False
    verdict = (
        "REJECT_GLOBAL_COMPILER_NATIVE_ABI"
        if not exact
        else "GLOBAL_NONLINEAR_COMPILER_SUBSTRATE_ESTABLISHED_NO_CORE_REWRITE"
    )

    traces = [trace for row in runs for trace in row["candidate_resource_trace"]]
    trace_totals = {
        key: sum(int(trace.get(key, 0) or 0) for trace in traces)
        for key in (
            "cold_bytes",
            "hot_materialized_bytes",
            "projection_calls",
            "integrity_probes",
            "decompression_ns",
            "materialization_ns",
        )
    }
    trace_per_transition = {
        key: (value / complete_steps if complete_steps else None)
        for key, value in trace_totals.items()
    }

    return {
        "mechanism": MECHANISM,
        "verdict": verdict,
        "candidate_load_ns": candidate_load_ns,
        "compile_wall_ns": compile_outer_ns,
        "compile_peak_rss_bytes": artifact.compile_peak_rss_bytes,
        "artifact_bytes": artifact.artifact_bytes,
        "manifest_sha256": artifact.manifest_sha256,
        "checkpoint_tensor_sha256": artifact.checkpoint_tensor_sha256,
        "config_sha256": artifact.config_sha256,
        "reference_layer_parameter_bytes": artifact.replaced_layer_reference_parameter_bytes,
        "compiled_layer_resident_parameter_bytes": artifact.compiled_layer_resident_parameter_bytes,
        "peak_projection_hot_bytes": peak_hot,
        "selected_joint_affine_mode": manifest["native_abi_guard"]["selected_joint_affine_mode"],
        "native_abi_guard": manifest["native_abi_guard"],
        "compiler_audit": audit,
        "identity_cse_optimistic_coefficient_fraction": exact_identity_fraction,
        "q4_fiber_tree_certified_fraction": q4_fiber_tree_fraction,
        "target_equivalent_fraction": TARGET_EQUIVALENT_FRACTION,
        "implemented_target_scale_native_rewrite": implemented_target_scale_native_rewrite,
        "G2_exact_transition": exact,
        "G3_full_transformer_layer_boundary": full_layer,
        "G4_continuous_decode": continuous,
        "G5_physical_saving": physical,
        "physical_gate": gate,
        "expected_transitions": expected_steps,
        "completed_transitions": complete_steps,
        "trace_totals": trace_totals,
        "trace_per_transition": trace_per_transition,
        "prompt_selection_evidence": prompt_selection_evidence,
        "runs": runs,
        "claim_boundary": {
            "compiler_semantic_unit": "complete SwiGLU function",
            "initial_lowering": "existing PyTorch affine/SiLU/multiply operations",
            "matrix_unit_execution_fully_eliminated": False,
            "native_all_input_theorem": False,
            "q4_fiber_screen_is_bf16_proof": False,
            "405b_execution": "NOT_TESTED",
            "eight_gib_target": "NOT_TESTED",
            "four_b_class_latency": "NOT_TESTED",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workload", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_path = Path(args.workload)
    workload = json.loads(workload_path.read_text())
    decode_steps = int(workload.get("decode_steps", 0))
    if not workload.get("frozen_before_results") or decode_steps < 128:
        raise RuntimeError("workload is not pre-frozen at >=128 steps")

    verify_pins()
    dev_identity = hf_identity(DEV_MODEL_ID, DEV_REVISION, True)
    target_identity = hf_identity(TARGET_MODEL_ID, TARGET_REVISION, False)
    reference, tokenizer, reference_load_ns = load_dev()
    first_prompt = render(workload["workloads"][0])
    official_tokens = official_128(reference, tokenizer, first_prompt, decode_steps)
    with torch.inference_mode():
        forward = reference(
            input_ids=tokenizer(first_prompt, return_tensors="pt")["input_ids"],
            use_cache=True,
            return_dict=True,
        )

    result: dict[str, Any] = {
        "schema": "exact-swiglu-global-compiler-result-v1",
        "github": {
            "repository": os.getenv("GITHUB_REPOSITORY"),
            "sha": os.getenv("GITHUB_SHA"),
            "ref": os.getenv("GITHUB_REF"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
        },
        "packages": packages(),
        "environment": environment_manifest(),
        "workload_sha256": digest_file(workload_path),
        "pins": {
            "DEV-W": {"repo_id": DEV_MODEL_ID, "revision": DEV_REVISION},
            "TARGET-W": {"repo_id": TARGET_MODEL_ID, "revision": TARGET_REVISION},
        },
        "hf_resolution": {"DEV-W": dev_identity, "TARGET-W": target_identity},
        "reference_load_ns": reference_load_ns,
        "architecture": {
            "class": reference.__class__.__name__,
            "model_type": reference.config.model_type,
            "hidden_size": reference.config.hidden_size,
            "intermediate_size": reference.config.intermediate_size,
            "num_hidden_layers": reference.config.num_hidden_layers,
            "num_attention_heads": reference.config.num_attention_heads,
            "num_key_value_heads": reference.config.num_key_value_heads,
            "hidden_act": reference.config.hidden_act,
            "rms_norm_eps": reference.config.rms_norm_eps,
            "torch_dtype": str(reference.config.torch_dtype),
            "parameter_bytes": module_parameter_bytes(reference),
        },
        "official_reference_forward": {
            "executed": True,
            "logits_sha256": _sha256_tensor(forward.logits),
        },
        "official_reference_generation": {
            "steps": len(official_tokens),
            "tokens": official_tokens,
        },
        "preregistration": {
            "document": "docs/research/EXACT_SWIGLU_GLOBAL_COMPILER_GATE.md",
            "fiber_tile_rows": DEFAULT_FIBER_TILE_ROWS,
            "output_tile_rows": DEFAULT_OUTPUT_TILE_ROWS,
            "target_equivalent_fraction": TARGET_EQUIVALENT_FRACTION,
        },
    }
    compiler = _run_compiler(
        reference,
        tokenizer,
        workload,
        output_dir,
        official_tokens,
    )
    result["compiler"] = compiler
    result["verdict"] = compiler["verdict"]
    result["highest_gate"] = (
        "G5_PHYSICAL_SAVING"
        if compiler["G5_physical_saving"]
        else "G4_CONTINUOUS_DECODE"
        if compiler["G4_continuous_decode"]
        else "G3_REAL_BOUNDARY"
        if compiler["G3_full_transformer_layer_boundary"]
        else "G2_EXACT_TRANSITION"
        if compiler["G2_exact_transition"]
        else "G1_ACTUAL_RUNTIME"
    )
    result["next_gate"] = (
        "Add a materially nonlinear proof-carrying rewrite to the established whole-SwiGLU IR and require a fully charged native target-scale equation before executor expansion."
    )

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "verdict": result["verdict"],
        "highest_gate": result["highest_gate"],
        "selected_joint_affine_mode": compiler["selected_joint_affine_mode"],
        "completed_transitions": compiler["completed_transitions"],
        "G5_physical_saving": compiler["G5_physical_saving"],
        "identity_cse_fraction": compiler["identity_cse_optimistic_coefficient_fraction"],
        "q4_fiber_tree_fraction": compiler["q4_fiber_tree_certified_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
