from __future__ import annotations

import argparse
import copy
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
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    TARGET_MODEL_ID,
    TARGET_REVISION,
    PINNED_TORCH,
    PINNED_TRANSFORMERS,
    PINNED_SAFETENSORS,
    RNGState,
    environment_manifest,
    module_parameter_bytes,
    percentile_ns,
)
from vortex_runtime.fixed_public_dynamic_executor import (  # noqa: E402
    compile_checkpoint,
    decode_step,
    initialize_state,
    make_reference_artifact,
    physical_gate,
    state_sha256,
    states_exact,
)
from vortex_runtime.fixed_public_dynamic_audit import audit_checkpoint_layer  # noqa: E402
from vortex_runtime.fixed_public_dynamic_inductor import compile_checkpoint_inductor, refresh_artifact_bytes  # noqa: E402
from vortex_runtime.fixed_public_dynamic_streaming import (  # noqa: E402
    DEFAULT_OUTPUT_TILE_ROWS,
    compile_checkpoint_row_streamed,
)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def packages() -> dict[str, str]:
    import safetensors

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
    }


def verify_pins() -> None:
    actual = packages()
    expected = {
        "torch": PINNED_TORCH,
        "transformers": PINNED_TRANSFORMERS,
        "safetensors": PINNED_SAFETENSORS,
    }
    bad = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
    if bad:
        raise RuntimeError(f"pinned runtime mismatch: {bad}")


def hf_identity(repo: str, revision: str, required: bool) -> dict[str, Any]:
    try:
        info = HfApi().model_info(repo, revision=revision)
        if info.sha != revision:
            raise RuntimeError(f"revision mismatch {info.sha} != {revision}")
        files = {}
        for name in ("config.json", "model.safetensors.index.json"):
            try:
                path = Path(hf_hub_download(repo, name, revision=revision))
                files[name] = {"sha256": digest_file(path), "bytes": path.stat().st_size}
            except Exception as exc:
                files[name] = {
                    "status": "NOT_ACCESSIBLE",
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
        return {
            "repo_id": repo,
            "requested_revision": revision,
            "resolved_sha": info.sha,
            "accessible": True,
            "files": files,
        }
    except Exception as exc:
        if required:
            raise
        return {
            "repo_id": repo,
            "requested_revision": revision,
            "resolved_sha": None,
            "accessible": False,
            "status": "NOT_TESTED_GATED_OR_UNAVAILABLE",
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }


def load_dev() -> tuple[Any, Any, int]:
    t0 = time.perf_counter_ns()
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
        raise RuntimeError(f"loaded SHA mismatch {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(DEV_MODEL_ID, revision=DEV_REVISION, use_fast=True)
    return model, tokenizer, time.perf_counter_ns() - t0


def render(spec: dict[str, Any]) -> str:
    if "prompt" in spec:
        return str(spec["prompt"])
    return str(spec.get("prompt_repeat", "")) * int(spec.get("repeat_count", 1)) + str(spec.get("suffix", ""))


def margin(model: Any, tokenizer: Any, prompt: str) -> float:
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[0, -1].float()
    values = torch.topk(logits, 2).values
    return float(values[0] - values[1])


def choose_prompt(model: Any, tokenizer: Any, spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "candidates" not in spec:
        return render(spec), {}
    scored = [(prompt, margin(model, tokenizer, prompt)) for prompt in spec["candidates"]]
    index = min(range(len(scored)), key=lambda i: (scored[i][1], i))
    return scored[index][0], {
        "selection_rule": spec["selection_rule"],
        "candidate_margins": scored,
        "selected_index": index,
    }


def rng(spec: dict[str, Any]) -> RNGState:
    if spec.get("mode") == "sample":
        return RNGState.sampled(
            int(spec["seed"]),
            temperature=float(spec.get("temperature", 1)),
            top_k=int(spec.get("top_k", 0)),
            top_p=float(spec.get("top_p", 1)),
        )
    return RNGState.greedy()


def official_128(model: Any, tokenizer: Any, prompt: str, steps: int) -> list[int]:
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    cfg = copy.deepcopy(model.generation_config)
    cfg.max_new_tokens = steps
    cfg.min_new_tokens = steps
    cfg.eos_token_id = None
    cfg.pad_token_id = tokenizer.eos_token_id or 0
    cfg.do_sample = False
    cfg.use_cache = True
    with torch.inference_mode():
        out = model.generate(ids, generation_config=cfg)
    result = [int(x) for x in out[0, ids.shape[1] :].tolist()]
    if len(result) != steps:
        raise RuntimeError(f"official generation length={len(result)}")
    return result


def transition(
    reference_artifact: Any,
    candidate_artifact: Any,
    tokenizer: Any,
    prompt: str,
    spec: dict[str, Any],
    steps: int,
    official: list[int] | None = None,
) -> dict[str, Any]:
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if ids.shape[1] < 2:
        raise RuntimeError("prompt must have >=2 tokens")
    state_rng = rng(spec)
    prefix, next_input = ids[:, :-1], ids[:, -1:]
    t0 = time.perf_counter_ns()
    reference_state = initialize_state(reference_artifact, prefix, rng_state=state_rng)
    reference_init = time.perf_counter_ns() - t0
    t0 = time.perf_counter_ns()
    candidate_state = initialize_state(candidate_artifact, prefix, rng_state=state_rng)
    candidate_init = time.perf_counter_ns() - t0
    init_equal = states_exact(reference_state, candidate_state)
    reference_tokens: list[int] = []
    candidate_tokens: list[int] = []
    reference_latency: list[int] = []
    candidate_latency: list[int] = []
    traces: list[dict[str, Any]] = []
    token_bad: list[int] = []
    state_bad: list[int] = []
    for index in range(steps):
        reference_token, reference_state, reference_trace = decode_step(reference_artifact, reference_state, next_input)
        candidate_token, candidate_state, candidate_trace = decode_step(candidate_artifact, candidate_state, next_input)
        reference_tokens.append(reference_token)
        candidate_tokens.append(candidate_token)
        reference_latency.append(reference_trace.wall_ns)
        candidate_latency.append(candidate_trace.wall_ns)
        traces.append(candidate_trace.to_dict())
        if reference_token != candidate_token:
            token_bad.append(index)
        if not states_exact(reference_state, candidate_state):
            state_bad.append(index)
        if token_bad or state_bad:
            break
        next_input = torch.tensor([[reference_token]], dtype=torch.long)
    official_match = None if official is None else official == reference_tokens
    exact = (
        init_equal
        and len(reference_tokens) == steps
        and not token_bad
        and not state_bad
        and official_match is not False
    )
    return {
        "prompt_tokens": int(ids.numel()),
        "reference_initialization_ns": reference_init,
        "candidate_initialization_ns": candidate_init,
        "steps_requested": steps,
        "steps_completed": len(reference_tokens),
        "token_mismatch_steps": token_bad,
        "state_mismatch_steps": state_bad,
        "reference_final_state_sha256": state_sha256(reference_state),
        "candidate_final_state_sha256": state_sha256(candidate_state),
        "manual_reference_matches_official_generate": official_match,
        "reference_tokens": reference_tokens,
        "candidate_tokens": candidate_tokens,
        "reference_latency_ns": reference_latency,
        "candidate_latency_ns": candidate_latency,
        "candidate_resource_trace": traces,
        "exact_transition_pass": bool(exact),
    }


def reuse_probe(model: Any, tokenizer: Any, prompt: str) -> dict[str, Any]:
    mlp_inputs: list[str] = []
    cross_equal = 0
    pending: dict[int, str] = {}

    def hash_tensor(tensor: torch.Tensor) -> str:
        raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def norm_hook(_module: Any, _inputs: Any, output: Any) -> None:
        mlp_inputs.append(hash_tensor(output))

    def layer_hook(index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            nonlocal cross_equal
            hidden = output[0] if isinstance(output, tuple) else output
            value = hash_tensor(hidden)
            if index == 0:
                pending[0] = value
            elif index == 1 and pending.get(0) == value:
                cross_equal += 1

        return hook

    handles = [
        model.model.layers[0].post_attention_layernorm.register_forward_hook(norm_hook),
        model.model.layers[0].register_forward_hook(layer_hook(0)),
    ]
    if len(model.model.layers) > 1:
        handles.append(model.model.layers[1].register_forward_hook(layer_hook(1)))
    try:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "dynamic_state_reuse_exact_duplicates": len(mlp_inputs) - len(set(mlp_inputs)),
        "dynamic_state_samples": len(mlp_inputs),
        "cross_layer_exact_intermediate_matches": cross_equal,
        "probe_uses_actual_checkpoint": True,
    }


def compile_named_mechanism(name: str, candidate: Any, artifact_dir: Path) -> Any:
    if name == "cold_packed_projection_sequential_materialization":
        return compile_checkpoint(candidate, artifact_dir=artifact_dir, layer_index=0)
    if name == "checkpoint_mlp_output_row_streamed_lossless_existing_isa":
        return compile_checkpoint_row_streamed(
            candidate,
            artifact_dir=artifact_dir,
            layer_index=0,
            output_tile_rows=DEFAULT_OUTPUT_TILE_ROWS,
        )
    if name == "checkpoint_mlp_torchinductor_existing_isa":
        return compile_checkpoint_inductor(candidate, artifact_dir=artifact_dir, layer_index=0)
    raise RuntimeError(f"unknown mechanism: {name}")


def run_mechanism(
    name: str,
    reference: Any,
    tokenizer: Any,
    workload: dict[str, Any],
    root: Path,
    official_tokens: list[int],
) -> dict[str, Any]:
    candidate, _, load_ns = load_dev()
    artifact_dir = root / name
    t0 = time.perf_counter_ns()
    artifact = compile_named_mechanism(name, candidate, artifact_dir)
    compile_outer = time.perf_counter_ns() - t0
    reference_artifact = make_reference_artifact(reference)
    runs: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    for index, spec in enumerate(workload["workloads"]):
        prompt, selection = choose_prompt(reference, tokenizer, spec)
        selections[spec["id"]] = selection if selection else None
        run = transition(
            reference_artifact,
            artifact,
            tokenizer,
            prompt,
            spec,
            int(workload["decode_steps"]),
            official_tokens if index == 0 else None,
        )
        run["id"] = spec["id"]
        run["category"] = spec["category"]
        runs.append(run)
        if not run["exact_transition_pass"]:
            break
    refresh_artifact_bytes(artifact)
    baseline = [value for run in runs for value in run["reference_latency_ns"]]
    candidate_latency = [value for run in runs for value in run["candidate_latency_ns"]]
    records = getattr(getattr(artifact.runtime_layer, "mlp", None), "records", {})
    peak_hot = max((record.raw_bytes for record in records.values()), default=0)
    gate = physical_gate(
        baseline_p50_ns=percentile_ns(baseline, 0.5),
        baseline_p95_ns=percentile_ns(baseline, 0.95),
        candidate_p50_ns=percentile_ns(candidate_latency, 0.5),
        candidate_p95_ns=percentile_ns(candidate_latency, 0.95),
        reference_layer_parameter_bytes=artifact.replaced_layer_reference_parameter_bytes,
        artifact_bytes=artifact.artifact_bytes,
        candidate_layer_resident_parameter_bytes=artifact.compiled_layer_resident_parameter_bytes,
        peak_projection_hot_bytes=peak_hot,
    )
    exact = len(runs) == len(workload["workloads"]) and all(run["exact_transition_pass"] for run in runs)
    full_layer = artifact.layer_index == 0 and artifact.runtime_layer is not None
    runtime_mlp = getattr(artifact.runtime_layer, "mlp", None)
    return {
        "mechanism": name,
        "candidate_load_ns": load_ns,
        "compile_wall_ns": compile_outer,
        "compile_peak_rss_bytes": artifact.compile_peak_rss_bytes,
        "artifact_bytes": artifact.artifact_bytes,
        "manifest_sha256": artifact.manifest_sha256,
        "checkpoint_tensor_sha256": artifact.checkpoint_tensor_sha256,
        "config_sha256": artifact.config_sha256,
        "reference_layer_parameter_bytes": artifact.replaced_layer_reference_parameter_bytes,
        "compiled_layer_resident_parameter_bytes": artifact.compiled_layer_resident_parameter_bytes,
        "peak_projection_hot_bytes": peak_hot,
        "output_tile_rows": getattr(runtime_mlp, "output_tile_rows", None),
        "runs": runs,
        "prompt_selection_evidence": selections,
        "G2_exact_transition": exact,
        "G3_full_transformer_layer_boundary": full_layer,
        "G4_physical_saving": bool(exact and full_layer and gate["passed"]),
        "physical_gate": gate,
        "isa_lowering": "existing host ISA via PyTorch eager output-row streaming/materialization or TorchInductor",
        "isa_instruction_count": None,
        "isa_instruction_status": "NOT_MEASURED_NO_PERF_EVENT",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workload", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    workload_path = Path(args.workload)
    workload = json.loads(workload_path.read_text())
    steps = int(workload.get("decode_steps", 0))
    if not workload.get("frozen_before_results") or steps < 128:
        raise RuntimeError("workload is not pre-frozen at >=128 steps")

    verify_pins()
    dev_identity = hf_identity(DEV_MODEL_ID, DEV_REVISION, True)
    target_identity = hf_identity(TARGET_MODEL_ID, TARGET_REVISION, False)
    reference, tokenizer, load_ns = load_dev()
    first_prompt = render(workload["workloads"][0])
    official = official_128(reference, tokenizer, first_prompt, steps)
    with torch.inference_mode():
        forward = reference(
            input_ids=tokenizer(first_prompt, return_tensors="pt")["input_ids"],
            use_cache=True,
            return_dict=True,
        )
    logits_raw = forward.logits.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    result: dict[str, Any] = {
        "schema": "fixed-public-dynamic-executor-result-v1",
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
        "reference_load_ns": load_ns,
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
            "logits_sha256": hashlib.sha256(logits_raw).hexdigest(),
        },
        "official_reference_generation": {"steps": len(official), "tokens": official},
        "actual_tensor_audit": audit_checkpoint_layer(reference.model.layers[0], layer_index=0),
        "actual_dynamic_reuse_audit": reuse_probe(reference, tokenizer, first_prompt),
        "row_streaming_preregistration": {
            "output_tile_rows": DEFAULT_OUTPUT_TILE_ROWS,
            "stream_axis": "output_rows",
            "reduction_axis_partitioned": False,
            "selection_basis": "prior hosted G4 footprint failure; fixed before current result",
        },
    }

    mechanisms = [
        run_mechanism(
            "cold_packed_projection_sequential_materialization",
            reference,
            tokenizer,
            workload,
            out / "artifacts",
            official,
        )
    ]
    if not any(mechanism["G4_physical_saving"] for mechanism in mechanisms):
        mechanisms.append(
            run_mechanism(
                "checkpoint_mlp_output_row_streamed_lossless_existing_isa",
                reference,
                tokenizer,
                workload,
                out / "artifacts",
                official,
            )
        )
    if not any(mechanism["G4_physical_saving"] for mechanism in mechanisms):
        mechanisms.append(
            run_mechanism(
                "checkpoint_mlp_torchinductor_existing_isa",
                reference,
                tokenizer,
                workload,
                out / "artifacts",
                official,
            )
        )

    result["mechanisms"] = mechanisms
    strongest = max(
        mechanisms,
        key=lambda mechanism: (
            int(mechanism["G2_exact_transition"]),
            int(mechanism["G3_full_transformer_layer_boundary"]),
            int(mechanism["G4_physical_saving"]),
        ),
    )
    result["strongest_mechanism"] = strongest["mechanism"]
    result["G1_actual_runtime"] = True
    result["verdict"] = (
        "REAL_CHECKPOINT_DYNAMIC_EXECUTOR_SURVIVES_G4"
        if strongest["G4_physical_saving"]
        else "REAL_CHECKPOINT_MECHANISM_CLASS_REJECTED_AT_G4"
        if strongest["G2_exact_transition"] and strongest["G3_full_transformer_layer_boundary"]
        else "REAL_CHECKPOINT_MECHANISM_CLASS_REJECTED_AT_G2"
    )
    result["full_model_path"] = {
        "formula": "TOTAL = non_MLP_resident + SUM_l[exact_artifact_l + peak_hot_l] + KV + runtime + temporaries + allocator_reserve",
        "latency_formula": "T_token = SUM_l(T_attention_l + T_compiled_MLP_l + T_norm_residual_l) + T_lm_head",
        "TARGET-W_terms": "NOT_EVALUATED_WITHOUT_ACTUAL_GATED_TENSORS",
        "four_b_budget_path": "NOT_ESTABLISHED_BY_DEV_W",
        "eight_gib_ledger": "NOT_ESTABLISHED_FOR_TARGET_W",
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "mechanisms": [
                    {
                        key: mechanism[key]
                        for key in (
                            "mechanism",
                            "G2_exact_transition",
                            "G3_full_transformer_layer_boundary",
                            "G4_physical_saving",
                        )
                    }
                    for mechanism in mechanisms
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
