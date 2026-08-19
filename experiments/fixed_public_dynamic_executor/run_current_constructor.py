from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fixed_public_dynamic_executor import run_real_checkpoint as harness
from vortex_runtime.fixed_public_dynamic_common import (
    DEV_MODEL_ID,
    DEV_REVISION,
    TARGET_MODEL_ID,
    TARGET_REVISION,
    environment_manifest,
    module_parameter_bytes,
)
from vortex_runtime.fixed_public_dynamic_audit import audit_checkpoint_layer
from vortex_runtime.fixed_public_dynamic_streaming import DEFAULT_OUTPUT_TILE_ROWS


CURRENT_MECHANISM = "checkpoint_mlp_output_row_streamed_lossless_existing_isa"


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

    harness.verify_pins()
    dev_identity = harness.hf_identity(DEV_MODEL_ID, DEV_REVISION, True)
    target_identity = harness.hf_identity(TARGET_MODEL_ID, TARGET_REVISION, False)
    reference, tokenizer, load_ns = harness.load_dev()
    first_prompt = harness.render(workload["workloads"][0])
    official = harness.official_128(reference, tokenizer, first_prompt, steps)
    with torch.inference_mode():
        forward = reference(
            input_ids=tokenizer(first_prompt, return_tensors="pt")["input_ids"],
            use_cache=True,
            return_dict=True,
        )
    logits_raw = forward.logits.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()

    mechanism = harness.run_mechanism(
        CURRENT_MECHANISM,
        reference,
        tokenizer,
        workload,
        out / "artifacts",
        official,
    )
    result: dict[str, Any] = {
        "schema": "fixed-public-dynamic-executor-result-v1",
        "github": {
            "repository": os.getenv("GITHUB_REPOSITORY"),
            "sha": os.getenv("GITHUB_SHA"),
            "ref": os.getenv("GITHUB_REF"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
        },
        "packages": harness.packages(),
        "environment": environment_manifest(),
        "workload_sha256": harness.digest_file(workload_path),
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
        "actual_dynamic_reuse_audit": harness.reuse_probe(reference, tokenizer, first_prompt),
        "row_streaming_preregistration": {
            "output_tile_rows": DEFAULT_OUTPUT_TILE_ROWS,
            "stream_axis": "output_rows",
            "reduction_axis_partitioned": False,
            "selection_basis": "prior hosted G4 footprint failure; fixed before current result",
        },
        "mechanisms": [mechanism],
        "strongest_mechanism": mechanism["mechanism"],
        "G1_actual_runtime": True,
        "full_model_path": {
            "formula": "TOTAL = non_MLP_resident + SUM_l[exact_artifact_l + peak_hot_l] + KV + runtime + temporaries + allocator_reserve",
            "latency_formula": "T_token = SUM_l(T_attention_l + T_compiled_MLP_l + T_norm_residual_l) + T_lm_head",
            "TARGET-W_terms": "NOT_EVALUATED_WITHOUT_ACTUAL_GATED_TENSORS",
            "four_b_budget_path": "NOT_ESTABLISHED_BY_DEV_W",
            "eight_gib_ledger": "NOT_ESTABLISHED_FOR_TARGET_W",
        },
    }
    result["verdict"] = (
        "REAL_CHECKPOINT_DYNAMIC_EXECUTOR_SURVIVES_G4"
        if mechanism["G4_physical_saving"]
        else "REAL_CHECKPOINT_MECHANISM_CLASS_REJECTED_AT_G4"
        if mechanism["G2_exact_transition"] and mechanism["G3_full_transformer_layer_boundary"]
        else "REAL_CHECKPOINT_MECHANISM_CLASS_REJECTED_AT_G2"
    )
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
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
