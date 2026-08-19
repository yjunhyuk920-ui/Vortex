from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_085a.run_experiment import (
    canonical_bytes, capture_states, load_model, sha256_bytes, sha256_file,
    verify_runtime, write_checksums,
)
from vortex_runtime.global_bf16_successive_refinement import (
    BF16_MANTISSA_BITS, P50_WHOLE_MODEL_FRACTION, P95_WHOLE_MODEL_FRACTION,
    aggregate_oracle_rows, compile_global_bf16_refinement, tensor_sha256,
)


def annotate(rows: list[dict[str, Any]], metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != len(metadata):
        raise RuntimeError("oracle/state population mismatch")
    return [{**row, **meta} for row, meta in zip(rows, metadata)]


def family_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        family: aggregate_oracle_rows([row for row in rows if row["family"] == family])
        for family in sorted({row["family"] for row in rows})
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["mantissa_bits"] != list(range(BF16_MANTISSA_BITS + 1)):
        raise RuntimeError("frozen mantissa sequence mismatch")
    if float(config["p50_whole_model_fraction"]) != P50_WHOLE_MODEL_FRACTION:
        raise RuntimeError("p50 fraction mismatch")
    if float(config["p95_whole_model_fraction"]) != P95_WHOLE_MODEL_FRACTION:
        raise RuntimeError("p95 fraction mismatch")

    runtime = verify_runtime()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model, tokenizer, load_ns = load_model(config)
    layer = model.model.layers[int(config["layer_index"])]
    start = time.perf_counter_ns()
    compiler = compile_global_bf16_refinement(layer.mlp)
    compile_ns = time.perf_counter_ns() - start

    activations: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for prompt in config["prompts"]:
        states, tokens = capture_states(
            model, tokenizer, prompt["prompt"],
            layer_index=int(config["layer_index"]),
            positions=int(config["decode_positions_per_prompt"]),
        )
        traces.append({
            "prompt_id": prompt["id"], "family": prompt["family"],
            "prompt_sha256": sha256_bytes(prompt["prompt"].encode()),
            "tokens": tokens, "state_count": len(states),
        })
        for position, state in enumerate(states):
            activation = state.squeeze(0).contiguous()
            activations.append(activation)
            metadata.append({
                "prompt_id": prompt["id"], "family": prompt["family"],
                "decode_position": position,
                "activation_sha256": tensor_sha256(activation),
                "activation_l2": float(torch.linalg.vector_norm(activation.float())),
                "future_generated_token_reads": 0,
            })

    start = time.perf_counter_ns()
    evaluated = compiler.evaluate(torch.stack(activations).to(torch.bfloat16))
    evaluation_ns = time.perf_counter_ns() - start
    global_rows = annotate(evaluated.global_rows, metadata)
    adaptive_rows = annotate(evaluated.row_adaptive_rows, metadata)
    global_aggregate = aggregate_oracle_rows(global_rows)
    adaptive_aggregate = aggregate_oracle_rows(adaptive_rows)
    families = family_aggregates(adaptive_rows)

    success = config["success"]
    controls_passed = (
        evaluated.controls["full_precision_control_passed"]
        and all(value == 0 for value in evaluated.controls["prefix_identity_weight_mismatches"].values())
    )
    exact_passed = adaptive_aggregate["all_exact"] is bool(success["row_adaptive_all_exact"])
    information_passed = (
        adaptive_aggregate["entropy_fraction_p50"] <= float(success["row_adaptive_entropy_fraction_p50_max"])
        and adaptive_aggregate["entropy_fraction_p95"] <= float(success["row_adaptive_entropy_fraction_p95_max"])
    )
    acceptance_passed = (
        adaptive_aggregate["minimum_perfect_acceptance_p50"] <= int(success["minimum_perfect_acceptance_p50_max"])
        and adaptive_aggregate["minimum_perfect_acceptance_p95"] <= int(success["minimum_perfect_acceptance_p95_max"])
    )
    if not controls_passed or not exact_passed:
        decision = "INVALID_GLOBAL_BF16_SUCCESSIVE_REFINEMENT_CONTROL_FAILURE"
    elif information_passed and acceptance_passed:
        decision = "PROMOTE_GLOBAL_BF16_SUCCESSIVE_REFINEMENT_TO_PACKED_KERNEL_GATE"
    else:
        decision = "REJECT_GLOBAL_BF16_MANTISSA_PREFIX_AS_COLD_QUERY_CORE"

    core = {
        "schema": config["schema"], "config_sha256": sha256_file(config_path),
        "checkpoint": {"model_id": config["model_id"], "revision": config["revision"]},
        "compiler_manifest": evaluated.manifest, "trace_rows": traces,
        "global_rows": global_rows, "row_adaptive_rows": adaptive_rows,
        "precision_grid": evaluated.precision_grid, "controls": evaluated.controls,
        "aggregates": {"global_uniform": global_aggregate, "row_adaptive": adaptive_aggregate, "families": families},
        "gates": {"controls_passed": controls_passed, "oracle_exact_passed": exact_passed,
                  "information_passed": information_passed, "acceptance_passed": acceptance_passed},
        "decision": decision,
    }
    core_hash = sha256_bytes(canonical_bytes(core))
    measured = {
        "activation_count": len(activations),
        "full_precision_control_passed": evaluated.controls["full_precision_control_passed"],
        "global_uniform_entropy_fraction_p50": global_aggregate["entropy_fraction_p50"],
        "global_uniform_entropy_fraction_p95": global_aggregate["entropy_fraction_p95"],
        "row_adaptive_entropy_fraction_p50": adaptive_aggregate["entropy_fraction_p50"],
        "row_adaptive_entropy_fraction_p95": adaptive_aggregate["entropy_fraction_p95"],
        "row_adaptive_minimum_perfect_acceptance_p50": adaptive_aggregate["minimum_perfect_acceptance_p50"],
        "row_adaptive_minimum_perfect_acceptance_p95": adaptive_aggregate["minimum_perfect_acceptance_p95"],
    }
    result = {
        "experiment": "EXP-086A", "name": "global_bf16_successive_refinement_gate",
        "evidence_level": "E1", "phase": ["A-structure", "B-reference", "C-small-real-checkpoint-observation"],
        "source_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNVERIFIED"), "runtime": runtime,
        "load_wall_ns": load_ns, "compile_wall_ns": compile_ns, "evaluation_wall_ns": evaluation_ns,
        "config_sha256": sha256_file(config_path), "compiler_manifest": evaluated.manifest,
        "controls": evaluated.controls, "aggregates": core["aggregates"], "gates": core["gates"],
        "authoritative_decision": decision, "deterministic_core_sha256": core_hash,
        "MEASURED": measured,
        "DERIVED": {
            "p50_whole_model_fraction": P50_WHOLE_MODEL_FRACTION,
            "p95_whole_model_fraction": P95_WHOLE_MODEL_FRACTION,
            "interpretation": "All non-MLP traffic, draft, selector, coding, addressing and physical kernel costs are free favorable grants.",
        },
        "claim_boundary": {
            "official_checkpoint_loaded": True, "actual_checkpoint_activations": True,
            "complete_real_mlp_replaced": False, "oracle_selector_deployable": False,
            "standard_isa_operation_saving": "NOT_ESTABLISHED", "candidate_successor_state": "NOT_TESTED",
            "405b_execution": "NOT_TESTED", "8gib_gpu": "NOT_TESTED", "physical_latency": "NOT_TESTED",
        },
        "unverified": ["cross-weight context coding", "sound residual-bit selector",
                       "packed bit-serial/fused decompression kernel",
                       "complete Transformer layer and successor state",
                       "405B, 8 GiB and same-machine 4B latency"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    raw = output_dir / "raw"; artifacts = output_dir / "artifacts"
    raw.mkdir(exist_ok=True); artifacts.mkdir(exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    for name, rows in (("global_rows.jsonl", global_rows), ("row_adaptive_rows.jsonl", adaptive_rows), ("precision_grid.jsonl", evaluated.precision_grid)):
        with (raw / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    (raw / "trace_rows.json").write_text(json.dumps(traces, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    (artifacts / "compiler_manifest.json").write_text(json.dumps(evaluated.manifest, indent=2, sort_keys=True) + "\n")
    (artifacts / "deterministic_core.json").write_bytes(canonical_bytes(core))
    write_checksums(output_dir, ["result.json", "raw/global_rows.jsonl", "raw/row_adaptive_rows.jsonl",
        "raw/precision_grid.jsonl", "raw/trace_rows.json", "artifacts/compiler_manifest.json", "artifacts/deterministic_core.json"])
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
        result = {"experiment": "EXP-086A", "source_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNVERIFIED"),
            "authoritative_decision": "INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION",
            "error": f"{type(exc).__name__}: {exc}",
            "claim_boundary": {"mechanism_scientifically_rejected": False, "official_checkpoint_result": "NOT_COMPLETED"}}
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        (args.output_dir / "checksums.sha256").write_text(f"{sha256_file(args.output_dir / 'result.json')}  result.json\n")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
