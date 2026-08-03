from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.exact_future_behavior_dag import (
    build_exact_future_dag,
    evaluate_heldout_future_body,
    export_future_dag,
    load_experiment_045_paths,
    quotient_frontier_point,
    replay_future_dag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the minimal exact finite-horizon future-decision DAG "
            "from authoritative Experiment 045 traces."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "results/decision_index_compiler_gate.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/exact_future_behavior_dag_gate.json"
        ),
    )
    parser.add_argument(
        "--vm-output",
        type=Path,
        default=Path(
            "results/exact_future_behavior_dag_compact40.vtx"
        ),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(
            "results/exact_future_behavior_dag_manifest.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source, compiled_with_control, heldout = load_experiment_045_paths(
        args.source
    )
    unique_compiled = [
        path
        for path in compiled_with_control
        if path.split == "compiled"
    ]
    duplicate_controls = [
        path
        for path in compiled_with_control
        if path.split == "duplicate_control"
    ]
    horizons = [int(value) for value in source["grammar"]["horizons"]]
    maximum_horizon = int(source["grammar"]["maximum_horizon"])
    codebook_limit = int(source["token_codebook"]["limit"])

    unique_frontier = [
        quotient_frontier_point(
            unique_compiled,
            horizon=horizon,
            codebook_limit=codebook_limit,
        )
        for horizon in horizons
    ]
    controlled_frontier = [
        quotient_frontier_point(
            compiled_with_control,
            horizon=horizon,
            codebook_limit=codebook_limit,
        )
        for horizon in horizons
    ]

    began = time.perf_counter_ns()
    dag = build_exact_future_dag(
        compiled_with_control,
        horizon=maximum_horizon,
        codebook_limit=codebook_limit,
    )
    dag_build_ns = time.perf_counter_ns() - began

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.vm_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter_ns()
    build, manifest_bytes = export_future_dag(
        dag,
        vm_path=args.vm_output,
        manifest_path=args.manifest_output,
        metadata={
            "experiment": "046_exact_future_behavior_dag",
            "source_experiment": source["experiment"],
            "source_model": source["model"],
            "source_grammar_sha256": source["grammar"]["sha256"],
            "source_head": "5fb32b30ceda3e362da7b6ee9ed2dee0c93231e5",
        },
    )
    export_ns = time.perf_counter_ns() - began

    replay = replay_future_dag(
        dag,
        compiled_with_control,
        vm_path=args.vm_output,
    )
    heldout_body = evaluate_heldout_future_body(dag, heldout)

    unique_final = unique_frontier[-1]
    controlled_final = controlled_frontier[-1]
    duplicate_raw_records = len(duplicate_controls) * maximum_horizon
    duplicate_incremental_nodes = (
        controlled_final.quotient_nodes
        - unique_final.quotient_nodes
    )
    duplicate_savings = (
        duplicate_raw_records - duplicate_incremental_nodes
    )
    cross_distinct_prompt_savings = (
        unique_final.raw_path_records
        - unique_final.quotient_nodes
    )

    prompt_to_start = dict(zip(dag.prompt_order, dag.start_addresses))
    duplicate_reuse = []
    for duplicate in duplicate_controls:
        source_id = str(duplicate.duplicate_of)
        duplicate_reuse.append(
            {
                "duplicate_prompt_id": duplicate.prompt_id,
                "source_prompt_id": source_id,
                "same_start_address": (
                    prompt_to_start[duplicate.prompt_id]
                    == prompt_to_start[source_id]
                ),
                "start_address": prompt_to_start[duplicate.prompt_id],
            }
        )

    raw_prefix_vm_bytes = int(source["build_accounting"]["vm_file_bytes"])
    implementation_passes = bool(
        replay.all_exact
        and build.atomic_replace
        and build.temporary_file_removed
        and all(item["same_start_address"] for item in duplicate_reuse)
        and duplicate_incremental_nodes == 0
        and controlled_final.quotient_nodes == dag.node_count
    )
    graph_body_compresses = bool(
        cross_distinct_prompt_savings > 0
    )
    start_router_barrier = bool(
        implementation_passes
        and heldout_body.causal_start_router_coverage == 0.0
    )

    token_metadata = source["token_codebook"]
    token_lookup = {
        int(token_id): {
            "tokenizer_token": token,
            "decoded_text": text,
        }
        for token_id, token, text in zip(
            token_metadata["token_ids"],
            token_metadata["tokenizer_tokens"],
            token_metadata["decoded_text"],
        )
    }

    payload = {
        "experiment": "046_exact_future_behavior_dag",
        "evidence_level": (
            "E1/E2 future-aware exact finite-horizon graph quotient"
        ),
        "source": {
            "experiment": source["experiment"],
            "model": source["model"],
            "grammar_sha256": source["grammar"]["sha256"],
            "compiled_prompt_paths": len(unique_compiled),
            "duplicate_controls": len(duplicate_controls),
            "heldout_prompt_paths": len(heldout),
            "new_model_forward_calls": 0,
            "source_compiler_forward_calls": source[
                "build_accounting"
            ]["compiled_model_forward_calls"],
        },
        "frontier_without_duplicate_control": [
            point.to_dict() for point in unique_frontier
        ],
        "frontier_with_duplicate_control": [
            point.to_dict() for point in controlled_frontier
        ],
        "compression_attribution": {
            "raw_unique_prompt_records": unique_final.raw_path_records,
            "unique_prompt_quotient_nodes": unique_final.quotient_nodes,
            "cross_distinct_prompt_suffix_savings": (
                cross_distinct_prompt_savings
            ),
            "duplicate_raw_records": duplicate_raw_records,
            "duplicate_incremental_nodes": duplicate_incremental_nodes,
            "duplicate_prompt_savings": duplicate_savings,
            "full_continuation_equivalence_classes": (
                unique_final.unique_full_continuations
            ),
        },
        "dag": {
            "horizon": dag.horizon,
            "nodes": dag.node_count,
            "starts": len(dag.start_addresses),
            "token_codebook_size": len(dag.token_codebook),
            "token_codebook_ids": list(dag.token_codebook),
            "token_codebook_text": {
                str(token_id): token_lookup.get(token_id, {})
                for token_id in dag.token_codebook
            },
            "build_ns": dag_build_ns,
            "vm_export_ns": export_ns,
            "vm_file_bytes": build.file_bytes,
            "manifest_bytes": manifest_bytes,
            "source_raw_prefix_vm_bytes": raw_prefix_vm_bytes,
            "vm_byte_reduction": raw_prefix_vm_bytes - build.file_bytes,
            "vm_byte_reduction_fraction": (
                (raw_prefix_vm_bytes - build.file_bytes)
                / raw_prefix_vm_bytes
                if raw_prefix_vm_bytes
                else 0.0
            ),
        },
        "duplicate_control": duplicate_reuse,
        "compiled_replay": replay.to_dict(),
        "heldout": heldout_body.to_dict(),
        "scope_separation": {
            "complete_future_tokens_used_for_offline_quotient": True,
            "new_model_calls_used": False,
            "finite_horizon_exact_equivalence_proven": implementation_passes,
            "compiled_paths_replay_exactly": replay.all_exact,
            "graph_body_compression_proven": graph_body_compresses,
            "future_aware_heldout_body_oracle_deployable": False,
            "causal_heldout_start_router_proven": False,
            "arbitrary_prompt_coverage_proven": False,
            "real_405b_execution_performed": False,
        },
        "conclusion": {
            "quotient_implementation_passes": implementation_passes,
            "graph_body_compresses_across_distinct_prompts": (
                graph_body_compresses
            ),
            "start_router_barrier_present": start_router_barrier,
            "max_horizon_raw_records": unique_final.raw_path_records,
            "max_horizon_quotient_nodes": unique_final.quotient_nodes,
            "max_horizon_compression_fraction": (
                unique_final.compression_fraction
            ),
            "future_aware_heldout_suffix_coverage": (
                heldout_body.future_suffix_coverage
            ),
            "causal_heldout_start_coverage": (
                heldout_body.causal_start_router_coverage
            ),
            "decision": (
                "accept exact suffix-DAG compression and make the sound "
                "start router the primary next Gate"
                if implementation_passes and graph_body_compresses
                else (
                    "reject future-aware suffix DAG as a meaningful body "
                    "compression mechanism"
                    if implementation_passes
                    else "DAG Gate failed and requires repair"
                )
            ),
            "fixed_target_status": (
                "unsolved: offline future-aware compression does not route "
                "unseen prompts, compile 405B behavior, or establish target "
                "hardware performance"
            ),
        },
        "next_obligation": (
            "if the body compresses, construct a causal and exact-certified "
            "prompt-to-start router; future-token oracle routing is forbidden. "
            "If causal routing cannot exceed zero held-out coverage, the "
            "decision-index architecture remains bounded memoization"
        ),
    }

    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
