#!/usr/bin/env python3
"""Run EXP-068 oracle global-demand necessary-condition Gate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import time
from typing import Any, Sequence

import numpy as np

from experiments.exp_050.run_experiment import (
    MODEL_PATTERNS,
    TOKENIZER_PATTERNS,
    resolve_snapshot,
)
from experiments.exp_059.run_experiment import (
    dump,
    dump_rows,
    git_commit,
    sha256_bytes,
    sha256_file,
    write_checksums,
)
from vortex_runtime.output_head_demand import (
    analyze_output_head_demand_lower_bound,
    exact_minimum_tile_count_bruteforce,
    subset_certifies_winner,
)

ROOT = Path(__file__).resolve().parents[2]
EXP058_ROWS = ROOT / "results/exp_058/raw/matrix_rank_rows.jsonl"


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
    return rows


def exp058_index() -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(EXP058_ROWS):
        key = (str(row["model_id"]), str(row["tensor_name"]))
        if key in indexed:
            raise ValueError(f"duplicate EXP-058 row: {key}")
        indexed[key] = row
    return indexed


def control_population(seed: int) -> tuple[list[dict[str, Any]], int, int]:
    controls: list[dict[str, Any]] = []
    failures = 0
    violations = 0

    sparse_weight = np.asarray(
        [[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    hidden = np.ones(4, dtype=np.float32)
    sparse = analyze_output_head_demand_lower_bound(sparse_weight, hidden)
    sparse_exact = exact_minimum_tile_count_bruteforce(sparse_weight, hidden)
    passed = sparse.necessary_tile_count == 1 and sparse_exact == 1
    failures += int(not passed)
    violations += int(sparse.necessary_tile_count > sparse_exact)
    controls.append(
        {
            "control": "large_margin_sparse_influence",
            "passed": passed,
            "exact_minimum_tile_count": sparse_exact,
            **sparse.as_dict(),
        }
    )

    late_weight = np.asarray(
        [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.9]],
        dtype=np.float32,
    )
    late = analyze_output_head_demand_lower_bound(late_weight, hidden)
    late_exact = exact_minimum_tile_count_bruteforce(late_weight, hidden)
    passed = (
        late.necessary_tile_count == 3
        and late_exact == 3
        and not subset_certifies_winner(late_weight, hidden, 0, (0, 1))
        and subset_certifies_winner(late_weight, hidden, 0, (0, 1, 2))
    )
    failures += int(not passed)
    violations += int(late.necessary_tile_count > late_exact)
    controls.append(
        {
            "control": "late_flip_and_unread_decisive_tile",
            "passed": passed,
            "exact_minimum_tile_count": late_exact,
            **late.as_dict(),
        }
    )

    rng = np.random.default_rng(seed)
    random_passes = 0
    for index in range(16):
        weight = rng.normal(size=(7, 6)).astype(np.float32)
        vector = rng.normal(size=6).astype(np.float32)
        result = analyze_output_head_demand_lower_bound(weight, vector)
        exact = exact_minimum_tile_count_bruteforce(weight, vector)
        valid = result.necessary_tile_count <= exact
        random_passes += int(valid)
        violations += int(not valid)
        controls.append(
            {
                "control": "bounded_random_exhaustive",
                "case_index": index,
                "passed": valid,
                "exact_minimum_tile_count": exact,
                **result.as_dict(),
            }
        )
    passed = random_passes == 16
    failures += int(not passed)

    try:
        analyze_output_head_demand_lower_bound(
            np.asarray([[1.0, np.nan], [0.0, 1.0]], dtype=np.float32),
            np.ones(2, dtype=np.float32),
        )
        nonfinite_passed = False
    except ValueError:
        nonfinite_passed = True
    failures += int(not nonfinite_passed)
    controls.append(
        {
            "control": "nonfinite_fail_closed",
            "passed": nonfinite_passed,
        }
    )
    return controls, failures, violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/exp_068/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_068_candidate",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache/exp_068_huggingface",
    )
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output = arguments.output_dir
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    arguments.cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(config["torch_num_threads"]))
    os.environ.setdefault("MKL_NUM_THREADS", str(config["torch_num_threads"]))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(int(config["torch_num_threads"]))
    rank_index = exp058_index()
    baseline_roles = set(str(value) for value in config["baseline_matrix_roles"])
    started = time.perf_counter_ns()

    tokenizer_entry = config["tokenizer"]
    tokenizer_snapshot, tokenizer_manifest = resolve_snapshot(
        model_id=str(tokenizer_entry["model_id"]),
        revision=str(tokenizer_entry["revision"]),
        cache_dir=arguments.cache_dir,
        allow_patterns=TOKENIZER_PATTERNS,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_snapshot, local_files_only=True
    )

    case_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    tensor_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = [
        {
            "kind": "tokenizer",
            "model_id": str(tokenizer_entry["model_id"]),
            "revision": str(tokenizer_entry["revision"]),
            "files": tokenizer_manifest,
        }
    ]
    source_hash_mismatches = 0
    missing_source_evidence = 0
    reference_mismatches = 0
    domain_bound_violations = 0

    for model_entry in config["models"]:
        model_id = str(model_entry["model_id"])
        revision = str(model_entry["revision"])
        snapshot, manifest = resolve_snapshot(
            model_id=model_id,
            revision=revision,
            cache_dir=arguments.cache_dir,
            allow_patterns=MODEL_PATTERNS,
        )
        snapshot_rows.append(
            {
                "kind": "model",
                "model_id": model_id,
                "revision": revision,
                "files": manifest,
            }
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        state = model.state_dict()

        local_tensor_count = 0
        for tensor_name, tensor in sorted(state.items()):
            if tensor.ndim != 2:
                continue
            local_tensor_count += 1
            floating = (
                tensor.detach()
                .cpu()
                .contiguous()
                .numpy()
                .astype(np.float32, copy=False)
            )
            digest = sha256_bytes(floating.tobytes())
            source = rank_index.get((model_id, tensor_name))
            if source is None:
                missing_source_evidence += 1
                expected_digest = None
            else:
                expected_digest = str(source["source_tensor_sha256"])
                source_hash_mismatches += int(digest != expected_digest)
            tensor_rows.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "tensor_name": tensor_name,
                    "shape": list(floating.shape),
                    "source_tensor_sha256": digest,
                    "expected_source_tensor_sha256": expected_digest,
                    "source_hash_match": digest == expected_digest,
                }
            )

        baseline_dense_slots = sum(
            int(row["shape"][0]) * int(row["shape"][1])
            for (source_model, _), row in rank_index.items()
            if source_model == model_id
            and str(row["matrix_role"]) in baseline_roles
        )
        output_head = model.get_output_embeddings()
        if output_head is None:
            raise RuntimeError(f"missing output head for {model_id}")
        head_weight_tensor = output_head.weight.detach().cpu().contiguous()
        head_weight = head_weight_tensor.numpy().astype(np.float32, copy=False)
        head_bias = (
            None
            if getattr(output_head, "bias", None) is None
            else output_head.bias.detach().cpu().contiguous().numpy().astype(
                np.float32, copy=False
            )
        )
        local_cases: list[dict[str, Any]] = []

        for prompt in config["held_out_prompts"]:
            family = str(prompt["family"])
            text = str(prompt["text"])
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=int(config["max_input_tokens"]),
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"]
            if int(input_ids.shape[1]) == 0:
                raise ValueError(f"empty tokenization for family {family}")
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=encoded.get("attention_mask"),
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                reference_logits = outputs.logits[0, -1, :]
                hidden = outputs.hidden_states[-1][0, -1, :]
                replay_logits = output_head(hidden)
            reference_winner = int(torch.argmax(reference_logits).item())
            replay_winner = int(torch.argmax(replay_logits).item())
            reference_mismatches += int(reference_winner != replay_winner)
            top_two = torch.topk(reference_logits, k=2).values
            reference_margin = float((top_two[0] - top_two[1]).item())

            bound = analyze_output_head_demand_lower_bound(
                head_weight,
                hidden.detach().cpu().contiguous().numpy().astype(
                    np.float32, copy=False
                ),
                bias=head_bias,
                tile_columns=int(config["tile_columns"]),
                expected_winner=reference_winner,
                competitor_chunk_rows=int(config["competitor_chunk_rows"]),
            )
            domain_bound_violations += int(
                bound.domain_winner_mismatch_count
            )
            necessary_entries = int(
                bound.necessary_competitor_weight_entries
            )
            global_fraction = necessary_entries / baseline_dense_slots
            row = {
                "model_id": model_id,
                "revision": revision,
                "family": family,
                "prompt": text,
                "input_token_count": int(input_ids.shape[1]),
                "reference_winner": reference_winner,
                "replayed_head_winner": replay_winner,
                "reference_head_replay_match": (
                    reference_winner == replay_winner
                ),
                "reference_margin": reference_margin,
                "winner_token_text": tokenizer.decode([reference_winner]),
                "baseline_dense_weight_slots": baseline_dense_slots,
                "necessary_head_competitor_weight_entries": necessary_entries,
                "favorable_global_weight_byte_fraction_lower_bound": (
                    global_fraction
                ),
                "favorable_global_operation_fraction_lower_bound": (
                    global_fraction
                ),
                "preceding_transformer_work_charged": 0,
                "winner_head_row_charged": False,
                "bound_metadata_charged": False,
                **bound.as_dict(),
            }
            case_rows.append(row)
            local_cases.append(row)

        model_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "two_dimensional_tensor_count": local_tensor_count,
                "prompt_case_count": len(local_cases),
                "baseline_dense_weight_slots": baseline_dense_slots,
                "p50_weight_fraction_lower_bound": percentile(
                    [
                        row[
                            "favorable_global_weight_byte_fraction_lower_bound"
                        ]
                        for row in local_cases
                    ],
                    0.50,
                ),
                "p90_weight_fraction_lower_bound": percentile(
                    [
                        row[
                            "favorable_global_weight_byte_fraction_lower_bound"
                        ]
                        for row in local_cases
                    ],
                    0.90,
                ),
                "p50_head_weight_fraction_lower_bound": percentile(
                    [row["head_weight_fraction_lower_bound"] for row in local_cases],
                    0.50,
                ),
            }
        )
        del model, state, output_head, head_weight_tensor, head_weight

    control_rows, control_failures, synthetic_bound_violations = (
        control_population(int(config["seed"]))
    )
    domain_bound_violations += synthetic_bound_violations

    fractions = [
        float(row["favorable_global_weight_byte_fraction_lower_bound"])
        for row in case_rows
    ]
    operations = [
        float(row["favorable_global_operation_fraction_lower_bound"])
        for row in case_rows
    ]
    family_values: dict[str, list[float]] = defaultdict(list)
    for row in case_rows:
        family_values[str(row["family"])].append(
            float(row["favorable_global_weight_byte_fraction_lower_bound"])
        )
    family_p90 = {
        family: percentile(values, 0.90)
        for family, values in sorted(family_values.items())
    }
    model_p50 = {
        str(row["model_id"]): float(row["p50_weight_fraction_lower_bound"])
        for row in model_rows
    }
    best_model = min(model_p50.values())
    largest_id = str(config["models"][-1]["model_id"])
    largest_degradation = (
        model_p50[largest_id] / best_model - 1.0
        if best_model > 0.0
        else math.inf
    )

    gate = config["gate"]
    correctness_gate = (
        source_hash_mismatches == 0
        and missing_source_evidence == 0
        and reference_mismatches
        <= int(gate["maximum_reference_mismatches"])
        and domain_bound_violations
        <= int(gate["maximum_bound_violations"])
        and control_failures == 0
    )
    population_gate = (
        len(model_rows) == int(gate["expected_model_count"])
        and len(case_rows) == int(gate["expected_prompt_case_count"])
        and len(tensor_rows) == int(gate["expected_two_dimensional_tensors"])
        and len(family_values) == int(gate["expected_family_count"])
    )
    byte_gate = (
        percentile(fractions, 0.50)
        <= float(gate["maximum_p50_weight_byte_fraction"])
        and percentile(fractions, 0.90)
        <= float(gate["maximum_p90_weight_byte_fraction"])
    )
    operation_gate = (
        percentile(operations, 0.50)
        <= float(gate["maximum_p50_operation_fraction"])
        and percentile(operations, 0.90)
        <= float(gate["maximum_p90_operation_fraction"])
    )
    family_gate = all(
        value <= float(gate["maximum_family_p90_fraction"])
        for value in family_p90.values()
    )
    trend_gate = largest_degradation <= float(
        gate["maximum_largest_model_degradation"]
    )
    survives = all(
        (
            correctness_gate,
            population_gate,
            byte_gate,
            operation_gate,
            family_gate,
            trend_gate,
        )
    )
    decision = (
        str(config["survival_decision"])
        if survives
        else str(config["failure_decision"])
    )
    p50_fraction = percentile(fractions, 0.50)
    p90_fraction = percentile(fractions, 0.90)
    failure_basis = None
    if not byte_gate:
        failure_basis = (
            "output-head competitor-entry lower bound exceeds the preregistered "
            f"global target: p50={p50_fraction:.12f}, "
            f"p90={p90_fraction:.12f}"
        )

    measured = {
        "model_count": len(model_rows),
        "prompt_case_count": len(case_rows),
        "family_count": len(family_values),
        "two_dimensional_tensor_count": len(tensor_rows),
        "source_hash_mismatches": source_hash_mismatches,
        "missing_source_evidence": missing_source_evidence,
        "reference_head_replay_mismatches": reference_mismatches,
        "bound_violations": domain_bound_violations,
        "control_failures": control_failures,
        "p50_weight_byte_fraction_lower_bound": p50_fraction,
        "p90_weight_byte_fraction_lower_bound": p90_fraction,
        "p50_operation_fraction_lower_bound": percentile(operations, 0.50),
        "p90_operation_fraction_lower_bound": percentile(operations, 0.90),
        "minimum_weight_fraction_lower_bound": min(fractions),
        "maximum_weight_fraction_lower_bound": max(fractions),
        "family_p90_weight_fraction_lower_bound": family_p90,
        "model_p50_weight_fraction_lower_bound": model_p50,
        "largest_model_degradation_fraction": largest_degradation,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_ns": time.perf_counter_ns() - started,
    }
    summary = {
        "experiment": "EXP-068",
        "name": "oracle_output_head_competitor_entry_demand_lower_bound_gate",
        "phase": ["A", "B", "C-small-model-oracle-bound"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "MEASURED": measured,
        "DERIVED": {
            "global_demand_certificate_survives_necessary_gate": survives,
            "decision": decision,
            "correctness_gate_pass": correctness_gate,
            "population_gate_pass": population_gate,
            "weight_byte_gate_pass": byte_gate,
            "operation_gate_pass": operation_gate,
            "family_coverage_gate_pass": family_gate,
            "model_trend_gate_pass": trend_gate,
            "failure_basis": failure_basis,
            "bound_family": (
                "exact output-head coordinate contributions with absolute unread "
                "winner-versus-competitor bounds"
            ),
            "favorable_free_grants": [
                "all preceding Transformer operations and weight reads",
                "the complete winning output-head row",
                "all bound and ordering metadata",
                "an independently optimal reveal order for every competitor",
            ],
            "logical_scope": (
                "failure is sufficient to close the registered norm/absolute-bound "
                "demand family because the output head alone exceeds the whole-model "
                "budget after every preceding layer is granted for free"
            ),
        },
        "UNVERIFIED": [
            "bitwise floating-point bound preservation",
            "deployable demand scheduler",
            "actual skipped Transformer operations",
            "physical lazy-execution kernel",
            "405B execution",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "exact_real_output_head_bound": "MEASURED",
            "full_network_demand_bound": "NOT REQUIRED IF NECESSARY GATE FAILS",
            "bitwise_floating_point_certificate": "NOT TESTED",
            "actual_operation_replacement": False,
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "target_hardware": "NOT TESTED",
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(arguments.config),
            "exp058_rows_sha256": sha256_file(EXP058_ROWS),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }

    dump_rows(output / "raw/case_rows.jsonl", case_rows)
    dump_rows(output / "raw/model_rows.jsonl", model_rows)
    dump_rows(output / "raw/tensor_rows.jsonl", tensor_rows)
    dump_rows(output / "raw/control_rows.jsonl", control_rows)
    dump_rows(output / "raw/snapshot_rows.jsonl", snapshot_rows)
    dump(output / "summary.json", summary)
    dump(output / "processed/aggregate.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(
        str(config["evidence_ceiling"]) + "\n", encoding="utf-8"
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not correctness_gate or not population_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
