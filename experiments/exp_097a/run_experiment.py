from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import torch
import transformers

from experiments.exp_094a.run_experiment import (
    checkpoint_identity,
    load_model,
    prefill_case,
    run_fine_guessed_sweep,
    verify_runtime,
    write_json,
)
from experiments.exp_095a.run_experiment import run_old_coarse_bank
from experiments.exp_096a.run_experiment import (
    capture_delayed_target,
    certificate_compiler,
)
from vortex_runtime.causal_suffix_action_cache import (
    quantize_rowwise_symmetric_int4,
)
from vortex_runtime.exact_jacobi_fixed_point import build_seed, jacobi_input
from vortex_runtime.online_residual_affine_hull import tensor_sha256
from vortex_runtime.residual_margin_certificate import canonical_sha256


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "exp-097a-residual-bank-specificity-v1":
        raise RuntimeError("unexpected config schema")
    if not bool(config.get("frozen_before_results")):
        raise RuntimeError("configuration was not frozen")
    if int(config["block_length"]) != 128:
        raise RuntimeError("K must remain 128")
    if str(config["seed_mode"]) != "prompt_suffix_cycle_16":
        raise RuntimeError("seed mode changed")
    positions = [int(x) for x in config["sample_positions"]]
    if positions != sorted(set(positions)) or positions[0] < 0 or positions[-1] >= 128:
        raise RuntimeError("invalid sample positions")
    if len(positions) < 8:
        raise RuntimeError("specificity Gate requires at least eight positions per case")
    if float(config["certificate"]["required_unique_margin"]) != 2.0 ** -20:
        raise RuntimeError("margin changed")
    if config["certificate"]["coefficient_bound"] is not None:
        raise RuntimeError("specificity audit must preserve EXP-096A unbounded capacity language")
    cases = list(config["cases"])
    if len(cases) != 6:
        raise RuntimeError("expected frozen six-case population")
    if len({str(row["id"]) for row in cases}) != len(cases):
        raise RuntimeError("duplicate case ID")


def build_seed_bank(
    model: Any,
    tokenizer: Any,
    q4_head: Any,
    case: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    prefill = prefill_case(model, tokenizer, case)
    seed, seed_manifest = build_seed(
        str(config["seed_mode"]),
        prompt_ids=prefill.prompt_ids,
        boundary_token=prefill.boundary_token,
        block_length=int(config["block_length"]),
    )
    guessed_inputs = jacobi_input(prefill.boundary_token, seed)
    fine_old, fine_meta = run_fine_guessed_sweep(model, prefill, guessed_inputs)
    coarse_old, hidden_old, coarse_meta = run_old_coarse_bank(
        model, q4_head, prefill, guessed_inputs
    )
    bank = (fine_old.to(torch.float64) - coarse_old.to(torch.float64)).contiguous()
    completed_ns = max(int(fine_meta["completed_ns"]), int(coarse_meta["completed_ns"]))
    meta = {
        "case_id": prefill.case_id,
        "split": prefill.split,
        "family": prefill.family,
        "prompt_sha256": prefill.prompt_sha256,
        "seed_manifest": seed_manifest.to_dict(),
        "bank_sha256": tensor_sha256(bank),
        "bank_bytes": int(bank.numel() * bank.element_size()),
        "bank_completed_ns": completed_ns,
        "fine_sweep_target_information_used": int(fine_meta["target_information_used"]),
    }
    del fine_old, coarse_old, hidden_old
    gc.collect()
    return prefill, bank, meta


def solve_selected(
    bank: torch.Tensor,
    coarse_true: torch.Tensor,
    target_tokens: Sequence[int],
    positions: Sequence[int],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    compiler = certificate_compiler(bank, config)
    rows: list[dict[str, Any]] = []
    for position in positions:
        result = compiler.solve(
            coarse_true[int(position)],
            int(target_tokens[int(position)]),
            initial_score_sources=(coarse_true[int(position)],),
        )
        row = {"position": int(position), **result.summary()}
        rows.append(row)
    del compiler
    gc.collect()
    return rows


def population_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    certs = [row["certificate"] for row in rows if row["certificate"] is not None]
    certified = sum(row["status"] == "certified" for row in rows)

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in certs]

    def pct(items: Sequence[float], q: float) -> float | None:
        if not items:
            return None
        return float(np.percentile(np.asarray(items, dtype=np.float64), q, method="linear"))

    return {
        "total": len(rows),
        "certified": certified,
        "infeasible": sum(row["status"] == "infeasible" for row in rows),
        "certificate_fraction": certified / len(rows) if rows else 0.0,
        "support_p50": pct(values("coefficient_support"), 50),
        "support_p95": pct(values("coefficient_support"), 95),
        "l1_p50": pct(values("coefficient_l1"), 50),
        "l1_p95": pct(values("coefficient_l1"), 95),
        "linf_p50": pct(values("coefficient_linf"), 50),
        "linf_p95": pct(values("coefficient_linf"), 95),
        "minimum_certified_margin": min(values("minimum_certified_margin")) if certs else None,
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    runtime = verify_runtime()
    torch.manual_seed(0)
    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.set_num_interop_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_load_ns = load_model(config)
    q4_head = quantize_rowwise_symmetric_int4(
        model.lm_head.weight,
        minimum=int(config["q4_head"]["minimum"]),
        maximum=int(config["q4_head"]["maximum"]),
    )
    identity = checkpoint_identity(model, tokenizer, config)

    # Critical ordering control: materialize and hash every candidate residual bank
    # before any delayed true target continuation starts.
    bank_phase_started = time.perf_counter_ns()
    prefills: dict[str, Any] = {}
    banks: dict[str, torch.Tensor] = {}
    bank_meta: dict[str, dict[str, Any]] = {}
    case_order = [str(row["id"]) for row in config["cases"]]
    case_by_id = {str(row["id"]): row for row in config["cases"]}
    for case in config["cases"]:
        prefill, bank, meta = build_seed_bank(model, tokenizer, q4_head, case, config)
        prefills[prefill.case_id] = prefill
        banks[prefill.case_id] = bank
        bank_meta[prefill.case_id] = meta
        print(f"bank {prefill.case_id}: {meta['bank_sha256']}", flush=True)
    all_banks_completed_ns = time.perf_counter_ns()

    permutation_rng = np.random.default_rng(int(config["vocabulary_permutation_seed"]))
    vocabulary_size = int(next(iter(banks.values())).shape[1])
    permutation = permutation_rng.permutation(vocabulary_size).astype(np.int64)
    if np.array_equal(permutation, np.arange(vocabulary_size, dtype=np.int64)):
        raise RuntimeError("identity vocabulary permutation")
    perm_index = torch.from_numpy(permutation)

    positions = [int(x) for x in config["sample_positions"]]
    rows: list[dict[str, Any]] = []
    first_target_started_ns: int | None = None
    for case_index, case_id in enumerate(case_order):
        prefill = prefills[case_id]
        fine_true, coarse_true, hidden_true, target_tokens, target_meta = capture_delayed_target(
            model, q4_head, prefill, int(config["block_length"])
        )
        if first_target_started_ns is None:
            first_target_started_ns = int(target_meta["started_ns"])
        donor_id = case_order[(case_index + 1) % len(case_order)]

        variants = [
            ("same", banks[case_id], case_id),
            ("cross_prompt", banks[donor_id], donor_id),
            ("vocab_permuted", banks[case_id].index_select(1, perm_index), case_id),
        ]
        for variant, variant_bank, source_id in variants:
            solved = solve_selected(
                variant_bank, coarse_true, target_tokens, positions, config
            )
            rows.append(
                {
                    "case_id": case_id,
                    "split": str(case_by_id[case_id]["split"]),
                    "family": str(case_by_id[case_id]["family"]),
                    "variant": variant,
                    "bank_source_case": source_id,
                    "bank_sha256": tensor_sha256(variant_bank),
                    "positions": solved,
                    "summary": population_summary(solved),
                }
            )
            if variant == "vocab_permuted":
                del variant_bank
                gc.collect()
        del fine_true, coarse_true, hidden_true
        gc.collect()

    if first_target_started_ns is None:
        raise RuntimeError("no delayed target was executed")
    ordering_passed = all_banks_completed_ns <= first_target_started_ns

    aggregate: dict[str, dict[str, dict[str, Any]]] = {}
    for split in ("build", "holdout"):
        aggregate[split] = {}
        for variant in ("same", "cross_prompt", "vocab_permuted"):
            selected = [
                position
                for row in rows
                if row["split"] == split and row["variant"] == variant
                for position in row["positions"]
            ]
            aggregate[split][variant] = population_summary(selected)

    hold = aggregate["holdout"]
    same = hold["same"]
    ratio_limit = float(config["decision"]["control_support_ratio_rejection_limit"])

    def control_matches(control: Mapping[str, Any]) -> bool:
        if float(control["certificate_fraction"]) < float(same["certificate_fraction"]):
            return False
        if same["support_p95"] is None or control["support_p95"] is None:
            return False
        return float(control["support_p95"]) <= ratio_limit * float(same["support_p95"])

    matched_controls = [
        name
        for name in ("cross_prompt", "vocab_permuted")
        if control_matches(hold[name])
    ]
    same_full = float(same["certificate_fraction"]) == 1.0
    integrity_failures: list[str] = []
    if not ordering_passed:
        integrity_failures.append("target_started_before_all_banks_frozen")
    if any(int(meta["fine_sweep_target_information_used"]) != 0 for meta in bank_meta.values()):
        integrity_failures.append("seed_bank_target_information_leak")
    if identity["resolved_revision"] not in ("", config["revision"]):
        integrity_failures.append("resolved_checkpoint_revision")

    if integrity_failures:
        decision = "INVALID_RESIDUAL_BANK_SPECIFICITY_CONTROL_FAILURE"
    elif matched_controls:
        decision = "REJECT_UNBOUNDED_RESIDUAL_MARGIN_CAPACITY_AS_MODEL_SPECIFIC_SOURCE"
    elif same_full:
        decision = "SURVIVES_RESIDUAL_BANK_SPECIFICITY_REQUIRES_FINITE_CODEBOOK_GATE"
    else:
        decision = "REJECT_RESIDUAL_BANK_SPECIFICITY_SOURCE"

    deterministic_core = {
        "schema": config["schema"],
        "config_sha256": canonical_sha256(config),
        "checkpoint": identity,
        "sample_positions": positions,
        "vocabulary_permutation_seed": int(config["vocabulary_permutation_seed"]),
        "vocabulary_permutation_sha256": canonical_sha256(permutation.tolist()),
        "all_banks_frozen_before_any_target": ordering_passed,
        "bank_meta": bank_meta,
        "aggregate": aggregate,
        "matched_controls": matched_controls,
        "integrity_failures": integrity_failures,
        "authoritative_decision": decision,
    }
    result = {
        **deterministic_core,
        "deterministic_core_sha256": canonical_sha256(deterministic_core),
        "case_rows": rows,
        "runtime": {
            **runtime,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "timing": {
            "model_load_ns": model_load_ns,
            "bank_phase_started_ns": bank_phase_started,
            "all_banks_completed_ns": all_banks_completed_ns,
            "first_target_started_ns": first_target_started_ns,
        },
        "claim_boundary": {
            "causal_coefficient_generator": "NOT_CONSTRUCTED",
            "fine_dense_arithmetic_reduction": "NOT_TESTED",
            "target_405b": "NOT_TESTED",
            "physical_8gib": "NOT_TESTED",
        },
    }
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "raw/case_rows.json", rows)
    write_json(output_dir / "artifacts/deterministic_core.json", deterministic_core)
    print(
        json.dumps(
            {
                "authoritative_decision": decision,
                "holdout_same": hold["same"],
                "holdout_cross_prompt": hold["cross_prompt"],
                "holdout_vocab_permuted": hold["vocab_permuted"],
                "matched_controls": matched_controls,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    execute(args.config, args.output_dir)


if __name__ == "__main__":
    main()
