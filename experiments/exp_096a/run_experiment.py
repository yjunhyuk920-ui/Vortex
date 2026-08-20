from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy
import torch
import transformers
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_094a.run_experiment import (  # noqa: E402
    PrefillState,
    checkpoint_identity,
    fresh_first_layer_cache,
    fresh_full_cache,
    load_model,
    prefill_case,
    run_fine_guessed_sweep,
    sha256_bytes,
    sha256_file,
    verify_runtime,
    write_checksums,
    write_json,
)
from experiments.exp_095a.run_experiment import (  # noqa: E402
    run_old_coarse_bank,
    shallow_step_with_hidden,
)
from vortex_runtime.causal_suffix_action_cache import (  # noqa: E402
    RowwiseSymmetricQ4Head,
    project_target_hot_state,
    quantize_rowwise_symmetric_int4,
)
from vortex_runtime.exact_jacobi_fixed_point import (  # noqa: E402
    build_seed,
    jacobi_input,
)
from vortex_runtime.fixed_public_dynamic_common import (  # noqa: E402
    DEV_MODEL_ID,
    DEV_REVISION,
    TARGET_MODEL_ID,
    TARGET_REVISION,
    module_parameter_bytes,
)
from vortex_runtime.online_residual_affine_hull import (  # noqa: E402
    residual_span_projection,
    tensor_sha256,
)
from vortex_runtime.parareal_residual_transport import (  # noqa: E402
    logical_source_fraction,
    residual_buffer_bytes,
)
from vortex_runtime.prefix_state_bisimulation import (  # noqa: E402
    cache_nbytes,
    cache_sha256,
)
from vortex_runtime.residual_margin_certificate import (  # noqa: E402
    MarginSolveResult,
    ResidualMarginCompiler,
    array_sha256,
    canonical_sha256,
    model_free_controls,
    project_target_resources,
)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "exp-096a-residual-margin-certificate-v1":
        raise RuntimeError("unexpected config schema")
    if not config.get("frozen_before_results"):
        raise RuntimeError("configuration was not frozen before results")
    if config.get("model_id") != DEV_MODEL_ID or config.get("revision") != DEV_REVISION:
        raise RuntimeError("frozen DEV-W identity mismatch")
    if int(config["block_length"]) != 128 or int(config["shallow_depth"]) != 1:
        raise RuntimeError("EXP-096A freezes K=128 and one shallow layer")
    if str(config["seed_mode"]) != "prompt_suffix_cycle_16":
        raise RuntimeError("seed changed from frozen predecessor")
    q4 = config["q4_head"]
    if q4["scheme"] != "rowwise_symmetric_int4":
        raise RuntimeError("Q4 head scheme changed")
    if int(q4["minimum"]) != -8 or int(q4["maximum"]) != 7:
        raise RuntimeError("Q4 head range changed")

    cert = config["certificate"]
    if cert["solver"] != "scipy_linprog_highs_ds_active_set":
        raise RuntimeError("certificate solver changed")
    if cert["objective"] != "minimum_l1_float64_coefficient_vector":
        raise RuntimeError("certificate objective changed")
    if cert["coefficient_bound"] is not None:
        raise RuntimeError("EXP-096A grants an unbounded minimum-L1 coefficient search")
    if float(cert["required_unique_margin"]) != 2.0**-20:
        raise RuntimeError("required unique margin changed")
    if int(cert["initial_topk_per_score_source"]) != 64:
        raise RuntimeError("initial active-set width changed")
    if int(cert["violator_batch_size"]) != 64:
        raise RuntimeError("violator batch changed")
    if int(cert["maximum_active_iterations"]) != 24:
        raise RuntimeError("active iteration limit changed")
    if not bool(cert["full_constraint_fallback"]):
        raise RuntimeError("full constraint fallback disabled")
    if float(cert["primal_feasibility_tolerance"]) != 1e-9:
        raise RuntimeError("primal tolerance changed")
    if float(cert["dual_feasibility_tolerance"]) != 1e-9:
        raise RuntimeError("dual tolerance changed")

    target = config["target_projection"]
    if target["model_id"] != TARGET_MODEL_ID or target["revision"] != TARGET_REVISION:
        raise RuntimeError("TARGET-W identity mismatch")
    if float(config["success"]["required_certificate_fraction"]) != 1.0:
        raise RuntimeError("certificate coverage threshold changed")

    cases = list(config["cases"])
    build = [row for row in cases if row["split"] == "build"]
    holdout = [row for row in cases if row["split"] == "holdout"]
    success = config["success"]
    if len(build) < int(success["minimum_build_cases"]):
        raise RuntimeError("insufficient build cases")
    if len(holdout) < int(success["minimum_holdout_cases"]):
        raise RuntimeError("insufficient holdout cases")
    ids = [str(row["id"]) for row in cases]
    prompt_hashes = [sha256_bytes(row["prompt"].encode("utf-8")) for row in cases]
    if len(ids) != len(set(ids)) or len(prompt_hashes) != len(set(prompt_hashes)):
        raise RuntimeError("duplicate case ID or prompt")


def capture_delayed_target(
    model: Any,
    q4_head: RowwiseSymmetricQ4Head,
    prefill: PrefillState,
    block_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], dict[str, Any]]:
    target_cache: Any = fresh_full_cache(prefill.prefix_cache)
    coarse_cache = fresh_first_layer_cache(prefill.prefix_cache)
    current = int(prefill.boundary_token)
    fine_rows: list[torch.Tensor] = []
    coarse_rows: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []
    tokens: list[int] = []
    fine_hashes: list[str] = []
    started_ns = time.perf_counter_ns()
    for offset in range(int(block_length)):
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([[current]], dtype=torch.long),
                past_key_values=target_cache,
                use_cache=True,
                return_dict=True,
            )
        fine = output.logits[0, -1].detach().to(torch.float32).contiguous().cpu()
        token = int(torch.argmax(fine).item())
        coarse, hidden = shallow_step_with_hidden(
            model,
            q4_head,
            token=current,
            cache=coarse_cache,
            position=len(prefill.prompt_ids) + offset,
        )
        fine_rows.append(fine)
        coarse_rows.append(coarse)
        hidden_rows.append(hidden)
        tokens.append(token)
        fine_hashes.append(tensor_sha256(fine))
        current = token
        target_cache = output.past_key_values
    completed_ns = time.perf_counter_ns()
    fine_block = torch.stack(fine_rows, dim=0).contiguous()
    coarse_block = torch.stack(coarse_rows, dim=0).contiguous()
    hidden_block = torch.stack(hidden_rows, dim=0).contiguous()
    return fine_block, coarse_block, hidden_block, tuple(tokens), {
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "wall_ns": completed_ns - started_ns,
        "target_calls": len(tokens),
        "true_coarse_calls": len(tokens),
        "target_sha256": canonical_sha256(tokens),
        "fine_block_sha256": tensor_sha256(fine_block),
        "coarse_block_sha256": tensor_sha256(coarse_block),
        "hidden_block_sha256": tensor_sha256(hidden_block),
        "fine_row_hash_sha256": canonical_sha256(fine_hashes),
        "final_target_cache_sha256": cache_sha256(target_cache),
        "final_target_cache_bytes": cache_nbytes(target_cache),
        "final_coarse_cache_sha256": cache_sha256(coarse_cache),
        "final_coarse_cache_bytes": cache_nbytes(coarse_cache),
    }


def certificate_compiler(
    residual_bank: torch.Tensor, config: Mapping[str, Any]
) -> ResidualMarginCompiler:
    cert = config["certificate"]
    return ResidualMarginCompiler(
        residual_bank,
        required_unique_margin=float(cert["required_unique_margin"]),
        initial_topk_per_score_source=int(cert["initial_topk_per_score_source"]),
        violator_batch_size=int(cert["violator_batch_size"]),
        maximum_active_iterations=int(cert["maximum_active_iterations"]),
        full_constraint_fallback=bool(cert["full_constraint_fallback"]),
        primal_feasibility_tolerance=float(cert["primal_feasibility_tolerance"]),
        dual_feasibility_tolerance=float(cert["dual_feasibility_tolerance"]),
    )


def solve_case_certificates(
    *,
    compiler: ResidualMarginCompiler,
    coarse_true: torch.Tensor,
    target_tokens: Sequence[int],
    residual_bank: torch.Tensor,
    aligned_scores: torch.Tensor,
    span_scores: torch.Tensor,
    output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    block_length = len(target_tokens)
    coefficients = np.zeros((block_length, compiler.k), dtype=np.float64)
    status_codes = np.full(block_length, -1, dtype=np.int8)
    certified_margins = np.full(block_length, np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    started_ns = time.perf_counter_ns()
    decisive_infeasible_position: int | None = None

    for position, token_value in enumerate(target_tokens):
        result = compiler.solve(
            coarse_true[position],
            int(token_value),
            initial_score_sources=(
                coarse_true[position],
                aligned_scores[position],
                span_scores[position],
            ),
        )
        row = {"position": position, **result.summary()}
        rows.append(row)
        if result.status == "certified":
            if result.certificate is None:
                raise RuntimeError("certified result has no certificate")
            coefficients[position] = result.certificate.coefficients
            certified_margins[position] = (
                result.certificate.minimum_certified_margin
            )
            status_codes[position] = 1
        elif result.status == "infeasible":
            status_codes[position] = 0
            decisive_infeasible_position = position
        else:
            raise RuntimeError(f"unexpected certificate status: {result.status}")
        if position % 16 == 0 or result.status == "infeasible":
            print(
                f"  margin position={position} status={result.status} "
                f"active={result.active_constraint_count}",
                flush=True,
            )
        # A single explicit infeasible position is decisive for the frozen
        # exact-all-positions source. Preserve cheapest-kill-first behavior.
        if decisive_infeasible_position is not None:
            break

    completed_ns = time.perf_counter_ns()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        coefficients=coefficients,
        status_codes=status_codes,
        certified_margins=certified_margins,
        target_tokens=np.asarray(target_tokens, dtype=np.int32),
    )
    evaluated = len(rows)
    certified = sum(row["status"] == "certified" for row in rows)
    infeasible = sum(row["status"] == "infeasible" for row in rows)
    cert_rows = [row["certificate"] for row in rows if row["certificate"] is not None]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in cert_rows]

    def percentile(items: list[float], q: float) -> float | None:
        if not items:
            return None
        return float(np.percentile(items, q, method="linear"))

    summary = {
        "block_length": block_length,
        "evaluated_positions": evaluated,
        "certified_positions": certified,
        "infeasible_positions": infeasible,
        "not_evaluated_positions": block_length - evaluated,
        "certificate_fraction_of_block": certified / block_length,
        "decisive_infeasible_position": decisive_infeasible_position,
        "coefficient_l1_p50": percentile(values("coefficient_l1"), 50),
        "coefficient_l1_p95": percentile(values("coefficient_l1"), 95),
        "coefficient_linf_p50": percentile(values("coefficient_linf"), 50),
        "coefficient_linf_p95": percentile(values("coefficient_linf"), 95),
        "support_p50": percentile(values("coefficient_support"), 50),
        "support_p95": percentile(values("coefficient_support"), 95),
        "minimum_certified_margin": (
            min(values("minimum_certified_margin")) if cert_rows else None
        ),
        "active_constraint_p50": percentile(
            [float(row["active_constraint_count"]) for row in rows], 50
        ),
        "active_constraint_p95": percentile(
            [float(row["active_constraint_count"]) for row in rows], 95
        ),
        "full_fallback_count": sum(
            bool(row["full_constraint_fallback_used"]) for row in rows
        ),
        "coefficient_matrix_sha256": array_sha256(coefficients),
        "status_code_sha256": array_sha256(status_codes),
        "margin_sha256": array_sha256(certified_margins),
        "artifact_sha256": sha256_file(output_path),
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "wall_ns": completed_ns - started_ns,
    }
    return rows, summary


def run_case(
    model: Any,
    tokenizer: Any,
    q4_head: RowwiseSymmetricQ4Head,
    case: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    prefill = prefill_case(model, tokenizer, case)
    seed, seed_manifest = build_seed(
        str(config["seed_mode"]),
        prompt_ids=prefill.prompt_ids,
        boundary_token=prefill.boundary_token,
        block_length=int(config["block_length"]),
    )
    guessed_inputs = jacobi_input(prefill.boundary_token, seed)
    fine_old, fine_meta = run_fine_guessed_sweep(
        model, prefill, guessed_inputs
    )
    coarse_old, hidden_old, coarse_meta = run_old_coarse_bank(
        model, q4_head, prefill, guessed_inputs
    )
    residual_bank = (
        fine_old.to(torch.float64) - coarse_old.to(torch.float64)
    ).contiguous()
    bank_completed_ns = max(
        int(fine_meta["completed_ns"]), int(coarse_meta["completed_ns"])
    )

    fine_true, coarse_true, hidden_true, target_tokens, target_meta = (
        capture_delayed_target(
            model,
            q4_head,
            prefill,
            int(config["block_length"]),
        )
    )
    target_residual = (
        fine_true.to(torch.float64) - coarse_true.to(torch.float64)
    ).contiguous()
    span = residual_span_projection(
        residual_bank,
        target_residual,
        relative_eigenvalue_cutoff=2.0**-40,
    )
    span_scores = coarse_true.to(torch.float64) + span.projected_residual
    aligned_scores = coarse_true.to(torch.float64) + residual_bank
    compiler = certificate_compiler(residual_bank, config)
    relative_artifact = Path("artifacts/certificates") / f"{prefill.case_id}.npz"
    certificate_rows, certificate_summary = solve_case_certificates(
        compiler=compiler,
        coarse_true=coarse_true,
        target_tokens=target_tokens,
        residual_bank=residual_bank,
        aligned_scores=aligned_scores,
        span_scores=span_scores,
        output_path=output_dir / relative_artifact,
    )

    first_control = (
        bool(certificate_rows)
        and certificate_rows[0]["status"] == "certified"
        and certificate_rows[0]["target_token"] == int(target_tokens[0])
    )
    row = {
        "case_id": prefill.case_id,
        "split": prefill.split,
        "family": prefill.family,
        "prompt_sha256": prefill.prompt_sha256,
        "prompt_token_count": len(prefill.prompt_ids),
        "boundary_token": prefill.boundary_token,
        "prefix_cache_sha256": prefill.prefix_cache_sha256,
        "prefix_cache_bytes": prefill.prefix_cache_bytes,
        "seed_manifest": seed_manifest.to_dict(),
        "fine_sweep": fine_meta,
        "coarse_old": coarse_meta,
        "residual_bank_sha256": tensor_sha256(residual_bank),
        "residual_bank_bytes": int(
            residual_bank.numel() * residual_bank.element_size()
        ),
        "hidden_bank_sha256": tensor_sha256(hidden_old),
        "hidden_bank_bytes": int(hidden_old.numel() * hidden_old.element_size()),
        "online_bank_completed_before_target": bank_completed_ns
        <= int(target_meta["started_ns"]),
        "target": target_meta,
        "target_residual_sha256": tensor_sha256(target_residual),
        "span_projection": {
            **span.summary(),
            "projected_residual_sha256": tensor_sha256(span.projected_residual),
            "coefficient_sha256": tensor_sha256(span.coefficients),
        },
        "certificate_artifact": relative_artifact.as_posix(),
        "certificate_rows": certificate_rows,
        "certificate_summary": certificate_summary,
        "first_position_certificate_control": first_control,
    }
    del (
        fine_old,
        coarse_old,
        hidden_old,
        residual_bank,
        fine_true,
        coarse_true,
        hidden_true,
        target_residual,
        span_scores,
        aligned_scores,
        compiler,
    )
    gc.collect()
    return row


def aggregate(rows: Sequence[Mapping[str, Any]], split: str, block_length: int) -> dict[str, Any]:
    selected = [row for row in rows if str(row["split"]) == str(split)]
    if not selected:
        raise RuntimeError(f"empty aggregate: {split}")
    total_positions = len(selected) * int(block_length)
    evaluated = sum(
        int(row["certificate_summary"]["evaluated_positions"])
        for row in selected
    )
    certified = sum(
        int(row["certificate_summary"]["certified_positions"])
        for row in selected
    )
    infeasible = sum(
        int(row["certificate_summary"]["infeasible_positions"])
        for row in selected
    )
    certificate_rows = [
        cert
        for row in selected
        for position in row["certificate_rows"]
        for cert in ([position["certificate"]] if position["certificate"] else [])
    ]

    def metric(key: str) -> list[float]:
        return [float(row[key]) for row in certificate_rows]

    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        return float(np.percentile(values, q, method="linear"))

    return {
        "split": split,
        "case_count": len(selected),
        "total_positions": total_positions,
        "evaluated_positions": evaluated,
        "certified_positions": certified,
        "infeasible_positions": infeasible,
        "not_evaluated_positions": total_positions - evaluated,
        "certificate_fraction": certified / total_positions,
        "all_positions_certified": certified == total_positions,
        "decisive_infeasible_cases": [
            {
                "case_id": row["case_id"],
                "position": row["certificate_summary"][
                    "decisive_infeasible_position"
                ],
            }
            for row in selected
            if row["certificate_summary"]["decisive_infeasible_position"]
            is not None
        ],
        "coefficient_l1_p50": pct(metric("coefficient_l1"), 50),
        "coefficient_l1_p95": pct(metric("coefficient_l1"), 95),
        "coefficient_linf_p50": pct(metric("coefficient_linf"), 50),
        "coefficient_linf_p95": pct(metric("coefficient_linf"), 95),
        "support_p50": pct(metric("coefficient_support"), 50),
        "support_p95": pct(metric("coefficient_support"), 95),
        "minimum_certified_margin": (
            min(metric("minimum_certified_margin"))
            if certificate_rows
            else None
        ),
        "all_bank_before_target": all(
            bool(row["online_bank_completed_before_target"]) for row in selected
        ),
        "all_first_position_controls": all(
            bool(row["first_position_certificate_control"]) for row in selected
        ),
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    runtime = verify_runtime()
    controls = model_free_controls()
    if not controls["passed"]:
        raise RuntimeError("model-free margin-certificate controls failed")

    torch.manual_seed(0)
    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.set_num_interop_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_load_ns = load_model(config)
    q4_started_ns = time.perf_counter_ns()
    q4_head = quantize_rowwise_symmetric_int4(
        model.lm_head.weight,
        minimum=int(config["q4_head"]["minimum"]),
        maximum=int(config["q4_head"]["maximum"]),
    )
    q4_compile_ns = time.perf_counter_ns() - q4_started_ns
    identity = checkpoint_identity(model, tokenizer, config)
    write_json(output_dir / "artifacts/checkpoint_identity.json", identity)
    write_json(
        output_dir / "artifacts/q4_head_manifest.json",
        q4_head.manifest.to_dict(),
    )

    rows: list[dict[str, Any]] = []
    started_ns = time.perf_counter_ns()
    for case in config["cases"]:
        row = run_case(model, tokenizer, q4_head, case, config, output_dir)
        rows.append(row)
        summary = row["certificate_summary"]
        print(
            f"{row['split']} {row['case_id']}: "
            f"certified={summary['certified_positions']}/{summary['block_length']} "
            f"infeasible_at={summary['decisive_infeasible_position']}",
            flush=True,
        )
    experiment_ns = time.perf_counter_ns() - started_ns

    block_length = int(config["block_length"])
    build = aggregate(rows, "build", block_length)
    holdout = aggregate(rows, "holdout", block_length)
    integrity_failures: list[str] = []
    if not controls["passed"]:
        integrity_failures.append("model_free_control_failure")
    if identity["resolved_revision"] not in ("", config["revision"]):
        integrity_failures.append("resolved_checkpoint_revision")
    if not all(
        bool(row["online_bank_completed_before_target"]) for row in rows
    ):
        integrity_failures.append("online_bank_target_order_failure")
    if not all(bool(row["first_position_certificate_control"]) for row in rows):
        integrity_failures.append("first_position_certificate_failure")
    if not all(
        int(row["fine_sweep"]["target_information_used"]) == 0
        and int(row["target"]["target_calls"]) == block_length
        and int(row["target"]["true_coarse_calls"]) == block_length
        for row in rows
    ):
        integrity_failures.append("incomplete_or_leaked_call_trace")
    if not all(
        position["status"] in {"certified", "infeasible"}
        and (
            position["status"] != "infeasible"
            or int(position["solver_status"]) == 2
        )
        for row in rows
        for position in row["certificate_rows"]
    ):
        integrity_failures.append("unresolved_solver_status")
    integrity_passed = len(integrity_failures) <= int(
        config["success"]["maximum_integrity_mismatches"]
    )

    all_certified = (
        integrity_passed
        and bool(build["all_positions_certified"])
        and bool(holdout["all_positions_certified"])
    )
    explicit_infeasible = (
        int(build["infeasible_positions"]) + int(holdout["infeasible_positions"])
    ) > 0
    if not integrity_passed:
        decision = "INVALID_RESIDUAL_MARGIN_CERTIFICATE_CONTROL_FAILURE"
    elif all_certified:
        decision = "PROMOTE_RESIDUAL_MARGIN_CERTIFICATE_TO_CAUSAL_COEFFICIENT_GATE"
    elif explicit_infeasible:
        decision = "REJECT_BOUNDED_BLOCK_RESIDUAL_MARGIN_CERTIFICATE_AS_EXACT_SOURCE"
    else:
        decision = "INVALID_RESIDUAL_MARGIN_CERTIFICATE_CONTROL_FAILURE"

    target_base = project_target_hot_state(config["target_projection"])
    residual_bytes = residual_buffer_bytes(
        block_length=block_length,
        vocabulary_size=int(config["target_projection"]["vocab_size"]),
        value_bytes=8,
    )
    hidden_bytes = (
        block_length
        * int(config["target_projection"]["hidden_size"])
        * 8
    )
    gram_bytes = block_length * block_length * 8
    inherited_hot = (
        int(target_base.total_hot_bytes)
        + residual_bytes
        + hidden_bytes
        + gram_bytes
    )
    resources = project_target_resources(
        inherited_hot_bytes=inherited_hot,
        block_length=block_length,
        vocabulary_size=int(config["target_projection"]["vocab_size"]),
    )
    hot_limit = int(config["target_projection"]["hot_limit_bytes"])
    resources["hot_limit_bytes"] = hot_limit
    resources["hot_margin_bytes"] = hot_limit - int(resources["total_hot_bytes"])
    resources["hot_limit_passed"] = int(resources["total_hot_bytes"]) <= hot_limit
    compression = float(
        config["target_projection"]["checkpoint_compression_ratio_favorable"]
    )
    resources.update(
        {
            "one_fine_sweep_fraction_at_full_block_raw": 1.0 / block_length,
            "one_fine_sweep_fraction_at_full_block_compressed": 1.0
            / (block_length * compression),
            "fine_dense_arithmetic_fraction_per_token": 1.0,
            "causal_coefficient_generator": "NOT_CONSTRUCTED",
        }
    )

    deterministic_core = {
        "schema": config["schema"],
        "mechanism_fingerprint": (
            "one-official-k128-fine-guessed-sweep/official-first-layer-q4-head/"
            "complete-online-float64-residual-bank/target-seeing-minimum-l1-direct-token-margin-lp/"
            "highs-ds-active-set/full-vocabulary-fp64-roundoff-certificate/"
            "build-and-untouched-holdout/cheapest-kill-first"
        ),
        "config_sha256": sha256_file(config_path),
        "checkpoint": identity,
        "q4_head": q4_head.manifest.to_dict(),
        "controls": {
            "model_free": controls,
            "integrity_failures": integrity_failures,
            "integrity_passed": integrity_passed,
        },
        "case_rows": rows,
        "build_aggregate": build,
        "holdout_aggregate": holdout,
        "target_projection": resources,
        "dev_resource": {
            "first_layer_parameter_bytes": module_parameter_bytes(
                model.model.layers[0]
            ),
            "q4_head_artifact_bytes": q4_head.manifest.artifact_bytes,
            "float64_residual_bank_bytes": residual_buffer_bytes(
                block_length=block_length,
                vocabulary_size=int(model.config.vocab_size),
                value_bytes=8,
            ),
            "float32_hidden_bank_bytes": block_length
            * int(model.config.hidden_size)
            * 4,
            "float64_coefficient_block_bytes": block_length
            * block_length
            * 8,
            "fine_block_calls": len(rows),
            "coarse_old_calls": len(rows) * block_length,
            "true_path_coarse_calls": len(rows) * block_length,
            "incremental_target_calls": len(rows) * block_length,
        },
        "gates": {
            "integrity_passed": integrity_passed,
            "build_all_positions_certified": build["all_positions_certified"],
            "holdout_all_positions_certified": holdout[
                "all_positions_certified"
            ],
            "explicit_infeasible_position_observed": explicit_infeasible,
            "hot_projection_passed": resources["hot_limit_passed"],
            "causal_coefficient_generator_constructed": False,
        },
        "authoritative_decision": decision,
    }
    result = {
        **deterministic_core,
        "deterministic_core_sha256": canonical_sha256(deterministic_core),
        "runtime": {
            **runtime,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "timing": {
            "model_load_ns": model_load_ns,
            "q4_compile_ns": q4_compile_ns,
            "experiment_ns": experiment_ns,
        },
        "provenance": {
            "measured": [
                "DEV-W guessed fine/coarse blocks, online residual bank, delayed true fine/coarse blocks, target-seeing LP statuses and complete-vocabulary FP64 margin certificates"
            ],
            "derived": [
                "minimum-L1 coefficients, FP64 roundoff lower margins, coefficient/certificate resource equations"
            ],
            "granted": [
                "the delayed target token and true shallow state are oracle inputs to coefficient synthesis"
            ],
            "unverified": [
                "causal coefficient generator, target-independent certificate, TARGET-W behavior, physical kernels, fine arithmetic reduction, 8-GiB allocation, 4B-class latency"
            ],
        },
    }

    write_json(output_dir / "artifacts/deterministic_core.json", deterministic_core)
    write_json(output_dir / "raw/case_rows.json", rows)
    write_json(output_dir / "result.json", result)
    certificate_paths = [row["certificate_artifact"] for row in rows]
    write_checksums(
        output_dir,
        [
            "artifacts/checkpoint_identity.json",
            "artifacts/q4_head_manifest.json",
            "artifacts/deterministic_core.json",
            "raw/case_rows.json",
            "result.json",
            *certificate_paths,
        ],
    )
    print(
        json.dumps(
            {
                "authoritative_decision": decision,
                "build_certificate_fraction": build["certificate_fraction"],
                "holdout_certificate_fraction": holdout["certificate_fraction"],
                "build_infeasible_cases": build["decisive_infeasible_cases"],
                "holdout_infeasible_cases": holdout[
                    "decisive_infeasible_cases"
                ],
                "hot_total_gib": resources["total_hot_gib"],
                "hot_limit_passed": resources["hot_limit_passed"],
            },
            indent=2,
            sort_keys=True,
        )
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
