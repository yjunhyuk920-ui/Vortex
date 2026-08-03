#!/usr/bin/env python3
"""Run EXP-047R on pinned, unmodified small dense checkpoints.

The run performs offline current-token LM-head pair-margin analysis. It does not
replace the LM-head operation and therefore cannot earn E2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

from vortex_runtime.cptc import CPTCConfig, certify_sum_sign, sign_decision
from vortex_runtime.cptc_audit import (
    certify_stratified_sum_sign,
    exact_state_range,
    global_symmetric_range,
    pair_margin_tile_contributions,
    quantile_strata,
    tile_bounds_from_weight_span,
)


ROOT = Path(__file__).resolve().parents[2]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    data = json.loads(raw)
    data["_sha256"] = sha256_bytes(raw)
    return data


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def median_elapsed_ns(
    callable_: Callable[[], object], repetitions: int
) -> tuple[object, int]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    result: object | None = None
    elapsed: list[int] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        result = callable_()
        elapsed.append(time.perf_counter_ns() - start)
    assert result is not None
    return result, int(statistics.median(elapsed))


def snapshot_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        records.append(
            {
                "path": file_path.relative_to(path).as_posix(),
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return records


def resolve_and_snapshot(
    *,
    model_id: str,
    requested_revision: str | None,
    cache_dir: Path,
) -> tuple[str, Path, list[dict[str, Any]]]:
    from huggingface_hub import model_info, snapshot_download

    info = model_info(model_id, revision=requested_revision or "main")
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a resolved SHA for {model_id}")
    resolved = str(info.sha)
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=resolved,
            cache_dir=cache_dir,
        )
    )
    return resolved, snapshot, snapshot_manifest(snapshot)


def model_layer_count(config: object) -> int | None:
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None


def run_case(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    weight: Any,
    bias: Any,
    weight_span: Any,
    model_record: dict[str, Any],
    prompt: str,
    prompt_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(config["max_input_tokens"]),
    )
    with torch.inference_mode():
        outputs = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden = outputs.hidden_states[-1][0, -1].detach().cpu().to(torch.float64)
    exact_logits = outputs.logits[0, -1].detach().cpu().to(torch.float64)
    output_layer = model.get_output_embeddings()
    with torch.inference_mode():
        reconstructed = (
            output_layer(hidden.to(dtype=weight.dtype).reshape(1, 1, -1))
            .reshape(-1)
            .detach()
            .cpu()
            .to(torch.float64)
        )
    reconstruction_error = float((reconstructed - exact_logits).abs().max().item())
    tolerance = float(config["logit_reconstruction_atol"])
    if not math.isfinite(reconstruction_error) or reconstruction_error > tolerance:
        raise RuntimeError(
            f"LM-head reconstruction mismatch {reconstruction_error} > {tolerance}"
        )

    top_values, top_indices = torch.topk(exact_logits, k=2)
    top_index = int(top_indices[0].item())
    competitor_index = int(top_indices[1].item())
    exact_logit_margin = float((top_values[0] - top_values[1]).item())
    if not exact_logit_margin > 0.0:
        raise RuntimeError("top-1 versus runner-up margin must be positive")

    hidden_values = hidden.tolist()
    top_weight = weight[top_index].detach().cpu().to(torch.float64).tolist()
    competitor_weight = (
        weight[competitor_index].detach().cpu().to(torch.float64).tolist()
    )
    tile_size = int(config["tile_size"])
    contributions = pair_margin_tile_contributions(
        hidden_values,
        top_weight,
        competitor_weight,
        tile_size=tile_size,
    )
    base_margin = 0.0
    if bias is not None:
        base_margin = float(
            (bias[top_index] - bias[competitor_index]).detach().cpu().item()
        )
    materialized_total = base_margin + math.fsum(contributions)
    if not math.isclose(
        materialized_total,
        exact_logit_margin,
        rel_tol=float(config["margin_reconstruction_rtol"]),
        abs_tol=float(config["margin_reconstruction_atol"]),
    ):
        raise RuntimeError(
            "tile contributions do not reconstruct the exact top-1 margin: "
            f"{materialized_total} versus {exact_logit_margin}"
        )
    reference_decision = sign_decision(materialized_total)

    bounds = tile_bounds_from_weight_span(
        hidden_values,
        weight_span.detach().cpu().to(torch.float64).tolist(),
        tile_size=tile_size,
    )
    tolerance_slack = float(config["bound_validation_atol"])
    bound_violations = sum(
        int(abs(value) > bound + tolerance_slack)
        for value, bound in zip(contributions, bounds)
    )
    if bound_violations:
        raise RuntimeError(f"checkpoint-derived bounds violated {bound_violations} times")

    certificate_config = CPTCConfig(
        delta=float(config["delta_per_decision"]),
        min_samples=int(config["min_samples"]),
        max_sample_fraction=1.0,
        seed=int(config["seed"]) + prompt_index,
    )
    repetitions = int(config["timing_repetitions"])

    _, reference_elapsed = median_elapsed_ns(
        lambda: base_margin + math.fsum(contributions), repetitions
    )

    c0_low, c0_high = global_symmetric_range(bounds)
    c0, c0_elapsed = median_elapsed_ns(
        lambda: certify_sum_sign(
            contributions,
            value_min=c0_low,
            value_max=c0_high,
            base_margin=base_margin,
            config=certificate_config,
        ),
        repetitions,
    )

    c1_low, c1_high = exact_state_range(contributions)
    c1, c1_elapsed = median_elapsed_ns(
        lambda: certify_sum_sign(
            contributions,
            value_min=c1_low,
            value_max=c1_high,
            base_margin=base_margin,
            config=certificate_config,
        ),
        repetitions,
    )

    strata = quantile_strata(bounds, int(config["stratum_count"]))
    c2, c2_elapsed = median_elapsed_ns(
        lambda: certify_stratified_sum_sign(
            contributions,
            lower_bounds=[-bound for bound in bounds],
            upper_bounds=bounds,
            strata=strata,
            base_margin=base_margin,
            config=certificate_config,
        ),
        repetitions,
    )

    def result_record(result: Any, elapsed_ns: int) -> dict[str, Any]:
        return {
            "decision": result.decision,
            "certified": result.certified,
            "fallback": result.fallback,
            "total_tiles_evaluated": result.total_tiles_evaluated,
            "evaluated_fraction": result.total_tiles_evaluated / len(contributions),
            "wrong_accept": bool(
                result.certified and result.decision != reference_decision
            ),
            "materialized_contribution_cpu_elapsed_ns": elapsed_ns,
            "elapsed_over_full_sum": elapsed_ns / max(reference_elapsed, 1),
        }

    prompt_hash = sha256_bytes(prompt.encode("utf-8"))
    return {
        "model_id": model_record["model_id"],
        "model_revision": model_record["resolved_revision"],
        "prompt_index": prompt_index,
        "prompt_sha256": prompt_hash,
        "input_tokens": int(encoded["input_ids"].numel()),
        "layer_count": model_record["layer_count"],
        "hidden_size": int(hidden.numel()),
        "vocab_size": int(weight.shape[0]),
        "tile_size": tile_size,
        "tile_count": len(contributions),
        "top_token_id": top_index,
        "runner_up_token_id": competitor_index,
        "exact_logit_margin": exact_logit_margin,
        "materialized_margin": materialized_total,
        "lm_head_reconstruction_max_abs_error": reconstruction_error,
        "bound_violations": bound_violations,
        "future_generated_tokens_used": False,
        "real_operation_replacement": False,
        "offline_full_contribution_oracle": True,
        "full_sum_materialized_contribution_cpu_elapsed_ns": reference_elapsed,
        "C0_global_checkpoint_bound": result_record(c0, c0_elapsed),
        "C1_exact_state_oracle_range": result_record(c1, c1_elapsed),
        "C2_checkpoint_span_stratified": result_record(c2, c2_elapsed),
        "C3_variance_adaptive": {
            "status": "NOT IMPLEMENTED",
            "reason": "requires a separately committed independent finite-population proof and validator",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "exp_047r" / "config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "exp_047r_candidate",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output
    raw_dir = output / "raw"
    processed_dir = output / "processed"
    logs_dir = output / "logs"
    artifacts_dir = output / "artifacts"
    for path in (raw_dir, processed_dir, logs_dir, artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.manual_seed(int(config["seed"]))
    cache_dir = Path(os.environ.get("HF_HOME", output / "hf_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_spec = config["tokenizer"]
    tokenizer_revision, tokenizer_path, tokenizer_manifest = resolve_and_snapshot(
        model_id=str(tokenizer_spec["model_id"]),
        requested_revision=tokenizer_spec.get("revision"),
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )

    model_records: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for model_index, spec in enumerate(config["models"]):
        model_id = str(spec["model_id"])
        revision, snapshot, manifest = resolve_and_snapshot(
            model_id=model_id,
            requested_revision=spec.get("revision"),
            cache_dir=cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
        )
        model.eval()
        output_layer = model.get_output_embeddings()
        weight = output_layer.weight.detach().cpu()
        bias = getattr(output_layer, "bias", None)
        weight_span = weight.amax(dim=0) - weight.amin(dim=0)
        model_record = {
            "model_id": model_id,
            "resolved_revision": revision,
            "snapshot_manifest": manifest,
            "parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "layer_count": model_layer_count(model.config),
            "hidden_size": int(weight.shape[1]),
            "vocab_size": int(weight.shape[0]),
            "lm_head_weight_bytes": int(weight.numel() * weight.element_size()),
            "column_span_metadata_bytes": int(
                weight_span.numel() * weight_span.element_size()
            ),
        }
        model_records.append(model_record)
        for prompt_index, prompt in enumerate(config["held_out_prompts"]):
            case = run_case(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                weight=weight,
                bias=bias,
                weight_span=weight_span,
                model_record=model_record,
                prompt=str(prompt),
                prompt_index=model_index * len(config["held_out_prompts"])
                + prompt_index,
                config=config,
            )
            cases.append(case)
            print(json.dumps({"case": case}, sort_keys=True), flush=True)
        del model
        del weight
        del weight_span

    raw_cases = raw_dir / "cases.jsonl"
    raw_cases.write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    checkpoint_manifest = {
        "tokenizer": {
            "model_id": tokenizer_spec["model_id"],
            "resolved_revision": tokenizer_revision,
            "files": tokenizer_manifest,
        },
        "models": model_records,
    }
    (raw_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    oracle_fractions = [
        case["C1_exact_state_oracle_range"]["evaluated_fraction"] for case in cases
    ]
    c2_fractions = [
        case["C2_checkpoint_span_stratified"]["evaluated_fraction"] for case in cases
    ]
    wrong_accepts = sum(
        int(case[name]["wrong_accept"])
        for case in cases
        for name in (
            "C0_global_checkpoint_bound",
            "C1_exact_state_oracle_range",
            "C2_checkpoint_span_stratified",
        )
    )
    bound_violations = sum(int(case["bound_violations"]) for case in cases)
    oracle_median = float(statistics.median(oracle_fractions))
    oracle_p90 = float(percentile(oracle_fractions, 0.90))
    c2_median = float(statistics.median(c2_fractions))
    c2_p90 = float(percentile(c2_fractions, 0.90))
    c2_cost_ratios = [
        case["C2_checkpoint_span_stratified"]["elapsed_over_full_sum"]
        for case in cases
    ]
    c2_cost_median = float(statistics.median(c2_cost_ratios))

    oracle_fraction_pass = (
        oracle_median <= float(config["gate"]["oracle_median_max_fraction"])
        and oracle_p90 <= float(config["gate"]["oracle_p90_max_fraction"])
    )
    correctness_pass = wrong_accepts == 0 and bound_violations == 0
    primitive_cost_pass = c2_cost_median <= float(
        config["gate"]["materialized_selector_max_full_sum_ratio"]
    )
    range_family_survives = (
        oracle_fraction_pass and correctness_pass and primitive_cost_pass
    )
    decision = (
        "CONTINUE_TO_INDEPENDENT_C3_AND_REAL_OPERATION_REPLACEMENT"
        if range_family_survives
        else "REJECT_RANGE_BASED_CPTC_CORE_RETAIN_CERTIFICATE_AUXILIARY"
    )

    target_parameters = int(config["projection"]["target_parameters"])
    baseline_parameters = int(config["projection"]["baseline_parameters"])
    bits_per_weight = int(config["projection"]["bits_per_weight"])
    target_multiplier = float(config["projection"]["target_traffic_multiplier"])
    target_full_gib = target_parameters * bits_per_weight / 8.0 / (2**30)
    baseline_full_gib = baseline_parameters * bits_per_weight / 8.0 / (2**30)
    allowed_gib = target_multiplier * baseline_full_gib
    required_fraction = allowed_gib / target_full_gib

    summary = {
        "experiment": "EXP-047R",
        "name": "oracle_tight_and_stratified_tile_bound_audit",
        "git_commit": git_commit(),
        "workflow_run": os.environ.get("GITHUB_RUN_ID"),
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "phase_c_operation_replacement": False,
        "phase_d_status": "NOT TESTED",
        "future_information_used": False,
        "config_sha256": config["_sha256"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "checkpoint_manifest_path": "raw/checkpoint_manifest.json",
        "MEASURED": {
            "model_count": len(model_records),
            "case_count": len(cases),
            "wrong_accepts": wrong_accepts,
            "bound_violations": bound_violations,
            "oracle_median_evaluated_fraction": oracle_median,
            "oracle_p90_evaluated_fraction": oracle_p90,
            "c2_median_evaluated_fraction": c2_median,
            "c2_p90_evaluated_fraction": c2_p90,
            "c2_materialized_cpu_elapsed_over_full_sum_median": c2_cost_median,
            "models": model_records,
        },
        "DERIVED": {
            "oracle_fraction_gate_pass": oracle_fraction_pass,
            "correctness_gate_pass": correctness_pass,
            "materialized_primitive_cost_gate_pass": primitive_cost_pass,
            "range_family_survives_gate": range_family_survives,
            "decision": decision,
            "c2_bound_contract": (
                "per-dimension output-weight column span times absolute current "
                "hidden activation, aggregated by tile and union-accounted strata"
            ),
            "c3_status": "NOT IMPLEMENTED pending independent proof",
        },
        "PROJECTED": {
            "target_q4_full_weight_gib_per_full_pass": target_full_gib,
            "baseline_q4_full_weight_gib_per_full_pass": baseline_full_gib,
            "allowed_1_2x_baseline_weight_gib_per_token": allowed_gib,
            "required_average_target_weight_fraction_before_overhead": required_fraction,
            "oracle_median_over_required_target_fraction": (
                oracle_median / required_fraction
            ),
        },
        "UNVERIFIED": [
            "variance-adaptive finite-population C3 certificate",
            "real LM-head operation replacement",
            "model-wide nonlinear propagation",
            "accelerator selector cost",
            "70B and 405B behavior",
            "8 GiB peak VRAM",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "gate": {
            "pre_registered_oracle_median_max_fraction": config["gate"][
                "oracle_median_max_fraction"
            ],
            "pre_registered_oracle_p90_max_fraction": config["gate"][
                "oracle_p90_max_fraction"
            ],
            "pre_registered_wrong_accept_limit": 0,
            "pre_registered_materialized_selector_max_full_sum_ratio": config[
                "gate"
            ]["materialized_selector_max_full_sum_ratio"],
            "passes": range_family_survives,
            "decision": decision,
        },
    }
    (processed_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "oracle_fractions": oracle_fractions,
                "c2_fractions": c2_fractions,
                "c2_cost_ratios": c2_cost_ratios,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (logs_dir / "run.log").write_text(
        json.dumps(
            {
                "decision": decision,
                "gate_passes": range_family_survives,
                "wrong_accepts": wrong_accepts,
                "bound_violations": bound_violations,
                "phase_d_status": "NOT TESTED",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifacts_dir / "contract.txt").write_text(
        "Offline current-token LM-head pair-margin range audit only. "
        "No real operation replacement; evidence ceiling E1; Phase D NOT TESTED.\n",
        encoding="utf-8",
    )

    checksum_lines: list[str] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "checksums.sha256" or cache_dir in path.parents:
            continue
        checksum_lines.append(
            f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        )
    (output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
