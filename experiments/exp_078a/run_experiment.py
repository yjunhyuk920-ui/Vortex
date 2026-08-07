#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import time
import types
from typing import Any, Iterator

from experiments.exp_077a.run_experiment import (
    compare_logits,
    dump,
    dump_rows,
    homogeneous_length_batches,
    load_target,
    logits_sha256,
    registered_teacher_forcing_tokens,
    tokenize_prompt,
    validate_registered_inputs,
    write_checksums,
)
from vortex_runtime.fractal_oracle import canonical_sha256
from vortex_runtime.tangent_macroblock import (
    aggregate_quality_rows,
    dense_macroblock_cost,
    family_aggregates,
    gate_decision,
    minimum_hot_tokens,
    valid_prefix_length,
)


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


class TangentTracker:
    def __init__(self, batch_size: int, horizon: int) -> None:
        if batch_size <= 0 or horizon <= 0:
            raise ValueError("tracker dimensions must be positive")
        self.batch_size = batch_size
        self.horizon = horizon
        self.squared_error = [0.0] * batch_size
        self.squared_target = [0.0] * batch_size
        self.horizon_squared_error = [[0.0] * horizon for _ in range(batch_size)]
        self.horizon_squared_target = [[0.0] * horizon for _ in range(batch_size)]
        self.calls = 0

    def record(self, candidate: Any, target: Any) -> None:
        if candidate.shape != target.shape:
            raise ValueError("candidate and target MLP shapes differ")
        if candidate.ndim != 3 or int(candidate.shape[0]) != self.batch_size:
            raise ValueError("tracker batch shape mismatch")
        if int(candidate.shape[1]) != self.horizon:
            raise ValueError("tracker horizon mismatch")
        delta = candidate.float() - target.float()
        errors = delta.square().sum(dim=-1).tolist()
        targets = target.float().square().sum(dim=-1).tolist()
        for batch_index in range(self.batch_size):
            for position in range(self.horizon):
                error = float(errors[batch_index][position])
                magnitude = float(targets[batch_index][position])
                self.squared_error[batch_index] += error
                self.squared_target[batch_index] += magnitude
                self.horizon_squared_error[batch_index][position] += error
                self.horizon_squared_target[batch_index][position] += magnitude
        self.calls += 1

    def case_summary(self, index: int) -> dict[str, Any]:
        return {
            "mlp_calls": self.calls,
            "mlp_relative_l2": math.sqrt(
                self.squared_error[index] / max(self.squared_target[index], 1e-30)
            ),
            "mlp_horizon_relative_l2": [
                math.sqrt(error / max(target, 1e-30))
                for error, target in zip(
                    self.horizon_squared_error[index],
                    self.horizon_squared_target[index],
                )
            ],
        }


class TangentController:
    def __init__(self, layer_count: int) -> None:
        self.phase = "capture"
        self.anchors: list[Any | None] = [None] * layer_count


@contextmanager
def frozen_tangent_oracle(
    target: Any, tracker: TangentTracker
) -> Iterator[TangentController]:
    controller = TangentController(len(target.model.layers))
    originals: list[tuple[Any, Any]] = []
    for layer_index, layer in enumerate(target.model.layers):
        mlp = layer.mlp
        original = mlp.forward

        def tangent_forward(self: Any, x: Any, *, _layer=layer_index) -> Any:
            up = self.up_proj(x)
            if controller.phase == "capture":
                coefficient = self.act_fn(self.gate_proj(x))
                controller.anchors[_layer] = coefficient[:, -1, :].detach().clone()
                return self.down_proj(coefficient * up)
            if controller.phase != "reuse":
                raise ValueError(f"unknown tangent phase: {controller.phase}")
            anchor = controller.anchors[_layer]
            if anchor is None:
                raise ValueError(f"missing tangent anchor for layer {_layer}")
            candidate = self.down_proj(anchor.unsqueeze(1) * up)
            full = self.down_proj(self.act_fn(self.gate_proj(x)) * up)
            tracker.record(candidate, full)
            return candidate

        originals.append((mlp, original))
        mlp.forward = types.MethodType(tangent_forward, mlp)
    try:
        yield controller
    finally:
        for mlp, original in originals:
            mlp.forward = original


def cached_tangent_logits(
    target: Any,
    prefix_ids: Any,
    conditioning_ids: Any,
    tracker: TangentTracker,
) -> Any:
    import torch

    with frozen_tangent_oracle(target, tracker) as controller:
        prefix_output = target.model(input_ids=prefix_ids, use_cache=True)
        first_logits = target.lm_head(prefix_output.last_hidden_state[:, -1:, :])
        controller.phase = "reuse"
        verify_output = target.model(
            input_ids=conditioning_ids,
            past_key_values=prefix_output.past_key_values,
            use_cache=True,
        )
        verify_logits = target.lm_head(verify_output.last_hidden_state)
    return torch.cat([first_logits, verify_logits], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()

    import torch
    import transformers
    import tokenizers
    import huggingface_hub
    import safetensors

    config_path = Path(arguments.config).resolve()
    model_dir = Path(arguments.model_dir).resolve()
    output = Path(arguments.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_started = time.perf_counter_ns()
    torch.set_num_threads(int(config["runtime"]["torch_num_threads"]))
    torch.use_deterministic_algorithms(True)
    inputs = validate_registered_inputs(config, model_dir)
    target, tokenizer, parameter_audit = load_target(model_dir)

    prompts_by_id = {
        row["id"]: {**row, "split": split}
        for split in ("build", "evaluation")
        for row in inputs["prompts"][split]
    }
    traces_by_id = {row["prompt_id"]: row for row in inputs["traces"]}
    token_count = int(config["registered_inputs"]["tokens_per_prompt"])
    horizon = int(config["registered_inputs"]["observed_reuse_horizon"])
    if horizon != token_count - 1:
        raise ValueError("registered horizon must equal tokens_per_prompt - 1")
    ordered_ids = [
        row["id"]
        for split in ("build", "evaluation")
        for row in inputs["prompts"][split]
    ]
    prepared: list[dict[str, Any]] = []
    for prompt_id in ordered_ids:
        prompt = prompts_by_id[prompt_id]
        conditioning, continuation = registered_teacher_forcing_tokens(
            traces_by_id[prompt_id], token_count=token_count
        )
        prefix_ids = tokenize_prompt(tokenizer, prompt["prompt"])
        prepared.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "continuation": continuation,
                "prefix_length": int(prefix_ids.shape[1]),
                "prefix_ids": prefix_ids[0],
                "conditioning_ids": torch.tensor(conditioning, dtype=torch.long),
            }
        )

    batches = homogeneous_length_batches(
        [int(row["prefix_ids"].shape[0]) for row in prepared]
    )
    baseline_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    baseline_logits_by_id: dict[str, Any] = {}
    baseline_mismatches = 0
    from experiments.exp_077a.run_experiment import cached_trace_logits

    for batch in batches:
        prefix_batch = torch.stack([prepared[index]["prefix_ids"] for index in batch])
        conditioning_batch = torch.stack(
            [prepared[index]["conditioning_ids"] for index in batch]
        )
        with torch.inference_mode():
            baseline_batch = cached_trace_logits(
                target, prefix_batch, conditioning_batch
            ).detach()
        for local_index, prepared_index in enumerate(batch):
            item = prepared[prepared_index]
            prompt = item["prompt"]
            logits = baseline_batch[local_index].clone()
            baseline_logits_by_id[item["prompt_id"]] = logits
            observed = [int(value) for value in logits.argmax(dim=-1).tolist()]
            mismatches = sum(
                left != right for left, right in zip(observed, item["continuation"])
            )
            baseline_mismatches += mismatches
            baseline_rows.append(
                {
                    "prompt_id": item["prompt_id"],
                    "split": prompt["split"],
                    "family": prompt["family"],
                    "prefix_token_count": item["prefix_length"],
                    "registered_target_tokens": item["continuation"],
                    "baseline_top_tokens": observed,
                    "trace_mismatches": mismatches,
                    "baseline_logits_sha256": logits_sha256(logits),
                }
            )

    for batch in batches:
        tracker = TangentTracker(len(batch), horizon)
        prefix_batch = torch.stack([prepared[index]["prefix_ids"] for index in batch])
        conditioning_batch = torch.stack(
            [prepared[index]["conditioning_ids"] for index in batch]
        )
        started = time.perf_counter_ns()
        with torch.inference_mode():
            candidate_batch = cached_tangent_logits(
                target, prefix_batch, conditioning_batch, tracker
            )
        wall_ns = time.perf_counter_ns() - started
        for local_index, prepared_index in enumerate(batch):
            item = prepared[prepared_index]
            prompt = item["prompt"]
            prompt_id = item["prompt_id"]
            target_logits = baseline_logits_by_id[prompt_id][1:]
            candidate_logits = candidate_batch[local_index][1:]
            comparison = compare_logits(target_logits, candidate_logits)
            valid = valid_prefix_length(
                target_top_tokens=comparison["target_top_tokens"],
                candidate_top_tokens=comparison["candidate_top_tokens"],
                token_kls=comparison["token_kls"],
                max_token_kl=float(
                    config["quality_gate"]["max_single_token_kl_for_valid_prefix"]
                ),
            )
            row = {
                "prompt_id": prompt_id,
                "split": prompt["split"],
                "family": prompt["family"],
                **comparison,
                **tracker.case_summary(local_index),
                "valid_prefix_length": valid,
                "candidate_logits_sha256": logits_sha256(candidate_logits),
                "homogeneous_batch_wall_ns": wall_ns,
            }
            candidate_rows.append(row)
            print(
                json.dumps(
                    {
                        "prompt_id": prompt_id,
                        "split": prompt["split"],
                        "top1_matches": comparison["top1_matches"],
                        "token_count": horizon,
                        "valid_prefix_length": valid,
                        "mean_kl": sum(comparison["token_kls"]) / horizon,
                        "mlp_relative_l2": row["mlp_relative_l2"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    build_rows = [row for row in candidate_rows if row["split"] == "build"]
    evaluation_rows = [row for row in candidate_rows if row["split"] == "evaluation"]
    aggregates = {
        "build": aggregate_quality_rows(build_rows),
        "evaluation": aggregate_quality_rows(evaluation_rows),
        "evaluation_families": family_aggregates(evaluation_rows),
    }
    cost_config = config["cost_gate"]
    small = cost_config["small_checkpoint"]
    surrogate = cost_config["qwen35_122b_surrogate"]
    small_cost = dense_macroblock_cost(
        hidden_size=int(small["hidden_size"]),
        intermediate_size=int(small["intermediate_size"]),
        active_paths=int(small["active_paths"]),
    )
    surrogate_cost = dense_macroblock_cost(
        hidden_size=int(surrogate["hidden_size"]),
        intermediate_size=int(surrogate["expert_intermediate_size"]),
        active_paths=int(surrogate["active_paths"]),
    )

    def lifetime_requirements(cost: dict[str, Any]) -> dict[str, int | None]:
        return {
            "p50_hot_tokens": minimum_hot_tokens(
                allowance_fraction=float(
                    cost_config["p50_allowance_fraction_of_exact_active_mlp"]
                ),
                hot_fraction=float(cost["hot_fraction_of_exact"]),
                construction_token_equivalents=float(
                    cost["materialization_exact_token_equivalents"]
                ),
                exact_anchor_token_equivalents=float(
                    cost_config["exact_anchor_token_equivalents"]
                ),
            ),
            "p05_hot_tokens_for_p95_latency": minimum_hot_tokens(
                allowance_fraction=float(
                    cost_config["p95_allowance_fraction_of_exact_active_mlp"]
                ),
                hot_fraction=float(cost["hot_fraction_of_exact"]),
                construction_token_equivalents=float(
                    cost["materialization_exact_token_equivalents"]
                ),
                exact_anchor_token_equivalents=float(
                    cost_config["exact_anchor_token_equivalents"]
                ),
            ),
        }

    small_requirements = lifetime_requirements(small_cost)
    surrogate_requirements = lifetime_requirements(surrogate_cost)
    gates, decision = gate_decision(
        baseline_mismatches=baseline_mismatches,
        population=aggregates["evaluation"],
        families=aggregates["evaluation_families"],
        observed_horizon=horizon,
        required_p50_hot_tokens=small_requirements["p50_hot_tokens"],
        required_p05_hot_tokens=small_requirements[
            "p05_hot_tokens_for_p95_latency"
        ],
        min_top1_agreement=float(config["quality_gate"]["min_top1_agreement"]),
        min_family_top1_agreement=float(
            config["quality_gate"]["min_each_family_top1_agreement"]
        ),
        max_mean_kl=float(
            config["quality_gate"]["max_mean_target_to_candidate_kl"]
        ),
        max_p95_kl=float(
            config["quality_gate"]["max_p95_target_to_candidate_kl"]
        ),
    )
    deterministic_core = {
        "baseline_trace_mismatches": baseline_mismatches,
        "aggregates": aggregates,
        "small_cost": small_cost,
        "small_lifetime_requirements": small_requirements,
        "surrogate_cost": surrogate_cost,
        "surrogate_lifetime_requirements": surrogate_requirements,
        "gates": gates,
        "decision": decision,
        "token_core": [
            {
                "prompt_id": row["prompt_id"],
                "target_top_tokens": row["target_top_tokens"],
                "candidate_top_tokens": row["candidate_top_tokens"],
                "token_kls": row["token_kls"],
                "valid_prefix_length": row["valid_prefix_length"],
            }
            for row in candidate_rows
        ],
    }
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "source_commit": git_commit(),
        "config_sha256": sha256_file(config_path),
    }
    summary = {
        "experiment": "EXP-078A",
        "name": config["name"],
        "phase": ["C-small-real-checkpoint-favorable-lifetime-observation"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "REGISTERED_EXTERNAL_INPUTS": {
            "checkpoint": {
                "model_id": config["checkpoint"]["model_id"],
                "revision": config["checkpoint"]["revision"],
                "weight_sha256": config["checkpoint"]["weight_sha256"],
            },
            "prompts_sha256": config["registered_inputs"]["prompts_sha256"],
            "target_trace_sha256": config["registered_inputs"]["target_trace_sha256"],
        },
        "MEASURED": {
            "baseline_case_count": len(baseline_rows),
            "candidate_case_count": len(candidate_rows),
            "candidate_token_count": len(candidate_rows) * horizon,
            "baseline_trace_mismatches": baseline_mismatches,
            "wall_ns": time.perf_counter_ns() - run_started,
            "deterministic_core_sha256": canonical_sha256(deterministic_core),
        },
        "DERIVED": {
            "parameter_audit": parameter_audit,
            "aggregates": aggregates,
            "small_checkpoint_cost": small_cost,
            "small_checkpoint_lifetime_requirements": small_requirements,
            "qwen35_122b_surrogate_cost": surrogate_cost,
            "qwen35_122b_surrogate_lifetime_requirements": surrogate_requirements,
            "gate": gates,
            "decision": decision,
        },
        "UNVERIFIED": [
            "a constructor asymptotically cheaper than direct macro materialization",
            "reuse beyond the seven-token registered observation horizon",
            "rank-compressed macro construction and quality",
            "sentinel correctness and cost",
            "exact cache repair and fallback cost",
            "physical macro materialization application traffic latency and VRAM",
            "35B 122B and arbitrary dense 405B scaling",
            "E2 through E7",
        ],
        "claim_boundary": {
            "checkpoint": "UNCHANGED_PINNED_OFFICIAL_BF16_PAYLOAD",
            "anchor": "CAUSAL_EXACT_LAST_PROMPT_TOKEN_NO_FUTURE_TARGET_READ",
            "candidate": "FROZEN_EXACT_ANCHOR_COEFFICIENT_REUSED_FOR_SEVEN_CONDITIONING_POSITIONS",
            "construction": "DIRECT_COST_DERIVED_NOT_PHYSICALLY_EXECUTED",
            "operation_replacement": "REFERENCE_FACTORIZED_UP_DOWN_EXECUTION_NOT_A_SPEED_KERNEL",
            "fallback": "NOT_IMPLEMENTED",
            "quality": "TARGET_LOGIT_DISTRIBUTION_AGREEMENT_NOT_GROUND_TRUTH_TASK_QUALITY",
            "target_server": "NO_COMMAND_EXECUTED",
        },
        "provenance": environment,
    }
    dump_rows(output / "raw/baseline_rows.jsonl", baseline_rows)
    dump_rows(output / "raw/candidate_rows.jsonl", candidate_rows)
    dump(output / "raw/checkpoint_manifest.json", inputs["checkpoint_rows"])
    dump(output / "raw/input_audit.json", inputs["join_audit"])
    dump(output / "processed/aggregate.json", aggregates)
    dump(output / "processed/cost_gate.json", {
        "small_checkpoint": {"cost": small_cost, "requirements": small_requirements},
        "qwen35_122b_surrogate": {"cost": surrogate_cost, "requirements": surrogate_requirements},
    })
    dump(output / "artifacts/environment.json", environment)
    dump(output / "summary.json", summary)
    write_checksums(output)
    print(json.dumps({
        "decision": decision,
        "evaluation": aggregates["evaluation"],
        "small_lifetime_requirements": small_requirements,
        "surrogate_lifetime_requirements": surrogate_requirements,
        "deterministic_core_sha256": summary["MEASURED"]["deterministic_core_sha256"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
