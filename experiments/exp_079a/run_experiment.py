#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import platform
from pathlib import Path
import time
import types
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_077a.run_experiment import (
    cached_trace_logits,
    compare_logits,
    dump,
    dump_rows,
    git_commit,
    load_target,
    logits_sha256,
    sha256_file,
    tokenize_prompt,
    validate_registered_inputs,
    write_checksums,
)
from vortex_runtime.causal_proof_state import (
    aggregate_gate_rows,
    block_ranges,
    dct_pilot_basis,
    family_aggregates,
    gate_decision,
    target_equivalent_fraction,
    uniform_budget_plan,
)
from vortex_runtime.fractal_oracle import (
    canonical_sha256,
    homogeneous_length_batches,
    registered_teacher_forcing_tokens,
)


def tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True)
class TorchPilotSidecar:
    layer_index: int
    basis: Any
    image: Any
    block_norms: Any
    row_ranges: tuple[tuple[int, int], ...]
    image_sha256: str
    block_norms_sha256: str

    def remaining_norm_factor(self, selected_blocks: int) -> float:
        import torch

        if selected_blocks < 0 or selected_blocks > int(self.block_norms.numel()):
            raise ValueError("selected block count outside sidecar")
        if selected_blocks == int(self.block_norms.numel()):
            return 0.0
        ordered = torch.sort(self.block_norms.square(), descending=True).values
        return math.sqrt(float(ordered[selected_blocks:].sum().item()))

    def audit_row(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "input_width": int(self.basis.shape[0]),
            "pilot_rank": int(self.basis.shape[1]),
            "output_width": int(self.image.shape[0]),
            "row_block_count": len(self.row_ranges),
            "image_dtype": str(self.image.dtype),
            "image_sha256": self.image_sha256,
            "block_norms_sha256": self.block_norms_sha256,
            "maximum_block_frobenius": float(self.block_norms.max().item()),
            "minimum_block_frobenius": float(self.block_norms.min().item()),
        }


def build_sidecars(
    target: Any, *, pilot_rank: int, row_block: int
) -> tuple[TorchPilotSidecar, ...]:
    import torch

    first_width = int(target.model.layers[0].mlp.down_proj.weight.shape[1])
    basis = torch.from_numpy(dct_pilot_basis(first_width, pilot_rank)).to(
        dtype=torch.float32
    )
    rows: list[TorchPilotSidecar] = []
    for layer_index, layer in enumerate(target.model.layers):
        weight = layer.mlp.down_proj.weight.detach().float()
        if int(weight.shape[1]) != first_width:
            raise ValueError("registered down projections must share input width")
        image = (weight @ basis).contiguous()
        residual = weight - image @ basis.T
        ranges = block_ranges(int(weight.shape[0]), row_block)
        squared_rows = residual.double().square().sum(dim=1)
        exact_norms = [
            math.sqrt(float(squared_rows[start:stop].sum().item()))
            for start, stop in ranges
        ]
        norms = torch.tensor(exact_norms, dtype=torch.float32)
        norms = torch.nextafter(norms, torch.full_like(norms, math.inf))
        rows.append(
            TorchPilotSidecar(
                layer_index=layer_index,
                basis=basis,
                image=image,
                block_norms=norms,
                row_ranges=ranges,
                image_sha256=tensor_sha256(image),
                block_norms_sha256=tensor_sha256(norms),
            )
        )
    return tuple(rows)


class ProofTracker:
    def __init__(
        self,
        *,
        arm: str,
        batch_size: int,
        selected_blocks: int,
        total_blocks: int,
    ) -> None:
        if batch_size <= 0 or total_blocks <= 0:
            raise ValueError("tracker dimensions must be positive")
        if selected_blocks < 0 or selected_blocks > total_blocks:
            raise ValueError("selected block count outside tracker")
        self.arm = arm
        self.batch_size = batch_size
        self.selected_blocks = selected_blocks
        self.total_blocks = total_blocks
        self.squared_error = [0.0] * batch_size
        self.squared_target = [0.0] * batch_size
        self.squared_minimum_sound_radius = [0.0] * batch_size
        self.selected_block_observations = [0] * batch_size
        self.total_block_observations = [0] * batch_size
        self.calls = 0

    def record(
        self,
        *,
        candidate: Any,
        target: Any,
        projection_input: Any,
        minimum_remaining_norm_factor: float,
    ) -> None:
        if candidate.shape != target.shape:
            raise ValueError("candidate and target MLP outputs differ in shape")
        if candidate.ndim != 3 or int(candidate.shape[0]) != self.batch_size:
            raise ValueError("tracker batch shape mismatch")
        delta = candidate.float() - target.float()
        errors = delta.square().sum(dim=(1, 2)).tolist()
        targets = target.float().square().sum(dim=(1, 2)).tolist()
        input_norms = projection_input.detach().float().square().sum(dim=-1).sqrt()
        bounds_squared = (
            input_norms.square() * (minimum_remaining_norm_factor**2)
        ).sum(dim=1).tolist()
        positions = int(candidate.shape[1])
        for index in range(self.batch_size):
            self.squared_error[index] += float(errors[index])
            self.squared_target[index] += float(targets[index])
            self.squared_minimum_sound_radius[index] += float(bounds_squared[index])
            self.selected_block_observations[index] += (
                positions * self.selected_blocks
            )
            self.total_block_observations[index] += positions * self.total_blocks
        self.calls += 1

    def case_summary(self, index: int) -> dict[str, Any]:
        denominator = max(self.squared_target[index], 1e-30)
        return {
            "arm": self.arm,
            "mlp_calls": self.calls,
            "selected_block_observations": self.selected_block_observations[index],
            "total_block_observations": self.total_block_observations[index],
            "realized_row_block_fraction": (
                self.selected_block_observations[index]
                / self.total_block_observations[index]
            ),
            "mlp_relative_l2": math.sqrt(self.squared_error[index] / denominator),
            "minimum_sound_radius_ratio": math.sqrt(
                self.squared_minimum_sound_radius[index] / denominator
            ),
        }


@contextmanager
def block_zonotope_oracle(
    target: Any,
    *,
    sidecars: tuple[TorchPilotSidecar, ...],
    selected_blocks: int,
    tracker: ProofTracker,
) -> Iterator[None]:
    import torch

    originals: list[tuple[Any, Any]] = []
    if len(sidecars) != len(target.model.layers):
        raise ValueError("sidecar/layer count mismatch")
    for layer_index, layer in enumerate(target.model.layers):
        mlp = layer.mlp
        original = mlp.forward
        sidecar = sidecars[layer_index]
        remaining_factor = sidecar.remaining_norm_factor(selected_blocks)

        def oracle_forward(
            self: Any,
            x: Any,
            *,
            _sidecar: TorchPilotSidecar = sidecar,
            _remaining_factor: float = remaining_factor,
        ) -> Any:
            intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            full = self.down_proj(intermediate)
            source = intermediate.float()
            coefficients = source @ _sidecar.basis
            center = coefficients @ _sidecar.image.T
            residual = full.float() - center
            block_energies = torch.stack(
                [
                    residual[..., start:stop].square().sum(dim=-1)
                    for start, stop in _sidecar.row_ranges
                ],
                dim=-1,
            )
            block_mask = torch.zeros_like(block_energies, dtype=torch.bool)
            if selected_blocks:
                chosen = torch.topk(
                    block_energies,
                    k=selected_blocks,
                    dim=-1,
                    largest=True,
                    sorted=False,
                ).indices
                block_mask.scatter_(-1, chosen, True)
            row_mask = torch.zeros_like(residual, dtype=torch.bool)
            for block_index, (start, stop) in enumerate(_sidecar.row_ranges):
                row_mask[..., start:stop] = block_mask[..., block_index].unsqueeze(-1)
            candidate = torch.where(row_mask, full.float(), center).to(full.dtype)
            tracker.record(
                candidate=candidate,
                target=full,
                projection_input=intermediate,
                minimum_remaining_norm_factor=_remaining_factor,
            )
            return candidate

        originals.append((mlp, original))
        mlp.forward = types.MethodType(oracle_forward, mlp)
    try:
        yield
    finally:
        for mlp, original in originals:
            mlp.forward = original


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
    started = time.perf_counter_ns()
    torch.set_num_threads(int(config["runtime"]["torch_num_threads"]))
    torch.use_deterministic_algorithms(True)

    inputs = validate_registered_inputs(config, model_dir)
    target, tokenizer, parameter_audit = load_target(model_dir)
    shapes = parameter_audit["mlp_shapes"]
    if not shapes or any(
        int(row["hidden_size"]) != int(shapes[0]["hidden_size"])
        or int(row["intermediate_size"]) != int(shapes[0]["intermediate_size"])
        for row in shapes
    ):
        raise ValueError("registered Gate requires uniform MLP shapes")

    candidate_config = config["candidate"]
    budget_config = config["budget_gate"]
    pilot_rank = int(candidate_config["pilot_rank"])
    row_block = int(candidate_config["row_block"])
    matrix_count = len(shapes)
    rows = int(shapes[0]["hidden_size"])
    columns = int(shapes[0]["intermediate_size"])
    total_parameters = int(parameter_audit["target_text_parameters"])

    p50_fraction = target_equivalent_fraction(
        baseline_billions=float(budget_config["baseline_billions"]),
        target_billions=float(budget_config["target_billions"]),
        latency_multiple=float(budget_config["p50_latency_multiple"]),
    )
    p95_fraction = target_equivalent_fraction(
        baseline_billions=float(budget_config["baseline_billions"]),
        target_billions=float(budget_config["target_billions"]),
        latency_multiple=float(budget_config["p95_latency_multiple"]),
    )
    if abs(p50_fraction - float(budget_config["p50_target_fraction"])) > 1e-15:
        raise ValueError("registered p50 target fraction mismatch")
    if abs(p95_fraction - float(budget_config["p95_target_fraction"])) > 1e-15:
        raise ValueError("registered p95 target fraction mismatch")

    plan_arguments = {
        "total_parameter_count": total_parameters,
        "matrix_count": matrix_count,
        "rows": rows,
        "columns": columns,
        "pilot_rank": pilot_rank,
        "row_block": row_block,
        "image_scalar_bytes": int(candidate_config["pilot_image_scalar_bytes"]),
        "norm_scalar_bytes": int(candidate_config["residual_norm_scalar_bytes"]),
        "matrix_metadata_bytes": int(candidate_config["matrix_metadata_bytes"]),
        "proof_record_bytes": int(candidate_config["proof_record_bytes"]),
    }
    plans = {
        "p50": uniform_budget_plan(
            **plan_arguments, allowance_fraction=p50_fraction
        ),
        "p95": uniform_budget_plan(
            **plan_arguments, allowance_fraction=p95_fraction
        ),
    }

    target_projection = config["target_projection"]
    target_plans: dict[str, Any] = {}
    for arm, fraction in (("p50", p50_fraction), ("p95", p95_fraction)):
        target_plans[arm] = uniform_budget_plan(
            total_parameter_count=int(
                target_projection["dense_405b_total_parameters"]
            ),
            matrix_count=int(target_projection["layer_count"]),
            rows=int(target_projection["hidden_size"]),
            columns=int(target_projection["intermediate_size"]),
            pilot_rank=int(target_projection["scaled_pilot_rank"]),
            row_block=int(target_projection["scaled_row_block"]),
            allowance_fraction=fraction,
            image_scalar_bytes=int(candidate_config["pilot_image_scalar_bytes"]),
            norm_scalar_bytes=int(candidate_config["residual_norm_scalar_bytes"]),
            matrix_metadata_bytes=int(candidate_config["matrix_metadata_bytes"]),
            proof_record_bytes=int(candidate_config["proof_record_bytes"]),
        )

    prompts_by_id = {
        row["id"]: {**row, "split": split}
        for split in ("build", "evaluation")
        for row in inputs["prompts"][split]
    }
    traces_by_id = {row["prompt_id"]: row for row in inputs["traces"]}
    token_count = int(config["registered_inputs"]["tokens_per_prompt"])
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
    baseline_logits_by_id: dict[str, Any] = {}
    baseline_mismatches = 0
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
            logits = baseline_batch[local_index].clone()
            baseline_logits_by_id[item["prompt_id"]] = logits
            observed = [int(value) for value in logits.argmax(dim=-1).tolist()]
            mismatches = sum(
                left != right
                for left, right in zip(observed, item["continuation"])
            )
            baseline_mismatches += mismatches
            baseline_rows.append(
                {
                    "prompt_id": item["prompt_id"],
                    "split": item["prompt"]["split"],
                    "family": item["prompt"]["family"],
                    "prefix_token_count": item["prefix_length"],
                    "registered_target_tokens": item["continuation"],
                    "baseline_top_tokens": observed,
                    "trace_mismatches": mismatches,
                    "baseline_logits_sha256": logits_sha256(logits),
                }
            )

    sidecars = build_sidecars(
        target, pilot_rank=pilot_rank, row_block=row_block
    )
    sidecar_rows = [row.audit_row() for row in sidecars]
    case_rows: list[dict[str, Any]] = []
    log_lines: list[str] = []
    for arm in ("p50", "p95"):
        plan = plans[arm]
        for batch in batches:
            tracker = ProofTracker(
                arm=arm,
                batch_size=len(batch),
                selected_blocks=plan.selected_blocks_per_matrix,
                total_blocks=plan.blocks_per_matrix,
            )
            prefix_batch = torch.stack(
                [prepared[index]["prefix_ids"] for index in batch]
            )
            conditioning_batch = torch.stack(
                [prepared[index]["conditioning_ids"] for index in batch]
            )
            batch_started = time.perf_counter_ns()
            with (
                block_zonotope_oracle(
                    target,
                    sidecars=sidecars,
                    selected_blocks=plan.selected_blocks_per_matrix,
                    tracker=tracker,
                ),
                torch.inference_mode(),
            ):
                candidate_batch = cached_trace_logits(
                    target, prefix_batch, conditioning_batch
                )
            batch_wall_ns = time.perf_counter_ns() - batch_started
            for local_index, prepared_index in enumerate(batch):
                item = prepared[prepared_index]
                prompt_id = item["prompt_id"]
                comparison = compare_logits(
                    baseline_logits_by_id[prompt_id],
                    candidate_batch[local_index],
                )
                row = {
                    "prompt_id": prompt_id,
                    "split": item["prompt"]["split"],
                    "family": item["prompt"]["family"],
                    **comparison,
                    **tracker.case_summary(local_index),
                    "candidate_logits_sha256": logits_sha256(
                        candidate_batch[local_index]
                    ),
                    "homogeneous_batch_wall_ns": batch_wall_ns,
                }
                case_rows.append(row)
                line = json.dumps(
                    {
                        "arm": arm,
                        "prompt_id": prompt_id,
                        "split": row["split"],
                        "top1_matches": row["top1_matches"],
                        "token_count": row["token_count"],
                        "mean_kl": sum(row["token_kls"]) / row["token_count"],
                        "mlp_relative_l2": row["mlp_relative_l2"],
                        "minimum_sound_radius_ratio": row[
                            "minimum_sound_radius_ratio"
                        ],
                    },
                    sort_keys=True,
                )
                print(line, flush=True)
                log_lines.append(line)

    aggregates: dict[str, Any] = {}
    for arm in ("p50", "p95"):
        arm_rows = [row for row in case_rows if row["arm"] == arm]
        evaluation = [row for row in arm_rows if row["split"] == "evaluation"]
        aggregates[arm] = {
            "build": aggregate_gate_rows(
                [row for row in arm_rows if row["split"] == "build"]
            ),
            "evaluation": aggregate_gate_rows(evaluation),
            "evaluation_families": family_aggregates(evaluation),
        }

    gate_arm = str(budget_config["gate_arm"])
    gate_plan = plans[gate_arm]
    quality_config = config["quality_gate"]
    gates, decision = gate_decision(
        baseline_mismatches=baseline_mismatches,
        budget_fraction=gate_plan.charged_traffic_fraction,
        maximum_budget_fraction=float(budget_config[f"{gate_arm}_target_fraction"]),
        population=aggregates[gate_arm]["evaluation"],
        families=aggregates[gate_arm]["evaluation_families"],
        min_top1_agreement=float(quality_config["min_top1_agreement"]),
        min_family_top1_agreement=float(
            quality_config["min_each_family_top1_agreement"]
        ),
        max_mean_kl=float(
            quality_config["max_mean_target_to_candidate_kl"]
        ),
        max_p95_kl=float(
            quality_config["max_p95_target_to_candidate_kl"]
        ),
        max_radius_ratio_p50=float(
            quality_config["max_minimum_sound_radius_ratio_p50"]
        ),
        max_radius_ratio_p95=float(
            quality_config["max_minimum_sound_radius_ratio_p95"]
        ),
    )

    deterministic_core = {
        "baseline_trace_mismatches": baseline_mismatches,
        "plans": {key: value.as_dict() for key, value in plans.items()},
        "target_plans": {
            key: value.as_dict() for key, value in target_plans.items()
        },
        "aggregates": aggregates,
        "gates": gates,
        "decision": decision,
        "input_join_sha256": inputs["join_audit"]["joined_core_sha256"],
        "sidecar_core": sidecar_rows,
        "token_core": [
            {
                "prompt_id": row["prompt_id"],
                "arm": row["arm"],
                "target_top_tokens": row["target_top_tokens"],
                "candidate_top_tokens": row["candidate_top_tokens"],
                "token_kls": row["token_kls"],
                "minimum_sound_radius_ratio": row[
                    "minimum_sound_radius_ratio"
                ],
            }
            for row in case_rows
        ],
    }
    core_hash = canonical_sha256(deterministic_core)
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
        "requirements_lock_sha256": inputs["requirements_lock_sha256"],
    }
    summary = {
        "experiment": "EXP-079A",
        "name": config["name"],
        "phase": ["B-reference", "C-small-real-checkpoint-favorable-oracle"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "REGISTERED_EXTERNAL_INPUTS": {
            "checkpoint": {
                "model_id": config["checkpoint"]["model_id"],
                "revision": config["checkpoint"]["revision"],
                "weight_sha256": config["checkpoint"]["weight_sha256"],
            },
            "prompts_sha256": config["registered_inputs"]["prompts_sha256"],
            "target_trace_sha256": config["registered_inputs"][
                "target_trace_sha256"
            ],
            "tokens_per_prompt": token_count,
        },
        "MEASURED": {
            "baseline_case_count": len(baseline_rows),
            "case_count": len(case_rows),
            "baseline_trace_mismatches": baseline_mismatches,
            "sidecar_count": len(sidecars),
            "wall_ns": time.perf_counter_ns() - started,
            "deterministic_core_sha256": core_hash,
        },
        "DERIVED": {
            "parameter_audit": parameter_audit,
            "budget_plans": {key: value.as_dict() for key, value in plans.items()},
            "target_405b_shape_plans": {
                key: value.as_dict() for key, value in target_plans.items()
            },
            "aggregates": aggregates,
            "gate_arm": gate_arm,
            "gate": gates,
            "decision": decision,
        },
        "UNVERIFIED": [
            "a deployable selector that does not compute the complete residual",
            "sound nonlinear propagation from local MLP balls to token margins",
            "joint proof-state correlation across layers and operators",
            "exact fallback cache repair traffic and latency",
            "attention DeltaNet and LM-head operation replacement",
            "ideal-Q4 traffic fidelity to the unchanged BF16 output contract",
            "ground-truth benchmark quality and sampling contracts",
            "sidecar construction amortization and persistent loading",
            "physical CUDA PCIe SSD latency and peak VRAM",
            "122B and arbitrary dense 405B execution",
            "E2 through E7",
        ],
        "claim_boundary": {
            "checkpoint": "UNCHANGED_PINNED_OFFICIAL_BF16_PAYLOAD",
            "pilot": "PROCEDURAL_DCT_IMAGE_DERIVED_AUTOMATICALLY_FROM_EACH_DOWN_MATRIX",
            "selector": "NON_DEPLOYABLE_DUAL_ORACLES_SEPARATELY_OPTIMIZE_QUALITY_AND_PROOF_RADIUS",
            "proof": "LOCAL_EXACT_REAL_ROW_BLOCK_L2_ENCLOSURE_ONLY",
            "operation_replacement": "ALL_MLP_DOWN_OUTPUTS_REPLACED_IN_REFERENCE_FORWARD",
            "traffic": "IDEAL_PACKED_Q4_LOGICAL_ACCOUNTING_NOT_MEASURED_IO",
            "fallback": "REQUIRED_FAIL_CLOSED_BUT_GRANTED_FREE_AND_NOT_EXECUTED",
            "physical_speed": "NOT_MEASURED",
            "target_server": "NO_COMMAND_EXECUTED",
        },
        "provenance": environment,
    }

    dump_rows(output / "raw/baseline_rows.jsonl", baseline_rows)
    dump_rows(output / "raw/case_rows.jsonl", case_rows)
    dump_rows(output / "raw/sidecar_rows.jsonl", sidecar_rows)
    dump(output / "raw/checkpoint_manifest.json", inputs["checkpoint_rows"])
    dump(output / "raw/input_audit.json", inputs["join_audit"])
    dump(output / "processed/aggregate.json", aggregates)
    dump(
        output / "processed/budget_plans.json",
        {
            "small_checkpoint": {
                key: value.as_dict() for key, value in plans.items()
            },
            "target_405b_shape_projection": {
                key: value.as_dict() for key, value in target_plans.items()
            },
        },
    )
    dump(output / "artifacts/environment.json", environment)
    (output / "artifacts/contract.txt").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "logs/run.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8", newline="\n"
    )
    dump(output / "summary.json", summary)
    write_checksums(output)
    print(
        json.dumps(
            {
                "decision": decision,
                "gate": gates,
                "p50_evaluation": aggregates["p50"]["evaluation"],
                "p95_evaluation": aggregates["p95"]["evaluation"],
                "deterministic_core_sha256": core_hash,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
