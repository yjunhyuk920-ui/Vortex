#!/usr/bin/env python3
"""Run EXP-048 causal block-verification amortization audit.

B0 is exact cached greedy with one logical target stream per token.
B1 verifies a perfect future-token proposal and is explicitly non-deployable.
B2 is the charged Jacobi control.
B3 uses only the same checkpoint's first N transformer layers plus final norm
and LM head to generate a causal proposal, followed by one exact full-target
teacher-forced verification pass.

The experiment measures small-checkpoint CPU reference behavior. It is not
405B, 8 GiB, CUDA, PCIe, SSD, TTFT, or tokens/second evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from typing import Any, Sequence

from vortex_runtime.block_verify import (
    ProposalBatch,
    jacobi_generate_exact,
    verify_exact_proposal,
)


ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)
MODEL_PATTERNS = (
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "*.safetensors",
)


class EarlyLayerExit(RuntimeError):
    def __init__(self, hidden: Any) -> None:
        super().__init__("intentional early-layer exit")
        self.hidden = hidden


@dataclass(frozen=True)
class DraftAudit:
    proposal: ProposalBatch
    cpu_elapsed_ns: int
    layer_count: int
    layer_parameter_fraction_per_step: float
    head_and_aux_parameter_fraction_per_step_mean: float


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


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


def resolve_snapshot(
    *,
    model_id: str,
    revision: str,
    cache_dir: Path,
    allow_patterns: Sequence[str],
) -> tuple[Path, list[dict[str, Any]]]:
    from huggingface_hub import model_info, snapshot_download

    info = model_info(model_id, revision=revision)
    if str(info.sha) != revision:
        raise RuntimeError(
            f"resolved revision mismatch for {model_id}: {info.sha} != {revision}"
        )
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=list(allow_patterns),
        )
    )
    return snapshot, snapshot_manifest(snapshot)


def model_parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def exact_cached_greedy(
    *,
    torch: Any,
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
) -> tuple[tuple[int, ...], int]:
    """Generate exact greedy tokens with target KV cache.

    Logical accounting remains one full target weight stream per token even
    though KV cache avoids recomputing previous activations.
    """

    generated: list[int] = []
    start = time.perf_counter_ns()
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        token = output.logits[:, -1, :].argmax(dim=-1)
        for index in range(max_new_tokens):
            generated.append(int(token.item()))
            if index + 1 == max_new_tokens:
                break
            output = model(
                input_ids=token.reshape(1, 1),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = output.past_key_values
            token = output.logits[:, -1, :].argmax(dim=-1)
    return tuple(generated), time.perf_counter_ns() - start


def exact_target_block_predictor(torch: Any, model: Any):
    def predict(
        prefix: tuple[int, ...], proposal: tuple[int, ...]
    ) -> tuple[int, ...]:
        if not prefix or not proposal:
            raise ValueError("prefix and proposal must be non-empty")
        sequence = torch.tensor(
            [prefix + proposal], dtype=torch.long, device=next(model.parameters()).device
        )
        with torch.inference_mode():
            logits = model(
                input_ids=sequence,
                use_cache=False,
                return_dict=True,
            ).logits[0]
        start = len(prefix) - 1
        stop = start + len(proposal)
        predictions = logits[start:stop].argmax(dim=-1)
        if int(predictions.numel()) != len(proposal):
            raise RuntimeError("target block prediction alignment failed")
        return tuple(int(token) for token in predictions.detach().cpu().tolist())

    return predict


def gpt_neo_components(model: Any) -> tuple[Any, Any, Any, Any]:
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None)
    final_norm = getattr(transformer, "ln_f", None)
    output_head = model.get_output_embeddings()
    if transformer is None or blocks is None or final_norm is None or output_head is None:
        raise RuntimeError(
            "EXP-048 B3 currently requires a GPT-Neo-style transformer.h/ln_f layout"
        )
    if len(blocks) <= 0:
        raise RuntimeError("model exposes no transformer blocks")
    return transformer, blocks, final_norm, output_head


def partial_layer_draft(
    *,
    torch: Any,
    model: Any,
    prefix: tuple[int, ...],
    width: int,
    layer_count: int,
    total_parameter_count: int,
) -> DraftAudit:
    """Generate one causal B3 proposal without future target information."""

    transformer, blocks, final_norm, output_head = gpt_neo_components(model)
    if layer_count <= 0 or layer_count > len(blocks):
        raise ValueError("layer_count is outside model depth")
    if width <= 0:
        raise ValueError("width must be positive")

    layer_parameters = sum(
        parameter.numel()
        for block in list(blocks)[:layer_count]
        for parameter in block.parameters()
    )
    head_parameters = sum(parameter.numel() for parameter in output_head.parameters())
    norm_parameters = sum(parameter.numel() for parameter in final_norm.parameters())
    hidden_size = int(getattr(model.config, "hidden_size"))
    layer_fraction = layer_parameters / total_parameter_count
    base_head_fraction = (head_parameters + norm_parameters) / total_parameter_count

    current = list(prefix)
    proposed: list[int] = []
    layer_streams = 0.0
    head_aux_streams = 0.0
    head_aux_step_fractions: list[float] = []
    device = next(model.parameters()).device
    start = time.perf_counter_ns()

    for _ in range(width):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)

        def stop_after_layer(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            raise EarlyLayerExit(hidden)

        handle = blocks[layer_count - 1].register_forward_hook(stop_after_layer)
        try:
            with torch.inference_mode():
                model(input_ids=input_ids, use_cache=False, return_dict=True)
        except EarlyLayerExit as captured:
            hidden = captured.hidden
        else:
            raise RuntimeError("early-layer hook did not terminate model forward")
        finally:
            handle.remove()

        with torch.inference_mode():
            final_hidden = final_norm(hidden[:, -1:, :])
            logits = output_head(final_hidden)[:, -1, :]
            token = int(logits.argmax(dim=-1).item())
        proposed.append(token)
        current.append(token)

        # The transformer internally gathers one token-embedding and one
        # position-embedding row for every sequence position in this reference
        # forward. Charge those scalar reads in addition to full LM-head/final
        # norm reads. Attention/KV arithmetic is recorded by elapsed time but is
        # not mislabeled as weight-stream traffic.
        embedding_lookup_elements = 2 * len(current[:-1]) * hidden_size
        lookup_fraction = embedding_lookup_elements / total_parameter_count
        step_head_aux_fraction = base_head_fraction + lookup_fraction
        layer_streams += layer_fraction
        head_aux_streams += step_head_aux_fraction
        head_aux_step_fractions.append(step_head_aux_fraction)

    elapsed = time.perf_counter_ns() - start
    proposal = ProposalBatch(
        tokens=tuple(proposed),
        draft_layer_equivalent_streams=layer_streams,
        draft_output_head_equivalent_streams=head_aux_streams,
        draft_steps=width,
        future_information_used=False,
        label=f"B3_partial_{layer_count}_layers",
    )
    proposal.validate(width)
    return DraftAudit(
        proposal=proposal,
        cpu_elapsed_ns=elapsed,
        layer_count=layer_count,
        layer_parameter_fraction_per_step=layer_fraction,
        head_and_aux_parameter_fraction_per_step_mean=(
            statistics.mean(head_aux_step_fractions)
        ),
    )


def run_case(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    model_record: dict[str, Any],
    prompt_record: dict[str, Any],
    prompt_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    encoded = tokenizer(
        str(prompt_record["text"]),
        return_tensors="pt",
        truncation=True,
        max_length=int(config["max_input_tokens"]),
    )
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    prefix = tuple(int(token) for token in input_ids[0].detach().cpu().tolist())
    if not prefix:
        raise RuntimeError("tokenizer produced an empty prefix")

    oracle_width = int(config["oracle_block_size"])
    deployable_width = int(config["deployable_block_size"])
    reference, baseline_elapsed = exact_cached_greedy(
        torch=torch,
        model=model,
        input_ids=input_ids,
        max_new_tokens=oracle_width,
    )
    target_predictor = exact_target_block_predictor(torch, model)

    b1_start = time.perf_counter_ns()
    b1_targets = target_predictor(prefix, reference)
    b1_elapsed = time.perf_counter_ns() - b1_start
    b1_verification = verify_exact_proposal(reference, b1_targets)
    if b1_verification.matching_prefix != oracle_width:
        raise RuntimeError("B1 perfect oracle failed exact target block verification")

    b2_start = time.perf_counter_ns()
    b2 = jacobi_generate_exact(
        prefix,
        max_new_tokens=deployable_width,
        block_size=deployable_width,
        max_iterations=int(config["jacobi_max_iterations"]),
        fill_token=int(config["jacobi_fill_token"]),
        target_block_predictor=target_predictor,
    )
    b2_elapsed = time.perf_counter_ns() - b2_start
    b2_exact = b2.generated_tokens == reference[:deployable_width]
    if not b2_exact:
        raise RuntimeError("B2 Jacobi output diverged from exact greedy reference")

    total_parameters = int(model_record["parameter_count"])
    variants: list[dict[str, Any]] = []
    valid_layer_counts = sorted(
        {
            int(layer_count)
            for layer_count in config["partial_layer_counts"]
            if 0 < int(layer_count) <= int(model_record["layer_count"])
        }
    )
    if not valid_layer_counts:
        raise RuntimeError("no valid B3 partial-layer count for model")

    for layer_count in valid_layer_counts:
        draft = partial_layer_draft(
            torch=torch,
            model=model,
            prefix=prefix,
            width=deployable_width,
            layer_count=layer_count,
            total_parameter_count=total_parameters,
        )
        target_start = time.perf_counter_ns()
        target_tokens = target_predictor(prefix, draft.proposal.tokens)
        target_elapsed = time.perf_counter_ns() - target_start
        verification = verify_exact_proposal(draft.proposal.tokens, target_tokens)
        expected = reference[: len(verification.committed_tokens)]
        exact_match = verification.committed_tokens == expected
        if not exact_match:
            raise RuntimeError(
                f"B3 exact correction contract failed at {layer_count} layers"
            )
        committed = len(verification.committed_tokens)
        target_equivalent_streams = 1.0 + draft.proposal.draft_target_equivalent_streams
        variants.append(
            {
                "layer_count": layer_count,
                "layer_fraction_of_depth": layer_count / model_record["layer_count"],
                "proposal_tokens": deployable_width,
                "matching_prefix": verification.matching_prefix,
                "committed_tokens": committed,
                "correction_used": verification.correction_used,
                "rejected_scored_positions": verification.rejected_scored_positions,
                "exact_output_match": exact_match,
                "future_information_used": False,
                "target_full_streams": 1.0,
                "draft_layer_equivalent_streams": (
                    draft.proposal.draft_layer_equivalent_streams
                ),
                "draft_output_head_and_aux_equivalent_streams": (
                    draft.proposal.draft_output_head_equivalent_streams
                ),
                "target_equivalent_streams": target_equivalent_streams,
                "target_equivalent_stream_fraction": (
                    target_equivalent_streams / committed
                ),
                "layer_parameter_fraction_per_draft_step": (
                    draft.layer_parameter_fraction_per_step
                ),
                "head_and_aux_parameter_fraction_per_draft_step_mean": (
                    draft.head_and_aux_parameter_fraction_per_step_mean
                ),
                "draft_steps": draft.proposal.draft_steps,
                "draft_cpu_elapsed_ns": draft.cpu_elapsed_ns,
                "target_verify_cpu_elapsed_ns": target_elapsed,
            }
        )

    best_variant = min(
        variants,
        key=lambda row: (
            row["target_equivalent_stream_fraction"],
            -row["committed_tokens"],
            row["layer_count"],
        ),
    )
    return {
        "model_id": model_record["model_id"],
        "model_revision": model_record["resolved_revision"],
        "parameter_count": total_parameters,
        "layer_count": model_record["layer_count"],
        "hidden_size": model_record["hidden_size"],
        "prompt_index": prompt_index,
        "prompt_family": prompt_record["family"],
        "prompt_sha256": sha256_bytes(str(prompt_record["text"]).encode("utf-8")),
        "input_tokens": len(prefix),
        "B0_exact_sequential": {
            "generated_tokens": oracle_width,
            "logical_target_full_streams": float(oracle_width),
            "target_equivalent_stream_fraction": 1.0,
            "exact_output_reference": True,
            "future_information_used": False,
            "cpu_elapsed_ns": baseline_elapsed,
        },
        "B1_perfect_future_oracle": {
            "proposal_tokens": oracle_width,
            "matching_prefix": b1_verification.matching_prefix,
            "committed_tokens": len(b1_verification.committed_tokens),
            "target_full_streams": 1.0,
            "target_equivalent_stream_fraction": 1.0 / oracle_width,
            "exact_output_match": True,
            "future_information_used": True,
            "deployable": False,
            "cpu_elapsed_ns": b1_elapsed,
        },
        "B2_jacobi": {
            "generated_tokens": b2.generated_count,
            "target_full_streams": b2.target_full_streams,
            "accepted_tokens_per_target_verification": (
                b2.accepted_tokens_per_target_verification
            ),
            "target_equivalent_stream_fraction": (
                b2.target_equivalent_streams_per_accepted_token
            ),
            "max_matching_prefix": b2.max_matching_prefix,
            "mean_matching_prefix": b2.mean_matching_prefix,
            "exact_output_match": b2_exact,
            "future_information_used": False,
            "cpu_elapsed_ns": b2_elapsed,
        },
        "B3_partial_layer_self_draft": variants,
        "B3_best_pre_registered_variant": best_variant,
        "future_generated_tokens_used_deployable": False,
        "real_operation_replacement": False,
        "evidence_ceiling": "E1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "exp_048" / "config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "exp_048_candidate",
    )
    args = parser.parse_args()
    config_raw = args.config.read_bytes()
    config = json.loads(config_raw)
    config_sha256 = sha256_bytes(config_raw)

    output = args.output
    raw_dir = output / "raw"
    processed_dir = output / "processed"
    logs_dir = output / "logs"
    artifacts_dir = output / "artifacts"
    for directory in (raw_dir, processed_dir, logs_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.manual_seed(int(config["seed"]))
    cache_dir = Path(os.environ.get("HF_HOME", output / "hf_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_spec = config["tokenizer"]
    tokenizer_path, tokenizer_manifest = resolve_snapshot(
        model_id=str(tokenizer_spec["model_id"]),
        revision=str(tokenizer_spec["revision"]),
        cache_dir=cache_dir,
        allow_patterns=TOKENIZER_PATTERNS,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )

    cases: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    for model_index, model_spec in enumerate(config["models"]):
        model_path, manifest = resolve_snapshot(
            model_id=str(model_spec["model_id"]),
            revision=str(model_spec["revision"]),
            cache_dir=cache_dir,
            allow_patterns=MODEL_PATTERNS,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
        )
        model.eval()
        _, blocks, _, output_head = gpt_neo_components(model)
        record = {
            "model_id": str(model_spec["model_id"]),
            "resolved_revision": str(model_spec["revision"]),
            "parameter_count": model_parameter_count(model),
            "layer_count": len(blocks),
            "hidden_size": int(getattr(model.config, "hidden_size")),
            "vocab_size": int(getattr(model.config, "vocab_size")),
            "output_head_parameters": int(
                sum(parameter.numel() for parameter in output_head.parameters())
            ),
            "snapshot_manifest": manifest,
        }
        model_records.append(record)
        for prompt_local_index, prompt_record in enumerate(config["held_out_prompts"]):
            prompt_index = model_index * len(config["held_out_prompts"]) + prompt_local_index
            case = run_case(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                model_record=record,
                prompt_record=prompt_record,
                prompt_index=prompt_index,
                config=config,
            )
            cases.append(case)
            print(json.dumps({"case": case}, sort_keys=True), flush=True)
        del model

    raw_cases_path = raw_dir / "cases.jsonl"
    raw_cases_path.write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    checkpoint_manifest = {
        "tokenizer": {
            "model_id": tokenizer_spec["model_id"],
            "resolved_revision": tokenizer_spec["revision"],
            "files": tokenizer_manifest,
        },
        "models": model_records,
    }
    (raw_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    best_rows = [case["B3_best_pre_registered_variant"] for case in cases]
    committed = [float(row["committed_tokens"]) for row in best_rows]
    fractions = [float(row["target_equivalent_stream_fraction"]) for row in best_rows]
    exact_mismatches = sum(
        int(not row["exact_output_match"])
        for case in cases
        for row in case["B3_partial_layer_self_draft"]
    )
    deployable_future_uses = sum(
        int(row["future_information_used"])
        for case in cases
        for row in case["B3_partial_layer_self_draft"]
    )
    b1_mismatches = sum(
        int(not case["B1_perfect_future_oracle"]["exact_output_match"])
        for case in cases
    )
    b2_mismatches = sum(
        int(not case["B2_jacobi"]["exact_output_match"])
        for case in cases
    )

    model_acceptance: dict[str, float] = {}
    for record in model_records:
        values = [
            float(case["B3_best_pre_registered_variant"]["committed_tokens"])
            for case in cases
            if case["model_id"] == record["model_id"]
        ]
        model_acceptance[record["model_id"]] = float(statistics.median(values))
    smallest = model_acceptance[model_records[0]["model_id"]]
    largest = model_acceptance[model_records[-1]["model_id"]]
    relative_size_drop = max(0.0, (smallest - largest) / max(smallest, 1.0))

    p50_committed = float(statistics.median(committed))
    p90_fraction = float(percentile(fractions, 0.90))
    gate_cfg = config["early_gate"]
    exact_pass = (
        exact_mismatches <= int(gate_cfg["wrong_output_limit"])
        and b1_mismatches == 0
        and b2_mismatches == 0
    )
    causal_pass = deployable_future_uses <= int(
        gate_cfg["future_information_limit_deployable"]
    )
    acceptance_pass = p50_committed >= float(
        gate_cfg["minimum_p50_committed_tokens_per_target_verification"]
    )
    traffic_pass = p90_fraction <= float(
        gate_cfg["maximum_p90_target_equivalent_stream_fraction"]
    )
    cost_pass = (
        p90_fraction < 1.0
        if bool(gate_cfg["require_cost_below_sequential"])
        else True
    )
    trend_pass = relative_size_drop <= float(
        gate_cfg["maximum_relative_model_size_acceptance_drop"]
    )
    survives = all(
        (exact_pass, causal_pass, acceptance_pass, traffic_pass, cost_pass, trend_pass)
    )
    decision = (
        "CONTINUE_PARTIAL_LAYER_SELF_DRAFT_TO_COMPLETE_GENERATION_PHASE_C"
        if survives
        else str(gate_cfg["decision_on_failure"])
    )

    projection = config["projection"]
    target_full_gib = (
        int(projection["target_parameters"])
        * int(projection["bits_per_weight"])
        / 8.0
        / (2**30)
    )
    baseline_full_gib = (
        int(projection["baseline_parameters"])
        * int(projection["bits_per_weight"])
        / 8.0
        / (2**30)
    )
    allowed_gib = float(projection["target_traffic_multiplier"]) * baseline_full_gib
    required_fraction = allowed_gib / target_full_gib
    perfect_oracle_fraction = 1.0 / int(config["oracle_block_size"])

    aggregate = {
        "best_b3_committed_tokens": committed,
        "best_b3_target_equivalent_stream_fractions": fractions,
        "model_median_best_b3_committed_tokens": model_acceptance,
        "relative_smallest_to_largest_acceptance_drop": relative_size_drop,
    }
    (processed_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "experiment": "EXP-048",
        "name": "causal_block_verification_amortization_gate",
        "git_commit": git_commit(),
        "workflow_run": os.environ.get("GITHUB_RUN_ID"),
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "complete_real_operation_replacement": False,
        "phase_d_status": "NOT TESTED",
        "config_sha256": config_sha256,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "MEASURED": {
            "model_count": len(model_records),
            "case_count": len(cases),
            "prompt_family_count": len(config["held_out_prompts"]),
            "exact_output_mismatches_b3": exact_mismatches,
            "exact_output_mismatches_b1": b1_mismatches,
            "exact_output_mismatches_b2": b2_mismatches,
            "deployable_future_information_uses": deployable_future_uses,
            "b1_oracle_block_size": int(config["oracle_block_size"]),
            "b1_oracle_target_equivalent_stream_fraction": perfect_oracle_fraction,
            "b3_best_p50_committed_tokens_per_target_verification": p50_committed,
            "b3_best_p90_target_equivalent_stream_fraction": p90_fraction,
            "b3_model_median_committed_tokens": model_acceptance,
            "b3_relative_smallest_to_largest_acceptance_drop": relative_size_drop,
            "models": model_records,
        },
        "DERIVED": {
            "exact_gate_pass": exact_pass,
            "causal_gate_pass": causal_pass,
            "acceptance_gate_pass": acceptance_pass,
            "traffic_gate_pass": traffic_pass,
            "cost_below_sequential_gate_pass": cost_pass,
            "model_size_trend_gate_pass": trend_pass,
            "partial_layer_self_draft_survives_early_gate": survives,
            "decision": decision,
            "b1_is_non_deployable_future_oracle": True,
        },
        "PROJECTED": {
            "target_q4_full_weight_gib_per_stream": target_full_gib,
            "baseline_q4_full_weight_gib_per_stream": baseline_full_gib,
            "allowed_1_2x_baseline_gib_per_token": allowed_gib,
            "required_target_equivalent_stream_fraction": required_fraction,
            "zero_cost_minimum_accepted_tokens_per_target_stream": math.ceil(
                1.0 / required_fraction
            ),
            "b1_oracle_fraction_over_required_fraction": (
                perfect_oracle_fraction / required_fraction
            ),
            "b3_p90_fraction_over_required_fraction": p90_fraction / required_fraction,
        },
        "UNVERIFIED": [
            "complete multi-cycle B3 generation runtime",
            "physical target-weight reuse across block positions",
            "accelerator draft and verification kernels",
            "KV rebuild and memory-transfer cost on target hardware",
            "70B and 405B behavior",
            "8 GiB peak VRAM",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "gate": {
            "passes": survives,
            "decision": decision,
            "early_thresholds": gate_cfg,
            "promotion_thresholds": config["promotion_gate"],
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (logs_dir / "run.log").write_text(
        json.dumps(
            {
                "decision": decision,
                "gate_passes": survives,
                "exact_output_mismatches_b3": exact_mismatches,
                "deployable_future_information_uses": deployable_future_uses,
                "phase_d_status": "NOT TESTED",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifacts_dir / "contract.txt").write_text(
        "B1 is a future-aware non-deployable upper bound. B3 uses only the "
        "current exact prefix, same-checkpoint early layers, final norm, and "
        "LM head. One exact target block pass verifies the proposal. This is "
        "small-checkpoint CPU E1 evidence; Phase D is NOT TESTED.\n",
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
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
