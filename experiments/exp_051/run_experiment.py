#!/usr/bin/env python3
"""Run EXP-051 intermediate-layer finalization and tail-skip oracle audit.

CPU reference observation only. Every measured row uses an exact warm greedy
prefix and executes every target block. Stable depths and per-state depth
selection are non-deployable reference labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from vortex_runtime.layer_finalization import (
    LateDecisionResidualChain,
    LayerTraffic,
    analyze_layer_probe,
    fixed_depths,
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
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1),
    )
    return ordered[index]


def snapshot_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": file_path.relative_to(path).as_posix(),
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return rows


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


def module_parameter_bytes(module: Any) -> int:
    return int(
        sum(
            parameter.numel() * parameter.element_size()
            for parameter in module.parameters()
        )
    )


def model_unique_parameter_bytes(model: Any) -> int:
    return int(
        sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
    )


def tensor_tree_bytes(value: Any) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_tree_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_tree_bytes(item) for item in value)
    if hasattr(value, "to_legacy_cache"):
        return tensor_tree_bytes(value.to_legacy_cache())
    return 0


def compile_traffic(model: Any) -> tuple[LayerTraffic, dict[str, Any]]:
    if getattr(model.config, "model_type", None) != "gpt_neo":
        raise RuntimeError("EXP-051 reference currently requires model_type=gpt_neo")
    transformer = model.transformer
    blocks = tuple(module_parameter_bytes(block) for block in transformer.h)
    final_norm_bytes = module_parameter_bytes(transformer.ln_f)
    lm_head_bytes = module_parameter_bytes(model.lm_head)
    if lm_head_bytes <= 0:
        # Tied weights can be omitted by recursive parameter iteration in some
        # wrappers; logical projection still reads the complete weight matrix.
        weight = model.lm_head.weight
        lm_head_bytes = int(weight.numel() * weight.element_size())
        bias = getattr(model.lm_head, "bias", None)
        if bias is not None:
            lm_head_bytes += int(bias.numel() * bias.element_size())
    token_row = int(
        transformer.wte.weight.shape[1]
        * transformer.wte.weight.element_size()
    )
    position_row = int(
        transformer.wpe.weight.shape[1]
        * transformer.wpe.weight.element_size()
    )
    traffic = LayerTraffic(
        block_parameter_bytes=blocks,
        embedding_row_bytes=token_row + position_row,
        final_norm_bytes=final_norm_bytes,
        lm_head_bytes=lm_head_bytes,
    )
    traffic.validate()
    return traffic, {
        "block_parameter_bytes": list(blocks),
        "embedding_token_row_bytes": token_row,
        "embedding_position_row_bytes": position_row,
        "embedding_rows_total_bytes_per_warm_state": token_row + position_row,
        "final_norm_bytes": final_norm_bytes,
        "lm_head_logical_bytes": lm_head_bytes,
        "full_logical_bytes_per_warm_token": traffic.full_logical_bytes,
        "lm_head_fraction": traffic.lm_head_fraction,
        "unique_checkpoint_parameter_bytes": model_unique_parameter_bytes(model),
        "fixed_depths": list(fixed_depths(len(blocks))),
    }


def probe_depths(
    *,
    torch: Any,
    model: Any,
    outputs: Any,
    traffic: LayerTraffic,
    reconstruction_atol: float,
) -> tuple[dict[str, Any], int]:
    hidden_states = outputs.hidden_states
    block_count = traffic.block_count
    if hidden_states is None or len(hidden_states) != block_count + 1:
        raise RuntimeError(
            f"expected {block_count + 1} hidden states, got "
            f"{None if hidden_states is None else len(hidden_states)}"
        )
    exact_logits = outputs.logits[:, -1, :]
    exact_token = int(exact_logits.argmax(dim=-1).item())
    intermediate_tokens: list[int] = []
    margins: list[float] = []
    probe_elapsed = 0
    final_reconstruction_error = math.inf

    with torch.inference_mode():
        for depth, state in enumerate(hidden_states):
            current = state[:, -1, :]
            started = time.perf_counter_ns()
            if depth < block_count:
                current = model.transformer.ln_f(current)
            logits = model.lm_head(current)
            probe_elapsed += time.perf_counter_ns() - started
            if not bool(torch.isfinite(logits).all().item()):
                raise RuntimeError("intermediate logits contain NaN or Inf")
            values, indices = torch.topk(logits[0], k=2)
            intermediate_tokens.append(int(indices[0].item()))
            margins.append(float((values[0] - values[1]).item()))
            if depth == block_count:
                final_reconstruction_error = float(
                    (logits - exact_logits).abs().max().item()
                )

    final_token_match = intermediate_tokens[-1] == exact_token
    reconstruction_match = (
        final_token_match
        and math.isfinite(final_reconstruction_error)
        and final_reconstruction_error <= reconstruction_atol
    )
    analysis = analyze_layer_probe(
        intermediate_tokens=intermediate_tokens,
        margins=margins,
        traffic=traffic,
    )
    fixed = {
        str(depth): intermediate_tokens[depth] == exact_token
        for depth in fixed_depths(block_count)
    }
    return (
        {
            "exact_token": exact_token,
            "intermediate_tokens": intermediate_tokens,
            "margins": margins,
            "final_reconstruction_max_abs_error": final_reconstruction_error,
            "final_reconstruction_match": reconstruction_match,
            "first_match_depth": analysis.first_match_depth,
            "suffix_stable_depth": analysis.suffix_stable_depth,
            "first_match_block_fraction": analysis.first_match_block_fraction,
            "suffix_stable_block_fraction": analysis.suffix_stable_block_fraction,
            "suffix_stable_logical_byte_fraction": (
                analysis.suffix_stable_logical_byte_fraction
            ),
            "post_first_match_wrong_depths": (
                analysis.post_first_match_wrong_depths
            ),
            "token_changes": analysis.token_changes,
            "fixed_depth_exact": fixed,
            "future_generated_tokens_used": False,
        },
        probe_elapsed,
    )


def run_prompt(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    model_record: dict[str, Any],
    prompt_record: dict[str, Any],
    prompt_index: int,
    traffic: LayerTraffic,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded = tokenizer(
        str(prompt_record["text"]),
        return_tensors="pt",
        truncation=True,
        max_length=int(config["max_input_tokens"]),
    )
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    prefix = tuple(int(token) for token in input_ids[0].detach().cpu().tolist())
    if not prefix:
        raise RuntimeError("tokenizer produced empty prefix")
    warmup = int(config["warmup_generated_tokens"])
    measured = int(config["measured_new_tokens"])
    position_limit = int(model_record["position_limit"])
    if len(prefix) + warmup + measured > position_limit:
        return [], {
            "excluded": True,
            "reason": "context_limit",
            "prefix_tokens": len(prefix),
            "position_limit": position_limit,
        }

    reconstruction_atol = float(config.get("final_logit_reconstruction_atol", 1e-5))
    rows: list[dict[str, Any]] = []
    full_forward_elapsed = 0
    probe_elapsed = 0
    kv_peak = 0

    with torch.inference_mode():
        started = time.perf_counter_ns()
        output = model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        full_forward_elapsed += time.perf_counter_ns() - started
        past = output.past_key_values
        kv_peak = max(kv_peak, tensor_tree_bytes(past))
        current = output.logits[:, -1, :].argmax(dim=-1)

        for _ in range(max(0, warmup - 1)):
            started = time.perf_counter_ns()
            output = model(
                input_ids=current.reshape(1, 1),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            full_forward_elapsed += time.perf_counter_ns() - started
            past = output.past_key_values
            kv_peak = max(kv_peak, tensor_tree_bytes(past))
            current = output.logits[:, -1, :].argmax(dim=-1)

        for token_index in range(measured):
            input_token = int(current.item())
            started = time.perf_counter_ns()
            output = model(
                input_ids=current.reshape(1, 1),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            full_forward_elapsed += time.perf_counter_ns() - started
            past = output.past_key_values
            kv_peak = max(kv_peak, tensor_tree_bytes(past))
            probe, elapsed = probe_depths(
                torch=torch,
                model=model,
                outputs=output,
                traffic=traffic,
                reconstruction_atol=reconstruction_atol,
            )
            probe_elapsed += elapsed
            row = {
                "model_id": model_record["model_id"],
                "model_revision": model_record["resolved_revision"],
                "prompt_index": prompt_index,
                "prompt_family": prompt_record["family"],
                "prompt_sha256": sha256_bytes(
                    str(prompt_record["text"]).encode("utf-8")
                ),
                "prefix_input_tokens": len(prefix),
                "warmup_generated_tokens": warmup,
                "measured_token_index": token_index,
                "current_input_token": input_token,
                "block_count": traffic.block_count,
                **probe,
            }
            rows.append(row)
            current = output.logits[:, -1, :].argmax(dim=-1)

    return rows, {
        "excluded": False,
        "prefix_tokens": len(prefix),
        "measured_rows": len(rows),
        "full_target_forward_elapsed_ns": full_forward_elapsed,
        "intermediate_probe_elapsed_ns": probe_elapsed,
        "kv_peak_bytes": kv_peak,
    }


def aggregate(
    *,
    rows: list[dict[str, Any]],
    prompt_runs: list[dict[str, Any]],
    traffic_records: dict[str, dict[str, Any]],
    adversary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("no token-state rows")
    stable_bytes = [
        float(row["suffix_stable_logical_byte_fraction"]) for row in rows
    ]
    stable_blocks = [
        float(row["suffix_stable_block_fraction"]) for row in rows
    ]
    first_blocks = [float(row["first_match_block_fraction"]) for row in rows]
    reconstruction_mismatches = sum(
        not bool(row["final_reconstruction_match"]) for row in rows
    )
    future_uses = sum(bool(row["future_generated_tokens_used"]) for row in rows)
    transient_rows = sum(int(row["post_first_match_wrong_depths"]) > 0 for row in rows)
    maximum_reconstruction_error = max(
        float(row["final_reconstruction_max_abs_error"]) for row in rows
    )

    model_medians: dict[str, float] = {}
    model_byte_medians: dict[str, float] = {}
    for model_id in sorted({str(row["model_id"]) for row in rows}):
        selected = [row for row in rows if row["model_id"] == model_id]
        model_medians[model_id] = statistics.median(
            float(row["suffix_stable_block_fraction"]) for row in selected
        )
        model_byte_medians[model_id] = statistics.median(
            float(row["suffix_stable_logical_byte_fraction"]) for row in selected
        )

    family_medians: dict[str, float] = {}
    for family in sorted({str(row["prompt_family"]) for row in rows}):
        family_medians[family] = statistics.median(
            float(row["suffix_stable_block_fraction"])
            for row in rows
            if row["prompt_family"] == family
        )

    fixed_accuracy: dict[str, dict[str, float]] = {}
    for model_id in sorted({str(row["model_id"]) for row in rows}):
        model_rows = [row for row in rows if row["model_id"] == model_id]
        depths = sorted(
            {depth for row in model_rows for depth in row["fixed_depth_exact"]},
            key=int,
        )
        fixed_accuracy[model_id] = {
            depth: sum(bool(row["fixed_depth_exact"][depth]) for row in model_rows)
            / len(model_rows)
            for depth in depths
        }

    fixed_family_accuracy: dict[str, dict[str, float]] = {}
    for family in sorted({str(row["prompt_family"]) for row in rows}):
        family_rows = [row for row in rows if row["prompt_family"] == family]
        depths = sorted(
            {depth for row in family_rows for depth in row["fixed_depth_exact"]},
            key=int,
        )
        fixed_family_accuracy[family] = {
            depth: sum(bool(row["fixed_depth_exact"][depth]) for row in family_rows)
            / len(family_rows)
            for depth in depths
        }

    model_order = [str(item["model_id"]) for item in config["models"]]
    smallest = model_medians[model_order[0]]
    largest = model_medians[model_order[-1]]
    if smallest == 0.0:
        size_ratio = 1.0 if largest == 0.0 else math.inf
    else:
        size_ratio = largest / smallest

    gate_cfg = config["early_gate"]
    median_byte = statistics.median(stable_bytes)
    p90_byte = percentile(stable_bytes, 0.90)
    median_block = statistics.median(stable_blocks)
    reconstruction_pass = reconstruction_mismatches <= int(
        gate_cfg["final_reconstruction_mismatch_limit"]
    )
    future_pass = future_uses <= int(gate_cfg["future_generated_token_use_limit"])
    median_byte_pass = median_byte <= float(
        gate_cfg["maximum_median_stable_logical_byte_fraction"]
    )
    p90_byte_pass = p90_byte <= float(
        gate_cfg["maximum_p90_stable_logical_byte_fraction"]
    )
    median_block_pass = median_block <= float(
        gate_cfg["maximum_median_stable_block_fraction"]
    )
    family_pass = all(
        value <= float(gate_cfg["maximum_family_median_stable_block_fraction"])
        for value in family_medians.values()
    )
    size_pass = size_ratio <= float(
        gate_cfg["maximum_largest_to_smallest_median_fraction_ratio"]
    )
    empirical_survives = all(
        [
            reconstruction_pass,
            future_pass,
            median_byte_pass,
            p90_byte_pass,
            median_block_pass,
            family_pass,
            size_pass,
        ]
    )
    universal_counterexample_succeeds = bool(
        adversary["late_decision_counterexample_succeeds"]
    )
    universal_survives = not universal_counterexample_succeeds

    if not empirical_survives:
        decision = str(gate_cfg["empirical_failure_decision"])
        continue_restricted = False
    elif not universal_survives:
        decision = (
            "REJECT_LAYER_FINALIZATION_TAIL_SKIP_AS_UNIVERSAL_CORE_"
            "CONTINUE_RESTRICTED_ADAPTIVE_CERTIFICATE"
        )
        continue_restricted = True
    else:
        decision = "CONTINUE_LAYER_FINALIZATION_TO_SOUND_TAIL_CERTIFICATE"
        continue_restricted = True

    measured = {
        "model_count": len(model_order),
        "prompt_case_count": len(prompt_runs),
        "token_state_count": len(rows),
        "expected_token_state_count": (
            len(model_order)
            * len(config["held_out_prompts"])
            * int(config["measured_new_tokens"])
        ),
        "excluded_prompt_count": sum(bool(item["excluded"]) for item in prompt_runs),
        "final_reconstruction_mismatches": reconstruction_mismatches,
        "maximum_final_reconstruction_abs_error": maximum_reconstruction_error,
        "future_generated_token_uses": future_uses,
        "first_match_median_block_fraction": statistics.median(first_blocks),
        "suffix_stable_median_block_fraction": median_block,
        "suffix_stable_p90_block_fraction": percentile(stable_blocks, 0.90),
        "suffix_stable_median_logical_byte_fraction": median_byte,
        "suffix_stable_p90_logical_byte_fraction": p90_byte,
        "suffix_stable_minimum_logical_byte_fraction": min(stable_bytes),
        "transient_first_match_state_count": transient_rows,
        "transient_first_match_rate": transient_rows / len(rows),
        "model_median_stable_block_fraction": model_medians,
        "model_median_stable_logical_byte_fraction": model_byte_medians,
        "family_median_stable_block_fraction": family_medians,
        "fixed_depth_accuracy_by_model": fixed_accuracy,
        "fixed_depth_accuracy_by_family": fixed_family_accuracy,
        "target_full_forward_elapsed_ns": sum(
            int(item.get("full_target_forward_elapsed_ns", 0)) for item in prompt_runs
        ),
        "offline_probe_elapsed_ns": sum(
            int(item.get("intermediate_probe_elapsed_ns", 0)) for item in prompt_runs
        ),
        "maximum_kv_peak_bytes": max(
            int(item.get("kv_peak_bytes", 0)) for item in prompt_runs
        ),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "late_decision_counterexample_succeeds": universal_counterexample_succeeds,
        "late_decision_first_match_depth": adversary["first_match_depth"],
        "late_decision_suffix_stable_depth": adversary["suffix_stable_depth"],
        "late_decision_stable_byte_fraction": adversary[
            "suffix_stable_logical_byte_fraction"
        ],
        "traffic_by_model": traffic_records,
    }
    derived = {
        "decision": decision,
        "empirical_oracle_survives_gate": empirical_survives,
        "universal_fixed_depth_survives_gate": universal_survives,
        "continue_restricted_adaptive_certificate": continue_restricted,
        "reconstruction_gate_pass": reconstruction_pass,
        "future_information_gate_pass": future_pass,
        "median_byte_gate_pass": median_byte_pass,
        "p90_byte_gate_pass": p90_byte_pass,
        "median_block_gate_pass": median_block_pass,
        "family_gate_pass": family_pass,
        "model_size_trend_gate_pass": size_pass,
        "largest_to_smallest_median_stable_fraction_ratio": size_ratio,
        "per_state_depth_selection_uses_complete_later_layer_reference": True,
    }
    return {"MEASURED": measured, "DERIVED": derived}


def late_adversary(config: dict[str, Any]) -> dict[str, Any]:
    item = config["late_decision_adversary"]
    chain = LateDecisionResidualChain(
        block_count=int(item["block_count"]),
        early_token=int(item["early_token"]),
        final_token=int(item["final_token"]),
    )
    tokens, margins, states = chain.probe()
    traffic = LayerTraffic(
        block_parameter_bytes=tuple(10 for _ in range(chain.block_count)),
        embedding_row_bytes=1,
        final_norm_bytes=1,
        lm_head_bytes=2,
    )
    analysis = analyze_layer_probe(
        intermediate_tokens=tokens,
        margins=margins,
        traffic=traffic,
    )
    succeeds = (
        analysis.first_match_depth == chain.block_count
        and analysis.suffix_stable_depth == chain.block_count
        and analysis.suffix_stable_logical_byte_fraction == 1.0
    )
    return {
        "block_count": chain.block_count,
        "tokens_by_depth": list(tokens),
        "margins_by_depth": list(margins),
        "hidden_states_by_depth": [list(state) for state in states],
        "first_match_depth": analysis.first_match_depth,
        "suffix_stable_depth": analysis.suffix_stable_depth,
        "suffix_stable_logical_byte_fraction": (
            analysis.suffix_stable_logical_byte_fraction
        ),
        "late_decision_counterexample_succeeds": succeeds,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_checksums(root: Path) -> None:
    lines: list[str] = []
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name != "checksums.sha256"
    ):
        lines.append(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        )
    (root / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/exp_051/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_051_candidate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir
    if output.exists():
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True)

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]) % (2**32))
    cache_dir = Path(os.environ.get("HF_HOME", output / "hf-cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_cfg = config["tokenizer"]
    tokenizer_path, tokenizer_files = resolve_snapshot(
        model_id=str(tokenizer_cfg["model_id"]),
        revision=str(tokenizer_cfg["revision"]),
        cache_dir=cache_dir,
        allow_patterns=TOKENIZER_PATTERNS,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )
    manifest: dict[str, Any] = {
        "tokenizer": {
            "model_id": tokenizer_cfg["model_id"],
            "resolved_revision": tokenizer_cfg["revision"],
            "files": tokenizer_files,
        },
        "models": [],
    }

    all_rows: list[dict[str, Any]] = []
    prompt_runs: list[dict[str, Any]] = []
    traffic_records: dict[str, dict[str, Any]] = {}

    for model_cfg in config["models"]:
        model_id = str(model_cfg["model_id"])
        revision = str(model_cfg["revision"])
        model_path, files = resolve_snapshot(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=MODEL_PATTERNS,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        traffic, traffic_record = compile_traffic(model)
        model_record = {
            "model_id": model_id,
            "resolved_revision": revision,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "parameter_bytes": model_unique_parameter_bytes(model),
            "block_count": traffic.block_count,
            "hidden_size": int(model.config.hidden_size),
            "vocab_size": int(model.config.vocab_size),
            "position_limit": int(
                getattr(
                    model.config,
                    "max_position_embeddings",
                    getattr(model.config, "n_positions", 2048),
                )
            ),
            "files": files,
        }
        manifest["models"].append(model_record)
        traffic_records[model_id] = traffic_record

        for prompt_index, prompt in enumerate(config["held_out_prompts"]):
            rows, run = run_prompt(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                model_record=model_record,
                prompt_record=prompt,
                prompt_index=prompt_index,
                traffic=traffic,
                config=config,
            )
            run.update(
                {
                    "model_id": model_id,
                    "prompt_index": prompt_index,
                    "prompt_family": prompt["family"],
                }
            )
            prompt_runs.append(run)
            all_rows.extend(rows)
            print(
                json.dumps(
                    {
                        "model": model_id,
                        "family": prompt["family"],
                        "rows": len(rows),
                        "excluded": run["excluded"],
                        "median_stable_depth": (
                            statistics.median(
                                row["suffix_stable_depth"] for row in rows
                            )
                            if rows
                            else None
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del model

    adversary = late_adversary(config)
    aggregate_data = aggregate(
        rows=all_rows,
        prompt_runs=prompt_runs,
        traffic_records=traffic_records,
        adversary=adversary,
        config=config,
    )

    target_parameters = int(config["projection"]["target_parameters"])
    baseline_parameters = int(config["projection"]["baseline_parameters"])
    bits = int(config["projection"]["bits_per_weight"])
    multiplier = float(config["projection"]["target_traffic_multiplier"])
    target_gib = target_parameters * bits / 8 / 1024**3
    baseline_gib = baseline_parameters * bits / 8 / 1024**3
    allowed_gib = baseline_gib * multiplier
    required_fraction = allowed_gib / target_gib
    p90_fraction = aggregate_data["MEASURED"][
        "suffix_stable_p90_logical_byte_fraction"
    ]

    summary = {
        "experiment": "EXP-051",
        "name": "oracle_layer_finalization_and_tail_skip_gate",
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "real_skipped_layer_operation_replacement": False,
        "phase_d_status": "NOT TESTED",
        **aggregate_data,
        "PROJECTED": {
            "target_q4_full_weight_gib_per_stream": target_gib,
            "baseline_q4_full_weight_gib_per_stream": baseline_gib,
            "allowed_1_2x_baseline_gib_per_token": allowed_gib,
            "required_target_equivalent_stream_fraction": required_fraction,
            "oracle_p90_projected_405b_gib_per_token": target_gib * p90_fraction,
            "oracle_p90_fraction_over_required": p90_fraction / required_fraction,
        },
        "UNVERIFIED": [
            "causal deployable early-exit selector",
            "sound nonlinear omitted-tail certificate",
            "real skipped target blocks during complete generation",
            "70B and 405B layer-finalization depths",
            "8 GiB peak VRAM",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "real_skipped_layer_operation_replacement": False,
            "cuda_pcie_ssd_ttft_tokens_per_second": "NOT TESTED",
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(args.config),
            "created_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        },
    }

    raw_dir = output / "raw"
    processed_dir = output / "processed"
    logs_dir = output / "logs"
    artifacts_dir = output / "artifacts"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    with (raw_dir / "token_states.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(raw_dir / "checkpoint_manifest.json", manifest)
    write_json(raw_dir / "prompt_runs.json", prompt_runs)
    write_json(raw_dir / "late_decision_adversary.json", adversary)
    write_json(raw_dir / "traffic_by_model.json", traffic_records)
    write_json(processed_dir / "aggregate.json", aggregate_data)
    write_json(output / "summary.json", summary)
    write_json(
        artifacts_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
    )
    (artifacts_dir / "contract.txt").write_text(
        "EXP-051 E1 CPU oracle audit only. Every row uses an exact warm target "
        "prefix and executes every target block. Suffix-stable depth and per-state "
        "depth choice use complete later-layer reference outputs and are not a "
        "deployable selector or certificate. 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, "
        "tokens/second, and real skipped-layer execution are NOT TESTED.\n",
        encoding="utf-8",
    )
    (logs_dir / "run.log").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
