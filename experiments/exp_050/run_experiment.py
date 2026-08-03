#!/usr/bin/env python3
"""Run EXP-050 target-independent external draft advice Gate.

CPU reference evidence only. The runner loads three pinned TinyStories models,
generates each continuation once with KV cache, cross-verifies every other model's
proposal, and preserves every pair/K row. Reference-selected pool winners are
favorable non-deployable upper bounds.
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

from vortex_runtime.block_verify import verify_exact_proposal
from vortex_runtime.external_draft import (
    ExternalDraftCandidate,
    construct_first_token_counterexample,
    dynamic_minimum_exact_prefix,
    evaluate_external_draft,
    favorable_external_draft,
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


def sha256_tokens(tokens: Sequence[int]) -> str:
    return sha256_bytes(
        ",".join(str(int(token)) for token in tokens).encode("utf-8")
    )


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


def model_parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def model_parameter_bytes(model: Any) -> int:
    return int(
        sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
    )


def cached_greedy(
    *,
    torch: Any,
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    generated: list[int] = []
    forward_count = 0
    kv_peak = 0
    started = time.perf_counter_ns()
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
        forward_count += 1
        past = output.past_key_values
        kv_peak = max(kv_peak, tensor_tree_bytes(past))
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
            forward_count += 1
            past = output.past_key_values
            kv_peak = max(kv_peak, tensor_tree_bytes(past))
            token = output.logits[:, -1, :].argmax(dim=-1)
    elapsed = time.perf_counter_ns() - started
    if forward_count != max_new_tokens:
        raise RuntimeError(
            f"cached generation forward count {forward_count} != {max_new_tokens}"
        )
    return {
        "tokens": tuple(generated),
        "forward_count": forward_count,
        "elapsed_ns": elapsed,
        "kv_peak_bytes": kv_peak,
    }


def target_block_predictions(
    *,
    torch: Any,
    model: Any,
    prefix: tuple[int, ...],
    proposal: tuple[int, ...],
) -> tuple[tuple[int, ...], int]:
    if not prefix or not proposal:
        raise ValueError("prefix and proposal must be non-empty")
    device = next(model.parameters()).device
    sequence = torch.tensor(
        [prefix + proposal], dtype=torch.long, device=device
    )
    started = time.perf_counter_ns()
    with torch.inference_mode():
        logits = model(
            input_ids=sequence,
            use_cache=False,
            return_dict=True,
        ).logits[0]
    elapsed = time.perf_counter_ns() - started
    start = len(prefix) - 1
    predictions = logits[start : start + len(proposal)].argmax(dim=-1)
    if int(predictions.numel()) != len(proposal):
        raise RuntimeError("target block alignment failed")
    return (
        tuple(int(token) for token in predictions.detach().cpu().tolist()),
        elapsed,
    )


def row_from_result(
    *,
    result: Any,
    proposal_tokens: Sequence[int],
    target_predictions: Sequence[int],
    target_reference: Sequence[int],
    draft_generation: dict[str, Any],
    target_verify_elapsed_ns: int,
) -> dict[str, Any]:
    return {
        "target_id": result.target_id,
        "draft_id": result.draft_id,
        "block_size": result.block_size,
        "matching_prefix": result.matching_prefix,
        "exact_committed_tokens": result.exact_committed_tokens,
        "correction_used": result.verification.correction_used,
        "rejected_positions": result.verification.rejected_scored_positions,
        "exact_output_match": result.exact_output_match,
        "target_future_information_used": (
            result.target_future_information_used
        ),
        "draft_parameter_ratio": result.draft_parameter_ratio,
        "draft_forward_count_charged": result.block_size,
        "draft_generation_full_forward_count": draft_generation[
            "forward_count"
        ],
        "draft_generation_elapsed_ns": draft_generation["elapsed_ns"],
        "draft_kv_peak_bytes": draft_generation["kv_peak_bytes"],
        "target_verification_count": 1,
        "target_verification_elapsed_ns": target_verify_elapsed_ns,
        "actual_small_model_target_equivalent_fraction": (
            result.actual_small_model_target_equivalent_fraction
        ),
        "normalized_4b_405b_fraction": (
            result.normalized_4b_405b_fraction
        ),
        "projected_dynamic_minimum_exact_prefix": (
            result.projected_dynamic_minimum_exact_prefix
        ),
        "proposal_sha256": sha256_tokens(proposal_tokens),
        "proposal_preview": list(proposal_tokens[:16]),
        "target_predictions_sha256": sha256_tokens(target_predictions),
        "target_reference_sha256": sha256_tokens(
            target_reference[: result.block_size]
        ),
    }


def verify_pair(
    *,
    target_id: str,
    draft_id: str,
    target_bytes: int,
    draft_bytes: int,
    proposal_256: tuple[int, ...],
    target_predictions_256: tuple[int, ...],
    target_reference_256: tuple[int, ...],
    draft_generation: dict[str, Any],
    target_verify_elapsed_ns: int,
    block_sizes: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block_size in block_sizes:
        proposal = proposal_256[:block_size]
        predictions = target_predictions_256[:block_size]
        result = evaluate_external_draft(
            ExternalDraftCandidate(
                target_id=target_id,
                draft_id=draft_id,
                block_size=block_size,
                target_parameter_bytes=target_bytes,
                draft_parameter_bytes=draft_bytes,
                proposal_tokens=proposal,
                target_tokens_under_proposal=predictions,
                target_reference_tokens=target_reference_256,
                draft_forward_count=block_size,
                target_verification_count=1,
                target_future_information_used=False,
            )
        )
        rows.append(
            row_from_result(
                result=result,
                proposal_tokens=proposal,
                target_predictions=predictions,
                target_reference=target_reference_256,
                draft_generation=draft_generation,
                target_verify_elapsed_ns=target_verify_elapsed_ns,
            )
        )
    return rows


def select_favorable(
    rows: Sequence[dict[str, Any]],
    draft_bytes: dict[str, int],
) -> dict[str, Any]:
    proxy_results = []
    row_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        proxy = evaluate_external_draft(
            ExternalDraftCandidate(
                target_id=str(row["target_id"]),
                draft_id=str(row["draft_id"]),
                block_size=int(row["block_size"]),
                target_parameter_bytes=1,
                draft_parameter_bytes=1,
                proposal_tokens=tuple(range(int(row["block_size"]))),
                target_tokens_under_proposal=tuple(
                    list(range(int(row["matching_prefix"])))
                    + [int(row["matching_prefix"]) + 1]
                    + [999999]
                    * max(
                        0,
                        int(row["block_size"])
                        - int(row["matching_prefix"])
                        - 1,
                    )
                )[: int(row["block_size"])],
                target_reference_tokens=tuple(
                    list(range(int(row["matching_prefix"])))
                    + [int(row["matching_prefix"]) + 1]
                    + [999998]
                    * int(row["block_size"])
                ),
                draft_forward_count=int(row["block_size"]),
            )
        )
        # Replace synthetic accounting with measured values while retaining
        # the selector's validated ordering fields.
        proxy = proxy.__class__(
            target_id=proxy.target_id,
            draft_id=proxy.draft_id,
            block_size=proxy.block_size,
            verification=proxy.verification,
            exact_output_match=True,
            matching_prefix=int(row["matching_prefix"]),
            exact_committed_tokens=int(row["exact_committed_tokens"]),
            actual_small_model_target_equivalent_fraction=float(
                row["actual_small_model_target_equivalent_fraction"]
            ),
            normalized_4b_405b_fraction=float(
                row["normalized_4b_405b_fraction"]
            ),
            projected_dynamic_minimum_exact_prefix=int(
                row["projected_dynamic_minimum_exact_prefix"]
            ),
            draft_parameter_ratio=float(row["draft_parameter_ratio"]),
            target_future_information_used=False,
        )
        proxy_results.append(proxy)
        row_by_key[(proxy.draft_id, proxy.block_size)] = row
    chosen = favorable_external_draft(
        proxy_results, draft_parameter_bytes=draft_bytes
    )
    selected = dict(row_by_key[(chosen.draft_id, chosen.block_size)])
    selected["draft_selection_uses_reference"] = True
    return selected


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
        default=ROOT / "experiments/exp_050/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_050_candidate",
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
    models: dict[str, Any] = {}
    model_records: dict[str, dict[str, Any]] = {}
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
        record = {
            "model_id": model_id,
            "resolved_revision": revision,
            "parameter_count": model_parameter_count(model),
            "parameter_bytes": model_parameter_bytes(model),
            "layer_count": int(
                getattr(
                    model.config,
                    "num_layers",
                    getattr(model.config, "num_hidden_layers", 0),
                )
            ),
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
        models[model_id] = model
        model_records[model_id] = record
        manifest["models"].append(record)

    max_new_tokens = int(config["max_new_tokens"])
    block_sizes = tuple(int(value) for value in config["block_sizes"])
    prompts: list[dict[str, Any]] = list(config["held_out_prompts"])
    encoded_prompts: dict[int, tuple[int, ...]] = {}
    generations: dict[tuple[str, int], dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(prompts):
        encoded = tokenizer(
            str(prompt["text"]),
            return_tensors="pt",
            truncation=True,
            max_length=int(config["max_input_tokens"]),
        )
        prefix = tuple(int(token) for token in encoded["input_ids"][0].tolist())
        if not prefix:
            raise RuntimeError("tokenizer produced empty prefix")
        encoded_prompts[prompt_index] = prefix
        for model_id, model in models.items():
            position_limit = int(model_records[model_id]["position_limit"])
            if len(prefix) + max_new_tokens > position_limit:
                exclusions.append(
                    {
                        "model_id": model_id,
                        "prompt_index": prompt_index,
                        "reason": "context_limit",
                        "prefix_tokens": len(prefix),
                        "max_new_tokens": max_new_tokens,
                        "position_limit": position_limit,
                    }
                )
                continue
            input_ids = torch.tensor([prefix], dtype=torch.long)
            generated = cached_greedy(
                torch=torch,
                model=model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
            )
            generations[(model_id, prompt_index)] = generated
            print(
                json.dumps(
                    {
                        "stage": "generation",
                        "model": model_id,
                        "family": prompt["family"],
                        "tokens": max_new_tokens,
                        "elapsed_ns": generated["elapsed_ns"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if exclusions:
        raise RuntimeError(
            f"pre-registered corpus had {len(exclusions)} excluded states"
        )

    pair_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    model_ids = [str(item["model_id"]) for item in config["models"]]
    draft_bytes_map = {
        model_id: int(model_records[model_id]["parameter_bytes"])
        for model_id in model_ids
    }

    for target_id in model_ids:
        target_model = models[target_id]
        target_bytes = int(model_records[target_id]["parameter_bytes"])
        for prompt_index, prompt in enumerate(prompts):
            prefix = encoded_prompts[prompt_index]
            target_reference = tuple(
                generations[(target_id, prompt_index)]["tokens"]
            )
            case_candidates: list[dict[str, Any]] = []
            for draft_id in model_ids:
                if draft_id == target_id:
                    continue
                draft_generation = generations[(draft_id, prompt_index)]
                proposal = tuple(draft_generation["tokens"])
                predictions, verify_elapsed = target_block_predictions(
                    torch=torch,
                    model=target_model,
                    prefix=prefix,
                    proposal=proposal,
                )
                rows = verify_pair(
                    target_id=target_id,
                    draft_id=draft_id,
                    target_bytes=target_bytes,
                    draft_bytes=int(
                        model_records[draft_id]["parameter_bytes"]
                    ),
                    proposal_256=proposal,
                    target_predictions_256=predictions,
                    target_reference_256=target_reference,
                    draft_generation=draft_generation,
                    target_verify_elapsed_ns=verify_elapsed,
                    block_sizes=block_sizes,
                )
                for row in rows:
                    row.update(
                        {
                            "target_revision": model_records[target_id][
                                "resolved_revision"
                            ],
                            "draft_revision": model_records[draft_id][
                                "resolved_revision"
                            ],
                            "prompt_index": prompt_index,
                            "prompt_family": prompt["family"],
                            "prompt_sha256": sha256_bytes(
                                str(prompt["text"]).encode("utf-8")
                            ),
                            "input_tokens": len(prefix),
                        }
                    )
                    pair_rows.append(row)
                    case_candidates.append(row)
            selected = select_favorable(case_candidates, draft_bytes_map)
            case_rows.append(
                {
                    "target_id": target_id,
                    "target_revision": model_records[target_id][
                        "resolved_revision"
                    ],
                    "prompt_index": prompt_index,
                    "prompt_family": prompt["family"],
                    "prompt_sha256": sha256_bytes(
                        str(prompt["text"]).encode("utf-8")
                    ),
                    "input_tokens": len(prefix),
                    "target_reference_sha256": sha256_tokens(
                        target_reference
                    ),
                    "target_reference_preview": list(
                        target_reference[:16]
                    ),
                    "selected_favorable_pool_row": selected,
                    "eligible_draft_ids": [
                        model_id
                        for model_id in model_ids
                        if model_id != target_id
                    ],
                }
            )

            oracle_predictions, oracle_elapsed = target_block_predictions(
                torch=torch,
                model=target_model,
                prefix=prefix,
                proposal=target_reference,
            )
            oracle_verification = verify_exact_proposal(
                target_reference, oracle_predictions
            )
            oracle_exact = (
                oracle_verification.matching_prefix == max_new_tokens
                and oracle_verification.committed_tokens == target_reference
            )
            if not oracle_exact:
                raise RuntimeError("E3 exact target future oracle failed")
            oracle_rows.append(
                {
                    "target_id": target_id,
                    "prompt_index": prompt_index,
                    "prompt_family": prompt["family"],
                    "block_size": max_new_tokens,
                    "matching_prefix": oracle_verification.matching_prefix,
                    "exact_committed_tokens": len(
                        oracle_verification.committed_tokens
                    ),
                    "target_verification_count": 1,
                    "target_verification_elapsed_ns": oracle_elapsed,
                    "normalized_zero_cost_fraction": 1 / max_new_tokens,
                    "target_future_information_used": True,
                    "deployable": False,
                    "exact_output_match": True,
                }
            )

    counter_cfg = config["universal_counterexample"]
    counter_result, adversarial_token = construct_first_token_counterexample(
        draft_first_token=int(counter_cfg["draft_first_token"]),
        vocabulary_size=int(counter_cfg["vocabulary_size"]),
        block_size=int(counter_cfg["block_size"]),
    )
    universal_counterexample = {
        "draft_first_token": int(counter_cfg["draft_first_token"]),
        "adversarial_target_first_token": adversarial_token,
        "matching_prefix": counter_result.matching_prefix,
        "exact_committed_tokens": counter_result.exact_committed_tokens,
        "correction_used": counter_result.verification.correction_used,
        "exact_output_match": counter_result.exact_output_match,
        "counterexample_succeeds": counter_result.matching_prefix == 0,
    }

    selected_rows = [row["selected_favorable_pool_row"] for row in case_rows]
    selected_prefixes = [float(row["matching_prefix"]) for row in selected_rows]
    selected_fractions = [
        float(row["normalized_4b_405b_fraction"])
        for row in selected_rows
    ]
    exact_mismatches = sum(
        not bool(row["exact_output_match"]) for row in pair_rows
    )
    future_uses = sum(
        bool(row["target_future_information_used"]) for row in pair_rows
    )
    p50_prefix = statistics.median(selected_prefixes)
    p90_fraction = percentile(selected_fractions, 0.90)
    maximum_prefix = max(selected_prefixes)
    selection_distribution = Counter(
        str(row["draft_id"]) for row in selected_rows
    )
    block_distribution = Counter(
        str(row["block_size"]) for row in selected_rows
    )

    family_useful: dict[str, bool] = {}
    for family in sorted({str(row["prompt_family"]) for row in case_rows}):
        family_useful[family] = any(
            int(row["selected_favorable_pool_row"]["matching_prefix"]) > 0
            for row in case_rows
            if row["prompt_family"] == family
        )
    all_families_useful = all(family_useful.values())

    model_medians: dict[str, float] = {}
    for target_id in model_ids:
        model_medians[target_id] = statistics.median(
            float(row["selected_favorable_pool_row"]["matching_prefix"])
            for row in case_rows
            if row["target_id"] == target_id
        )
    smallest = model_medians[model_ids[0]]
    largest = model_medians[model_ids[-1]]
    size_trend_pass = largest >= (
        1.0
        - float(
            config["early_gate"][
                "maximum_relative_largest_target_prefix_drop"
            ]
        )
    ) * smallest

    prefix_pass = p50_prefix >= float(
        config["early_gate"]["minimum_p50_matching_prefix"]
    )
    traffic_pass = p90_fraction <= float(
        config["early_gate"]["maximum_p90_normalized_fraction"]
    )
    exact_pass = exact_mismatches <= int(
        config["early_gate"]["exact_mismatch_limit"]
    )
    causal_pass = future_uses <= int(
        config["early_gate"]["target_future_information_limit"]
    )
    universal_pass = not bool(
        universal_counterexample["counterexample_succeeds"]
    )
    survives = all(
        [
            prefix_pass,
            traffic_pass,
            exact_pass,
            causal_pass,
            all_families_useful,
            size_trend_pass,
            universal_pass,
        ]
    )
    decision = (
        "CONTINUE_EXTERNAL_DRAFT_POOL_TO_RESTRICTED_COMPLETE_PHASE_C"
        if survives
        else str(config["early_gate"]["decision_on_failure"])
    )

    projected_target = int(config["projection"]["target_parameters"])
    projected_draft = int(config["projection"]["draft_parameters"])
    baseline_parameters = int(config["projection"]["baseline_parameters"])
    bits = int(config["projection"]["bits_per_weight"])
    multiplier = float(config["projection"]["target_traffic_multiplier"])
    target_gib = projected_target * bits / 8 / 1024**3
    draft_gib = projected_draft * bits / 8 / 1024**3
    baseline_gib = baseline_parameters * bits / 8 / 1024**3
    allowed_gib = baseline_gib * multiplier
    required_fraction = allowed_gib / target_gib
    dynamic_required = dynamic_minimum_exact_prefix(
        target_parameters=projected_target,
        draft_parameters=projected_draft,
        allowed_fraction=required_fraction,
    )

    measured = {
        "model_count": len(model_ids),
        "target_prompt_case_count": len(case_rows),
        "target_draft_pair_count": len(
            {
                (row["target_id"], row["draft_id"], row["prompt_index"])
                for row in pair_rows
            }
        ),
        "pair_block_row_count": len(pair_rows),
        "prompt_family_count": len(prompts),
        "excluded_state_count": len(exclusions),
        "all_pair_exact_output_mismatches": exact_mismatches,
        "all_pair_target_future_information_uses": future_uses,
        "favorable_pool_p50_matching_prefix": p50_prefix,
        "favorable_pool_maximum_matching_prefix": maximum_prefix,
        "favorable_pool_p90_normalized_4b_405b_fraction": p90_fraction,
        "favorable_pool_selected_draft_distribution": dict(
            sorted(selection_distribution.items())
        ),
        "favorable_pool_selected_block_distribution": dict(
            sorted(block_distribution.items())
        ),
        "target_median_matching_prefix": model_medians,
        "family_useful_proposal_acceptance": family_useful,
        "all_families_useful": all_families_useful,
        "universal_first_token_counterexample_succeeds": bool(
            universal_counterexample["counterexample_succeeds"]
        ),
        "universal_counterexample_matching_prefix": int(
            universal_counterexample["matching_prefix"]
        ),
        "E3_exact_future_oracle_failures": sum(
            not bool(row["exact_output_match"]) for row in oracle_rows
        ),
        "peak_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
    }
    derived = {
        "decision": decision,
        "fixed_external_draft_pool_survives_early_gate": survives,
        "prefix_gate_pass": prefix_pass,
        "traffic_gate_pass": traffic_pass,
        "exact_gate_pass": exact_pass,
        "causal_gate_pass": causal_pass,
        "family_coverage_gate_pass": all_families_useful,
        "target_size_trend_gate_pass": size_trend_pass,
        "universal_counterexample_absent_gate_pass": universal_pass,
        "draft_selection_uses_reference": True,
        "projected_dynamic_minimum_exact_prefix_for_4b_draft": dynamic_required,
        "universal_claim_scope": (
            "fixed target-independent external drafting is rejected for the "
            "arbitrary-target exact mission when the first-token counterexample succeeds"
        ),
    }
    summary = {
        "experiment": "EXP-050",
        "name": "target_independent_external_draft_advice_gate",
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "complete_real_operation_replacement": False,
        "phase_d_status": "NOT TESTED",
        "MEASURED": measured,
        "DERIVED": derived,
        "PROJECTED": {
            "target_q4_full_weight_gib_per_stream": target_gib,
            "draft_q4_full_weight_gib_per_stream": draft_gib,
            "allowed_1_2x_baseline_gib_per_token": allowed_gib,
            "required_target_equivalent_stream_fraction": required_fraction,
            "draft_target_parameter_ratio": projected_draft / projected_target,
            "perfect_4b_draft_minimum_exact_prefix": dynamic_required,
            "favorable_pool_p90_fraction_over_required": (
                p90_fraction / required_fraction
            ),
        },
        "UNVERIFIED": [
            "causal deployable external draft selector",
            "complete multi-cycle target operation replacement",
            "combined target draft and KV hot state under 8 GiB",
            "physical target/draft residency and overlap",
            "70B and 405B cross-model exact-prefix behavior",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "complete_real_operation_replacement": False,
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
    with (raw_dir / "pair_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in pair_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (raw_dir / "case_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in case_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(raw_dir / "checkpoint_manifest.json", manifest)
    write_json(raw_dir / "universal_counterexample.json", universal_counterexample)
    write_json(raw_dir / "E3_oracle_rows.json", oracle_rows)
    write_json(raw_dir / "generation_records.json", {
        f"{model_id}|{prompt_index}": {
            "token_sha256": sha256_tokens(record["tokens"]),
            "token_preview": list(record["tokens"][:16]),
            "forward_count": record["forward_count"],
            "elapsed_ns": record["elapsed_ns"],
            "kv_peak_bytes": record["kv_peak_bytes"],
        }
        for (model_id, prompt_index), record in sorted(generations.items())
    })
    write_json(processed_dir / "aggregate.json", {
        "MEASURED": measured,
        "DERIVED": derived,
    })
    write_json(output / "summary.json", summary)
    write_json(artifacts_dir / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
    })
    (artifacts_dir / "contract.txt").write_text(
        "EXP-050 E1 CPU reference only. E1/E2 external drafts use no target "
        "future tokens or target-specific training. Favorable draft selection "
        "uses the exact reference and is non-deployable. E3 is future-aware. "
        "405B, 8 GiB, CUDA, PCIe, SSD, TTFT, tokens/second, combined hot-state "
        "fit, and physical overlap are NOT TESTED.\n",
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
