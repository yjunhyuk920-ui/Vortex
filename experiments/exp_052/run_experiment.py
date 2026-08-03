#!/usr/bin/env python3
"""Run EXP-052 exact target-specific advice tradeoff Gate.

CPU E1 evidence only. The runner computes exact small-checkpoint states first,
then audits exact prefix/state memoization with disjoint leave-one-family-out
folds. Every logical build and fallback target call is charged even though the
corpus is physically generated once for deterministic comparison.
"""

from __future__ import annotations

import argparse
from collections import Counter
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

from vortex_runtime.advice_closure import (
    budget_coverage_audit,
    hot_index_capacity,
    required_exact_repetitions,
    required_hit_rate_for_fraction,
)
from vortex_runtime.exact_advice import (
    AdviceAccounting,
    ExactPrefixAdviceTable,
    ExactStateAdviceTable,
    PrefixAdviceEntry,
    StateAdviceEntry,
    independent_state_audit,
    prefix_digest,
    prefix_universe_size,
    reuse_histogram,
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
    return [
        {
            "path": file_path.relative_to(path).as_posix(),
            "size_bytes": file_path.stat().st_size,
            "sha256": sha256_file(file_path),
        }
        for file_path in sorted(item for item in path.rglob("*") if item.is_file())
    ]


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


def model_parameter_bytes(model: Any) -> int:
    return int(
        sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
    )


def canonical_cache(value: Any) -> Any:
    return value.to_legacy_cache() if hasattr(value, "to_legacy_cache") else value


def hash_cache(value: Any) -> tuple[bytes, bytes, int]:
    """Hash KV state with dtype, shape, tree order, and exact tensor bytes."""

    import torch

    digest256 = hashlib.sha256()
    digest512 = hashlib.sha512()
    total_bytes = 0

    def feed(data: bytes) -> None:
        digest256.update(data)
        digest512.update(data)

    def walk(item: Any) -> None:
        nonlocal total_bytes
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            metadata = json.dumps(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            raw = tensor.numpy().tobytes(order="C")
            feed(b"T" + len(metadata).to_bytes(4, "big") + metadata)
            feed(len(raw).to_bytes(8, "big"))
            feed(raw)
            total_bytes += len(raw)
            return
        if isinstance(item, dict):
            feed(b"D" + len(item).to_bytes(4, "big"))
            for key in sorted(item):
                feed(str(key).encode("utf-8") + b"\0")
                walk(item[key])
            return
        if isinstance(item, (tuple, list)):
            feed(b"L" + len(item).to_bytes(4, "big"))
            for child in item:
                walk(child)
            return
        if item is None:
            feed(b"N")
            return
        raise TypeError(f"unsupported cache item {type(item)!r}")

    walk(canonical_cache(value))
    return digest256.digest(), digest512.digest(), total_bytes


def generate_prompt_states(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    model_record: dict[str, Any],
    prompt: dict[str, Any],
    prompt_index: int,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    encoded = tokenizer(
        str(prompt["text"]),
        return_tensors="pt",
        truncation=True,
        max_length=int(config["max_input_tokens"]),
    )
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    prefix = [int(token) for token in input_ids[0].detach().cpu().tolist()]
    needed = int(config["warmup_tokens"]) + int(
        config["measured_states_per_prompt"]
    )
    if len(prefix) + needed > int(model_record["position_limit"]):
        return None

    rows: list[dict[str, Any]] = []
    forward_count = 0
    peak_kv_bytes = 0
    started = time.perf_counter_ns()
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
        forward_count += 1
        past = output.past_key_values
        token = int(output.logits[:, -1, :].argmax(dim=-1).item())

        for _ in range(int(config["warmup_tokens"])):
            prefix.append(token)
            token_input = torch.tensor(
                [[token]], dtype=torch.long, device=input_ids.device
            )
            output = model(
                input_ids=token_input,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            forward_count += 1
            past = output.past_key_values
            token = int(output.logits[:, -1, :].argmax(dim=-1).item())

        for state_index in range(int(config["measured_states_per_prompt"])):
            state_sha256, state_sha512, state_raw_bytes = hash_cache(past)
            peak_kv_bytes = max(peak_kv_bytes, state_raw_bytes)
            exact_prefix = tuple(prefix)
            rows.append(
                {
                    "model_id": model_record["model_id"],
                    "model_revision": model_record["resolved_revision"],
                    "prompt_index": prompt_index,
                    "prompt_family": prompt["family"],
                    "prompt_sha256": hashlib.sha256(
                        str(prompt["text"]).encode("utf-8")
                    ).hexdigest(),
                    "state_index": state_index,
                    "prefix_tokens": list(exact_prefix),
                    "prefix_sha256": hashlib.sha256(
                        ",".join(str(token) for token in exact_prefix).encode("utf-8")
                    ).hexdigest(),
                    "current_token": exact_prefix[-1],
                    "position": len(exact_prefix) - 1,
                    "next_token": token,
                    "state_sha256": state_sha256.hex(),
                    "state_sha512": state_sha512.hex(),
                    "state_raw_bytes": state_raw_bytes,
                    "future_information_used": False,
                }
            )
            if state_index + 1 == int(config["measured_states_per_prompt"]):
                break
            prefix.append(token)
            token_input = torch.tensor(
                [[token]], dtype=torch.long, device=input_ids.device
            )
            output = model(
                input_ids=token_input,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            forward_count += 1
            past = output.past_key_values
            token = int(output.logits[:, -1, :].argmax(dim=-1).item())

    return {
        "rows": rows,
        "forward_count": forward_count,
        "elapsed_ns": time.perf_counter_ns() - started,
        "peak_kv_bytes": peak_kv_bytes,
        "input_tokens": int(input_ids.numel()),
    }


def prefix_entry(record: dict[str, Any], contract: str) -> PrefixAdviceEntry:
    return PrefixAdviceEntry(
        target_revision=str(record["model_revision"]),
        decode_contract=contract,
        prefix_tokens=tuple(int(token) for token in record["prefix_tokens"]),
        next_token=int(record["next_token"]),
    )


def state_entry(record: dict[str, Any], contract: str) -> StateAdviceEntry:
    return StateAdviceEntry(
        target_revision=str(record["model_revision"]),
        decode_contract=contract,
        state_sha256=bytes.fromhex(str(record["state_sha256"])),
        state_sha512=bytes.fromhex(str(record["state_sha512"])),
        state_raw_bytes=int(record["state_raw_bytes"]),
        current_token=int(record["current_token"]),
        position=int(record["position"]),
        exact_prefix_tokens=tuple(int(token) for token in record["prefix_tokens"]),
        next_token=int(record["next_token"]),
    )


def audit_fold(
    *,
    model_id: str,
    revision: str,
    held_family: str,
    records: Sequence[dict[str, Any]],
    contract: str,
    reuse_checkpoints: Sequence[int],
    observed_reuse: float,
    allowed_fraction: float,
) -> list[dict[str, Any]]:
    build = [row for row in records if row["prompt_family"] != held_family]
    evaluate = [row for row in records if row["prompt_family"] == held_family]
    build_prompts = {int(row["prompt_index"]) for row in build}
    evaluation_prompts = {int(row["prompt_index"]) for row in evaluate}
    prompt_leakage = len(build_prompts & evaluation_prompts)
    result_rows: list[dict[str, Any]] = []

    for condition in ("P0_prefix", "S0_state"):
        if condition == "P0_prefix":
            table: Any = ExactPrefixAdviceTable()
            for row in build:
                table.add(prefix_entry(row, contract))

            def query(row: dict[str, Any]) -> Any:
                return table.query(
                    target_revision=revision,
                    decode_contract=contract,
                    prefix_tokens=tuple(row["prefix_tokens"]),
                )

            advice_bytes = table.serialized_bytes
            raw_witness_bytes = 0
        else:
            table = ExactStateAdviceTable()
            for row in build:
                table.add(state_entry(row, contract))

            def query(row: dict[str, Any]) -> Any:
                return table.query(
                    target_revision=revision,
                    decode_contract=contract,
                    state_sha256=bytes.fromhex(row["state_sha256"]),
                    state_sha512=bytes.fromhex(row["state_sha512"]),
                    current_token=int(row["current_token"]),
                    position=int(row["position"]),
                    exact_prefix_tokens=tuple(row["prefix_tokens"]),
                )

            advice_bytes = table.serialized_bytes_prefix_collision_witness
            raw_witness_bytes = table.serialized_bytes_with_raw_collision_witness

        hits = 0
        wrong_hits = 0
        probes = 0
        for row in evaluate:
            outcome = query(row)
            probes += outcome.probes
            if outcome.hit:
                hits += 1
                wrong_hits += int(outcome.next_token != int(row["next_token"]))
        fallbacks = len(evaluate) - hits
        accounting = AdviceAccounting(
            query_count=len(evaluate),
            advice_hits=hits,
            target_fallback_calls=fallbacks,
            build_target_calls=table.build_target_calls,
            advice_bytes=advice_bytes,
            lookup_probes=probes,
        )
        accounting.validate()
        result_rows.append(
            {
                "model_id": model_id,
                "model_revision": revision,
                "held_out_family": held_family,
                "condition": condition,
                "build_prompt_count": len(build_prompts),
                "evaluation_prompt_count": len(evaluation_prompts),
                "build_state_count": len(build),
                "query_count": len(evaluate),
                "entry_count": table.entry_count,
                "build_target_calls": table.build_target_calls,
                "advice_hits": hits,
                "wrong_hits": wrong_hits,
                "target_fallback_calls": fallbacks,
                "hit_rate": accounting.hit_rate,
                "online_fallback_fraction": accounting.online_target_fallback_fraction,
                "cold_start_target_fraction": accounting.target_forward_component_per_query,
                "observed_reuse": observed_reuse,
                "observed_reuse_fully_accounted_fraction": (
                    accounting.amortized_target_forward_fraction(
                        max(1, int(observed_reuse))
                    )
                ),
                "required_repetitions_for_target": required_exact_repetitions(
                    query_count=len(evaluate),
                    build_target_calls=table.build_target_calls,
                    hit_rate=accounting.hit_rate,
                    allowed_fraction=allowed_fraction,
                ),
                "reuse_checkpoint_fractions": {
                    str(repetition): accounting.amortized_target_forward_fraction(
                        int(repetition)
                    )
                    for repetition in reuse_checkpoints
                },
                "advice_bytes_prefix_witness": advice_bytes,
                "advice_bytes_raw_state_witness": raw_witness_bytes,
                "lookup_probes": probes,
                "mean_lookup_probes": probes / len(evaluate),
                "build_eval_prompt_leakage": prompt_leakage,
                "future_information_used": False,
            }
        )
    return result_rows


def replay_control(
    *,
    model_id: str,
    revision: str,
    records: Sequence[dict[str, Any]],
    contract: str,
    reuse_checkpoints: Sequence[int],
    allowed_fraction: float,
) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for condition in ("P0_prefix", "S0_state"):
        if condition == "P0_prefix":
            table: Any = ExactPrefixAdviceTable()
            for row in records:
                table.add(prefix_entry(row, contract))

            def query(row: dict[str, Any]) -> Any:
                return table.query(
                    target_revision=revision,
                    decode_contract=contract,
                    prefix_tokens=tuple(row["prefix_tokens"]),
                )

            advice_bytes = table.serialized_bytes
        else:
            table = ExactStateAdviceTable()
            for row in records:
                table.add(state_entry(row, contract))

            def query(row: dict[str, Any]) -> Any:
                return table.query(
                    target_revision=revision,
                    decode_contract=contract,
                    state_sha256=bytes.fromhex(row["state_sha256"]),
                    state_sha512=bytes.fromhex(row["state_sha512"]),
                    current_token=int(row["current_token"]),
                    position=int(row["position"]),
                    exact_prefix_tokens=tuple(row["prefix_tokens"]),
                )

            advice_bytes = table.serialized_bytes_prefix_collision_witness

        hits = 0
        wrong_hits = 0
        probes = 0
        for row in records:
            outcome = query(row)
            probes += outcome.probes
            hits += int(outcome.hit)
            wrong_hits += int(
                outcome.hit and outcome.next_token != int(row["next_token"])
            )
        accounting = AdviceAccounting(
            query_count=len(records),
            advice_hits=hits,
            target_fallback_calls=len(records) - hits,
            build_target_calls=table.build_target_calls,
            advice_bytes=advice_bytes,
            lookup_probes=probes,
        )
        controls.append(
            {
                "model_id": model_id,
                "condition": condition,
                "query_count": len(records),
                "entry_count": table.entry_count,
                "hits": hits,
                "wrong_hits": wrong_hits,
                "hit_rate": accounting.hit_rate,
                "fallback_fraction": accounting.online_target_fallback_fraction,
                "build_target_calls": table.build_target_calls,
                "minimum_repetitions_for_target": required_exact_repetitions(
                    query_count=len(records),
                    build_target_calls=table.build_target_calls,
                    hit_rate=accounting.hit_rate,
                    allowed_fraction=allowed_fraction,
                ),
                "reuse_checkpoint_fractions": {
                    str(repetition): accounting.amortized_target_forward_fraction(
                        int(repetition)
                    )
                    for repetition in reuse_checkpoints
                },
            }
        )
    return controls


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
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
        default=ROOT / "experiments/exp_052/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_052_candidate",
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
    import transformers
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
    state_rows: list[dict[str, Any]] = []
    prompt_runs: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    peak_rss_kib = 0

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
        model_record = {
            "model_id": model_id,
            "resolved_revision": revision,
            "parameter_count": model_parameter_count(model),
            "parameter_bytes": model_parameter_bytes(model),
            "position_limit": int(
                getattr(
                    model.config,
                    "max_position_embeddings",
                    getattr(model.config, "n_positions", 2048),
                )
            ),
            "hidden_size": int(model.config.hidden_size),
            "vocab_size": int(model.config.vocab_size),
            "files": files,
        }
        manifest["models"].append(model_record)
        model_rows: list[dict[str, Any]] = []
        for prompt_index, prompt in enumerate(config["held_out_prompts"]):
            run = generate_prompt_states(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                model_record=model_record,
                prompt=prompt,
                prompt_index=prompt_index,
                config=config,
            )
            if run is None:
                exclusions.append(
                    {
                        "model_id": model_id,
                        "prompt_index": prompt_index,
                        "family": prompt["family"],
                        "reason": "context_limit",
                    }
                )
                continue
            model_rows.extend(run["rows"])
            state_rows.extend(run["rows"])
            prompt_runs.append(
                {
                    "model_id": model_id,
                    "prompt_index": prompt_index,
                    "family": prompt["family"],
                    "input_tokens": run["input_tokens"],
                    "measured_states": len(run["rows"]),
                    "physical_target_forward_calls": run["forward_count"],
                    "elapsed_ns": run["elapsed_ns"],
                    "peak_kv_bytes": run["peak_kv_bytes"],
                }
            )
        records_by_model[model_id] = model_rows
        del model
        peak_rss_kib = max(
            peak_rss_kib, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )

    expected_states = (
        len(config["models"])
        * len(config["held_out_prompts"])
        * int(config["measured_states_per_prompt"])
    )
    if not exclusions and len(state_rows) != expected_states:
        raise RuntimeError(
            f"state row count {len(state_rows)} != {expected_states}"
        )

    contract = str(config["decode_contract"])
    allowed_fraction = float(
        config["projection"]["required_target_equivalent_fraction"]
    )
    reuse_points = [int(value) for value in config["reuse_checkpoints"]]
    fold_rows: list[dict[str, Any]] = []
    replay_controls: list[dict[str, Any]] = []
    natural_reuse_values: list[int] = []

    for model_cfg in config["models"]:
        model_id = str(model_cfg["model_id"])
        revision = str(model_cfg["revision"])
        model_rows = records_by_model[model_id]
        keys = [
            prefix_digest(
                target_revision=revision,
                decode_contract=contract,
                tokens=tuple(row["prefix_tokens"]),
            )
            for row in model_rows
        ]
        counts = Counter(keys)
        natural_reuse_values.extend(int(count) for count in counts.values())
        observed_reuse = statistics.median(counts.values()) if counts else 0.0
        for prompt in config["held_out_prompts"]:
            fold_rows.extend(
                audit_fold(
                    model_id=model_id,
                    revision=revision,
                    held_family=str(prompt["family"]),
                    records=model_rows,
                    contract=contract,
                    reuse_checkpoints=reuse_points,
                    observed_reuse=observed_reuse,
                    allowed_fraction=allowed_fraction,
                )
            )
        replay_controls.extend(
            replay_control(
                model_id=model_id,
                revision=revision,
                records=model_rows,
                contract=contract,
                reuse_checkpoints=reuse_points,
                allowed_fraction=allowed_fraction,
            )
        )

    independent_rows: list[dict[str, Any]] = []
    independent_cfg = config["independent_state_audit"]
    for coverage in independent_cfg["coverage_fractions"]:
        audit = independent_state_audit(
            state_count=int(independent_cfg["executable_state_count"]),
            coverage_fraction=float(coverage),
            vocabulary_size=int(independent_cfg["vocabulary_size"]),
            seed=int(config["seed"]),
        )
        independent_rows.append(audit.__dict__)

    sampled_state_entries = [state_entry(row, contract) for row in state_rows]
    sampled_prefix_entries = [prefix_entry(row, contract) for row in state_rows]
    mean_state_entry_bytes = math.ceil(
        statistics.mean(
            entry.serialized_bytes_prefix_collision_witness
            for entry in sampled_state_entries
        )
    )
    mean_raw_state_entry_bytes = math.ceil(
        statistics.mean(
            entry.serialized_bytes_with_raw_collision_witness
            for entry in sampled_state_entries
        )
    )
    mean_prefix_entry_bytes = math.ceil(
        statistics.mean(entry.serialized_bytes for entry in sampled_prefix_entries)
    )

    storage_state_count = int(independent_cfg["storage_adversary_state_count"])
    budgets = config["budgets"]
    cold_audit = budget_coverage_audit(
        state_count=storage_state_count,
        entry_bytes=mean_state_entry_bytes,
        budget_bytes=int(budgets["cold_advice_bytes"]),
    )
    hot_capacity = hot_index_capacity(
        budget_bytes=int(budgets["hot_index_bytes"]),
        slot_bytes=int(budgets["hot_index_slot_bytes"]),
    )
    combined_entries = min(
        storage_state_count, cold_audit.maximum_entries, hot_capacity
    )
    combined_coverage = combined_entries / storage_state_count
    combined_fallback = 1.0 - combined_coverage

    family_hit_rates: dict[str, dict[str, float]] = {}
    families = [str(prompt["family"]) for prompt in config["held_out_prompts"]]
    for condition in ("P0_prefix", "S0_state"):
        family_hit_rates[condition] = {}
        for family in families:
            selected = [
                row
                for row in fold_rows
                if row["condition"] == condition
                and row["held_out_family"] == family
            ]
            query_count = sum(int(row["query_count"]) for row in selected)
            hits = sum(int(row["advice_hits"]) for row in selected)
            family_hit_rates[condition][family] = (
                hits / query_count if query_count else 0.0
            )

    wrong_hits = sum(int(row["wrong_hits"]) for row in fold_rows) + sum(
        int(row["wrong_hits"]) for row in replay_controls
    )
    prompt_leakage = sum(
        int(row["build_eval_prompt_leakage"]) for row in fold_rows
    )
    minimum_family_hit = min(
        rate for condition in family_hit_rates.values() for rate in condition.values()
    )
    maximum_family_fallback = max(
        1.0 - rate
        for condition in family_hit_rates.values()
        for rate in condition.values()
    )
    median_natural_reuse = (
        statistics.median(natural_reuse_values) if natural_reuse_values else 0.0
    )
    observed_fractions = [
        float(row["observed_reuse_fully_accounted_fraction"])
        for row in fold_rows
    ]
    p90_observed_fraction = percentile(observed_fractions, 0.90)

    gate = config["early_gate"]
    exact_gate_pass = wrong_hits <= int(gate["wrong_advice_hit_limit"])
    leakage_gate_pass = prompt_leakage <= int(
        gate["build_eval_leakage_limit"]
    )
    hit_gate_pass = minimum_family_hit >= float(
        gate["minimum_each_family_held_out_hit_rate"]
    )
    fallback_gate_pass = maximum_family_fallback <= float(
        gate["maximum_each_family_online_fallback_fraction"]
    )
    reuse_gate_pass = median_natural_reuse >= float(
        gate["minimum_median_observed_exact_reuse"]
    )
    fraction_gate_pass = p90_observed_fraction <= float(
        gate["maximum_p90_fully_accounted_target_fraction"]
    )
    storage_gate_pass = combined_fallback <= float(
        gate["maximum_one_tib_independent_state_fallback_fraction"]
    )
    survives = all(
        (
            exact_gate_pass,
            leakage_gate_pass,
            hit_gate_pass,
            fallback_gate_pass,
            reuse_gate_pass,
            fraction_gate_pass,
            storage_gate_pass,
        )
    )
    decision = (
        "CONTINUE_ENUMERATIVE_EXACT_ADVICE_TO_OPERATION_REPLACEMENT"
        if survives
        else str(gate["decision_on_failure"])
    )

    maximum_prefix_length = max(len(row["prefix_tokens"]) for row in state_rows)
    prefix_universe = prefix_universe_size(
        int(independent_cfg["vocabulary_size"]), maximum_prefix_length
    )

    measured = {
        "model_count": len(config["models"]),
        "prompt_case_count": len(config["models"])
        * len(config["held_out_prompts"]),
        "state_count": len(state_rows),
        "expected_state_count": expected_states,
        "excluded_prompt_count": len(exclusions),
        "physical_corpus_target_forward_calls": sum(
            int(row["physical_target_forward_calls"]) for row in prompt_runs
        ),
        "fold_row_count": len(fold_rows),
        "wrong_advice_hits": wrong_hits,
        "build_eval_prompt_leakage": prompt_leakage,
        "family_hit_rates": family_hit_rates,
        "minimum_held_out_family_hit_rate": minimum_family_hit,
        "maximum_held_out_family_online_fallback_fraction": (
            maximum_family_fallback
        ),
        "median_natural_exact_reuse": median_natural_reuse,
        "maximum_natural_exact_reuse": max(natural_reuse_values, default=0),
        "p90_observed_reuse_fully_accounted_target_fraction": (
            p90_observed_fraction
        ),
        "replay_all_exact": all(
            row["hit_rate"] == 1.0 and row["wrong_hits"] == 0
            for row in replay_controls
        ),
        "replay_minimum_repetitions": max(
            int(row["minimum_repetitions_for_target"] or 0)
            for row in replay_controls
        ),
        "mean_prefix_entry_bytes": mean_prefix_entry_bytes,
        "mean_state_prefix_witness_entry_bytes": mean_state_entry_bytes,
        "mean_state_raw_witness_entry_bytes": mean_raw_state_entry_bytes,
        "mean_state_raw_kv_bytes": statistics.mean(
            int(row["state_raw_bytes"]) for row in state_rows
        ),
        "peak_state_raw_kv_bytes": max(
            int(row["state_raw_bytes"]) for row in state_rows
        ),
        "mean_lookup_probes": statistics.mean(
            float(row["mean_lookup_probes"]) for row in fold_rows
        ),
        "peak_rss_kib": peak_rss_kib,
        "independent_state_wrong_hits": sum(
            int(row["wrong_hits"]) for row in independent_rows
        ),
    }
    derived = {
        "exact_gate_pass": exact_gate_pass,
        "leakage_gate_pass": leakage_gate_pass,
        "held_out_hit_gate_pass": hit_gate_pass,
        "fallback_gate_pass": fallback_gate_pass,
        "natural_reuse_gate_pass": reuse_gate_pass,
        "fully_accounted_fraction_gate_pass": fraction_gate_pass,
        "storage_gate_pass": storage_gate_pass,
        "enumerative_exact_advice_survives_gate": survives,
        "decision": decision,
        "required_hit_rate_infinite_reuse": required_hit_rate_for_fraction(
            allowed_fraction=allowed_fraction,
            amortized_build_fraction=0.0,
        ),
        "required_hit_rate_at_85_reuses": required_hit_rate_for_fraction(
            allowed_fraction=allowed_fraction,
            amortized_build_fraction=1.0 / 85.0,
        ),
        "hot_index_capacity_entries": hot_capacity,
        "cold_one_tib_capacity_entries": cold_audit.maximum_entries,
        "combined_budget_capacity_entries": combined_entries,
        "combined_budget_coverage_fraction_over_2_48_states": (
            combined_coverage
        ),
        "combined_budget_minimum_fallback_fraction": combined_fallback,
        "maximum_observed_prefix_length": maximum_prefix_length,
        "prefix_universe_bit_length": prefix_universe.bit_length(),
        "prefix_universe_decimal_digits": len(str(prefix_universe)),
        "claim_scope": (
            "rejects enumerative exact prefix/KV-state advice under the "
            "measured and independent-state budgets; not an unconditional "
            "lower bound for every symbolic compiler"
        ),
    }

    summary = {
        "experiment": "EXP-052",
        "name": "runtime_only_exact_advice_tradeoff_and_constraint_closure_gate",
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "complete_real_operation_replacement": False,
        "phase_d_status": "NOT TESTED",
        "MEASURED": measured,
        "DERIVED": derived,
        "PROJECTED": {
            "target_q4_full_weight_gib_per_stream": 405_000_000_000
            * 4
            / 8
            / 2**30,
            "baseline_q4_full_weight_gib_per_stream": 4_000_000_000
            * 4
            / 8
            / 2**30,
            "allowed_1_2x_baseline_gib_per_token": 1.2
            * 4_000_000_000
            * 4
            / 8
            / 2**30,
            "required_target_equivalent_fraction": allowed_fraction,
        },
        "UNVERIFIED": [
            "non-enumerative automatic target program compiler",
            "real unseen-state target operation replacement",
            "physical advice lookup latency and SSD behavior",
            "70B and 405B advice coverage",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "authoritative_decision": decision,
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(args.config),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "raw_evidence": {
            "state_records": "raw/state_records.jsonl",
            "fold_rows": "raw/fold_rows.jsonl",
            "replay_controls": "raw/replay_controls.json",
            "independent_state_audit": "raw/independent_state_audit.json",
            "checkpoint_manifest": "raw/checkpoint_manifest.json",
        },
        "claim_boundary": {
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "complete_real_operation_replacement": False,
            "cuda_pcie_ssd_ttft_tokens_per_second": "NOT TESTED",
        },
    }
    aggregate = {
        "family_hit_rates": family_hit_rates,
        "natural_reuse_histogram": reuse_histogram(
            [
                prefix_digest(
                    target_revision=str(row["model_revision"]),
                    decode_contract=contract,
                    tokens=tuple(row["prefix_tokens"]),
                )
                for row in state_rows
            ]
        ),
        "storage": {
            "mean_prefix_entry_bytes": mean_prefix_entry_bytes,
            "mean_state_prefix_witness_entry_bytes": mean_state_entry_bytes,
            "mean_state_raw_witness_entry_bytes": mean_raw_state_entry_bytes,
            "hot_index_capacity": hot_capacity,
            "cold_capacity": cold_audit.__dict__,
            "combined_entries": combined_entries,
            "combined_coverage": combined_coverage,
            "combined_fallback": combined_fallback,
        },
        "gate": derived,
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "peak_rss_kib": peak_rss_kib,
    }

    write_jsonl(output / "raw/state_records.jsonl", state_rows)
    write_jsonl(output / "raw/fold_rows.jsonl", fold_rows)
    write_json(output / "raw/replay_controls.json", replay_controls)
    write_json(
        output / "raw/independent_state_audit.json",
        {
            "executable": independent_rows,
            "storage_adversary": {
                "state_count": storage_state_count,
                "entry_bytes": mean_state_entry_bytes,
                "hot_capacity": hot_capacity,
                "cold_capacity": cold_audit.__dict__,
                "combined_entries": combined_entries,
                "combined_coverage_fraction": combined_coverage,
                "combined_fallback_fraction": combined_fallback,
            },
        },
    )
    write_json(output / "raw/prompt_runs.json", prompt_runs)
    write_json(output / "raw/exclusions.json", exclusions)
    write_json(output / "raw/checkpoint_manifest.json", manifest)
    write_json(output / "processed/aggregate.json", aggregate)
    write_json(output / "summary.json", summary)
    write_json(output / "artifacts/environment.json", environment)
    (output / "artifacts/contract.txt").write_text(
        "EXP-052 E1 CPU exact-advice audit; no 405B, 8 GiB, CUDA, "
        "PCIe, SSD, TTFT, tokens/s, or real unseen-state operation replacement.\n",
        encoding="utf-8",
    )
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "logs/run.log").write_text(
        json.dumps(
            {
                "decision": decision,
                "state_count": len(state_rows),
                "family_min_hit": minimum_family_hit,
                "p90_fraction": p90_observed_fraction,
                "combined_fallback": combined_fallback,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
