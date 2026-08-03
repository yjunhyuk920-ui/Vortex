#!/usr/bin/env python3
"""Run EXP-049 continuous block fixed-point falsification Gate.

This is CPU reference evidence on pinned tiny checkpoints. It does not measure
405B, 8 GiB VRAM, CUDA, PCIe, SSD, TTFT, tokens/second, or physical weight reuse.
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
from typing import Any, Sequence

import numpy as np

from vortex_runtime.block_verify import verify_exact_proposal
from vortex_runtime.fixed_point import (
    GatedTriangularChain,
    MapEvaluation,
    SolverResult,
    matching_prefix,
    run_anderson,
    run_damped_picard,
    triangular_transcript_indistinguishable,
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


def model_parameter_bytes(model: Any) -> int:
    return int(
        sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
    )


def exact_cached_greedy(
    *,
    torch: Any,
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
) -> tuple[tuple[int, ...], int]:
    generated: list[int] = []
    start = time.perf_counter_ns()
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
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
        device = next(model.parameters()).device
        sequence = torch.tensor(
            [prefix + proposal], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            logits = model(
                input_ids=sequence,
                use_cache=False,
                return_dict=True,
            ).logits[0]
        start = len(prefix) - 1
        predictions = logits[start : start + len(proposal)].argmax(dim=-1)
        if int(predictions.numel()) != len(proposal):
            raise RuntimeError("target block prediction alignment failed")
        return tuple(
            int(token) for token in predictions.detach().cpu().tolist()
        )

    return predict


class ContinuousTargetMap:
    def __init__(
        self,
        *,
        torch: Any,
        model: Any,
        prefix: tuple[int, ...],
        block_size: int,
        top_k: int,
        temperature: float,
    ) -> None:
        if not prefix or block_size <= 0 or top_k <= 0 or temperature <= 0.0:
            raise ValueError("invalid continuous target map configuration")
        self.torch = torch
        self.model = model
        self.prefix = prefix
        self.block_size = int(block_size)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.device = next(model.parameters()).device
        self.embedding = model.get_input_embeddings()
        if self.embedding is None:
            raise RuntimeError("model exposes no input embedding")
        self.hidden_size = int(self.embedding.weight.shape[1])
        if self.top_k > int(self.embedding.weight.shape[0]):
            raise ValueError("top_k exceeds vocabulary")
        prefix_ids = torch.tensor(
            [prefix], dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            self.prefix_embeddings = self.embedding(prefix_ids).detach()
        self.calls = 0
        self.elapsed_ns = 0

    def token_embedding(self, token: int) -> np.ndarray:
        with self.torch.inference_mode():
            row = self.embedding.weight[int(token)].detach().to(
                "cpu", self.torch.float64
            )
        return row.numpy().copy()

    def repeated_token_state(self, token: int) -> np.ndarray:
        return np.repeat(
            self.token_embedding(token)[None, :], self.block_size, axis=0
        )

    def zero_state(self) -> np.ndarray:
        return np.zeros(
            (self.block_size, self.hidden_size), dtype=np.float64
        )

    def exact_token_state(self, tokens: Sequence[int]) -> np.ndarray:
        if len(tokens) != self.block_size:
            raise ValueError("exact token state width mismatch")
        indices = self.torch.tensor(
            tokens, dtype=self.torch.long, device=self.device
        )
        with self.torch.inference_mode():
            rows = self.embedding(indices).detach().to(
                "cpu", self.torch.float64
            )
        return rows.numpy().copy()

    def __call__(self, state: np.ndarray) -> MapEvaluation:
        candidate = np.asarray(state, dtype=np.float64)
        expected = (self.block_size, self.hidden_size)
        if candidate.shape != expected or not np.all(np.isfinite(candidate)):
            raise ValueError("continuous state shape/finite contract failed")
        state_tensor = self.torch.from_numpy(candidate).to(
            device=self.device,
            dtype=self.embedding.weight.dtype,
        )
        combined = self.torch.cat(
            (self.prefix_embeddings, state_tensor.unsqueeze(0)), dim=1
        )
        attention_mask = self.torch.ones(
            combined.shape[:2],
            dtype=self.torch.long,
            device=self.device,
        )
        started = time.perf_counter_ns()
        with self.torch.inference_mode():
            logits = self.model(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).logits[0]
            start = len(self.prefix) - 1
            aligned = logits[start : start + self.block_size]
            if int(aligned.shape[0]) != self.block_size:
                raise RuntimeError("continuous map logit alignment failed")
            values, indices = self.torch.topk(
                aligned, k=self.top_k, dim=-1
            )
            probabilities = self.torch.softmax(
                values.to(self.torch.float32) / self.temperature,
                dim=-1,
            )
            selected = self.embedding.weight[indices]
            projected = (
                probabilities.unsqueeze(-1)
                * selected.to(self.torch.float32)
            ).sum(dim=1)
            hard = indices[:, 0]
        self.elapsed_ns += time.perf_counter_ns() - started
        self.calls += 1
        projection_read_bytes = int(
            aligned.numel() * aligned.element_size()
            + selected.numel() * selected.element_size()
        )
        projection_ops = int(aligned.numel() + 2 * selected.numel())
        return MapEvaluation(
            projected_state=projected.detach()
            .to("cpu", self.torch.float64)
            .numpy(),
            hard_tokens=tuple(
                int(token) for token in hard.detach().cpu().tolist()
            ),
            projection_read_bytes=projection_read_bytes,
            projection_ops=projection_ops,
        )


def proposal_sha(tokens: Sequence[int]) -> str:
    payload = ",".join(str(int(token)) for token in tokens).encode("utf-8")
    return sha256_bytes(payload)


def solver_rows(
    *,
    condition: str,
    variant: str,
    block_size: int,
    result: SolverResult,
    reference: Sequence[int],
    map_elapsed_ns: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in result.snapshots:
        prefix = matching_prefix(snapshot.hard_tokens, reference)
        rows.append(
            {
                "condition": condition,
                "variant": variant,
                "block_size": block_size,
                "solver_passes": snapshot.iteration,
                "proposal_sha256": proposal_sha(snapshot.hard_tokens),
                "proposal_preview": list(snapshot.hard_tokens[:16]),
                "matching_prefix": prefix,
                "solver_target_full_streams": snapshot.iteration,
                "projection_read_bytes": snapshot.projection_read_bytes_total,
                "projection_ops": snapshot.projection_ops_total,
                "residual_l2": snapshot.residual_l2,
                "residual_linf": snapshot.residual_linf,
                "coefficient_abs_max": snapshot.coefficient_abs_max,
                "anderson_condition": snapshot.anderson_condition,
                "numerical_fallbacks": snapshot.numerical_fallbacks,
                "anderson_history_bytes_peak": (
                    result.anderson_history_bytes_peak
                ),
                "future_information_used": result.future_information_used,
                "solver_cpu_elapsed_ns": map_elapsed_ns,
                "_proposal": snapshot.hard_tokens,
            }
        )
    return rows


def next_repeat_rows(
    *,
    torch: Any,
    model: Any,
    prefix: tuple[int, ...],
    block_size: int,
    top_k: int,
    temperature: float,
    damping: float,
    reference: Sequence[int],
) -> list[dict[str, Any]]:
    mapping = ContinuousTargetMap(
        torch=torch,
        model=model,
        prefix=prefix,
        block_size=block_size,
        top_k=top_k,
        temperature=temperature,
    )
    last_state = mapping.repeated_token_state(prefix[-1])
    seed = mapping(last_state)
    seed_residual = np.asarray(seed.projected_state) - last_state
    rows = [
        {
            "condition": "S1",
            "variant": "S1_next_repeat_k8_l1",
            "block_size": block_size,
            "solver_passes": 1,
            "proposal_sha256": proposal_sha(seed.hard_tokens),
            "proposal_preview": list(seed.hard_tokens[:16]),
            "matching_prefix": matching_prefix(
                seed.hard_tokens, reference
            ),
            "solver_target_full_streams": 1,
            "projection_read_bytes": seed.projection_read_bytes,
            "projection_ops": seed.projection_ops,
            "residual_l2": float(np.linalg.norm(seed_residual)),
            "residual_linf": float(np.max(np.abs(seed_residual))),
            "coefficient_abs_max": 1.0,
            "anderson_condition": None,
            "numerical_fallbacks": 0,
            "anderson_history_bytes_peak": 0,
            "future_information_used": False,
            "solver_cpu_elapsed_ns": mapping.elapsed_ns,
            "_proposal": seed.hard_tokens,
        }
    ]
    repeated = mapping.repeated_token_state(seed.hard_tokens[0])
    continuation = run_damped_picard(
        repeated,
        map_fn=mapping,
        iterations=3,
        damping=damping,
        record_steps=(1, 3),
    )
    for snapshot in continuation.snapshots:
        total_passes = snapshot.iteration + 1
        hard = snapshot.hard_tokens
        rows.append(
            {
                "condition": "S1",
                "variant": "S1_next_repeat_k8_l1",
                "block_size": block_size,
                "solver_passes": total_passes,
                "proposal_sha256": proposal_sha(hard),
                "proposal_preview": list(hard[:16]),
                "matching_prefix": matching_prefix(hard, reference),
                "solver_target_full_streams": total_passes,
                "projection_read_bytes": (
                    seed.projection_read_bytes
                    + snapshot.projection_read_bytes_total
                ),
                "projection_ops": (
                    seed.projection_ops + snapshot.projection_ops_total
                ),
                "residual_l2": snapshot.residual_l2,
                "residual_linf": snapshot.residual_linf,
                "coefficient_abs_max": 1.0,
                "anderson_condition": None,
                "numerical_fallbacks": 0,
                "anderson_history_bytes_peak": 0,
                "future_information_used": False,
                "solver_cpu_elapsed_ns": mapping.elapsed_ns,
                "_proposal": hard,
            }
        )
    return rows


def hard_jacobi_rows(
    *,
    prefix: tuple[int, ...],
    reference: Sequence[int],
    block_size: int,
    fill_token: int,
    target_predictor: Any,
    checkpoints: set[int],
) -> list[dict[str, Any]]:
    guesses = tuple(fill_token for _ in range(block_size))
    rows: list[dict[str, Any]] = []
    elapsed = 0
    for target_pass in range(1, max(checkpoints) + 1):
        started = time.perf_counter_ns()
        predictions = target_predictor(prefix, guesses)
        elapsed += time.perf_counter_ns() - started
        guesses = predictions
        if target_pass in checkpoints:
            rows.append(
                {
                    "condition": "S0",
                    "variant": "S0_hard_jacobi",
                    "block_size": block_size,
                    "solver_passes": target_pass,
                    "proposal_sha256": proposal_sha(guesses),
                    "proposal_preview": list(guesses[:16]),
                    "matching_prefix": matching_prefix(
                        guesses, reference
                    ),
                    "solver_target_full_streams": target_pass,
                    "projection_read_bytes": 0,
                    "projection_ops": 0,
                    "residual_l2": None,
                    "residual_linf": None,
                    "coefficient_abs_max": 1.0,
                    "anderson_condition": None,
                    "numerical_fallbacks": 0,
                    "anderson_history_bytes_peak": 0,
                    "future_information_used": False,
                    "solver_cpu_elapsed_ns": elapsed,
                    "_proposal": guesses,
                }
            )
    return rows


def verify_selected(
    *,
    row: dict[str, Any],
    prefix: tuple[int, ...],
    reference: Sequence[int],
    target_predictor: Any,
    parameter_bytes: int,
) -> dict[str, Any]:
    proposal = tuple(int(token) for token in row["_proposal"])
    started = time.perf_counter_ns()
    targets = target_predictor(prefix, proposal)
    verify_elapsed = time.perf_counter_ns() - started
    verification = verify_exact_proposal(proposal, targets)
    committed = verification.committed_tokens
    exact_match = committed == tuple(reference[: len(committed)])
    projection_equivalent_streams = (
        row["projection_read_bytes"] / parameter_bytes
    )
    total_streams = (
        row["solver_target_full_streams"]
        + 1.0
        + projection_equivalent_streams
    )
    result = {
        key: value for key, value in row.items() if key != "_proposal"
    }
    result.update(
        {
            "variant_selection_uses_reference": True,
            "exact_verify_full_streams": 1.0,
            "exact_verify_cpu_elapsed_ns": verify_elapsed,
            "verified_matching_prefix": verification.matching_prefix,
            "exact_committed_tokens": len(committed),
            "correction_used": verification.correction_used,
            "rejected_positions": verification.rejected_scored_positions,
            "exact_output_match": exact_match,
            "projection_equivalent_streams": (
                projection_equivalent_streams
            ),
            "target_equivalent_streams": total_streams,
            "target_equivalent_stream_fraction": (
                total_streams / len(committed)
            ),
        }
    )
    if not exact_match:
        raise RuntimeError(
            "retained exact verifier diverged from greedy reference"
        )
    return result


def favorable_key(
    row: dict[str, Any], parameter_bytes: int
) -> tuple[Any, ...]:
    committed_upper = min(
        row["block_size"], row["matching_prefix"] + 1
    )
    streams = (
        row["solver_target_full_streams"]
        + 1.0
        + row["projection_read_bytes"] / parameter_bytes
    )
    return (
        -row["matching_prefix"],
        streams / committed_upper,
        row["solver_passes"],
        row["variant"],
        row["block_size"],
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
    input_ids = encoded["input_ids"].to(
        next(model.parameters()).device
    )
    prefix = tuple(
        int(token) for token in input_ids[0].detach().cpu().tolist()
    )
    if not prefix:
        raise RuntimeError("tokenizer produced an empty prefix")
    max_block = max(int(value) for value in config["block_sizes"])
    reference, baseline_elapsed = exact_cached_greedy(
        torch=torch,
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_block,
    )
    target_predictor = exact_target_block_predictor(torch, model)
    parameter_bytes = int(model_record["parameter_bytes"])
    position_limit = int(
        getattr(
            model.config,
            "max_position_embeddings",
            getattr(model.config, "n_positions", 2048),
        )
    )
    checkpoints = {
        int(value) for value in config["solver_pass_checkpoints"]
    }
    temperature = float(config["projection"]["temperature"])
    all_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []

    for block_size in (
        int(value) for value in config["block_sizes"]
    ):
        if len(prefix) + block_size > position_limit:
            exclusions.append(
                {
                    "block_size": block_size,
                    "reason": "context_limit",
                    "prefix_tokens": len(prefix),
                    "position_limit": position_limit,
                }
            )
            continue
        exact = reference[:block_size]
        all_rows.extend(
            hard_jacobi_rows(
                prefix=prefix,
                reference=exact,
                block_size=block_size,
                fill_token=int(config["hard_jacobi_fill_token"]),
                target_predictor=target_predictor,
                checkpoints=checkpoints,
            )
        )

        for variant in config["picard_variants"]:
            if variant["initialization"] == "next_repeat":
                all_rows.extend(
                    next_repeat_rows(
                        torch=torch,
                        model=model,
                        prefix=prefix,
                        block_size=block_size,
                        top_k=int(variant["top_k"]),
                        temperature=temperature,
                        damping=float(variant["damping"]),
                        reference=exact,
                    )
                )
                continue
            mapping = ContinuousTargetMap(
                torch=torch,
                model=model,
                prefix=prefix,
                block_size=block_size,
                top_k=int(variant["top_k"]),
                temperature=temperature,
            )
            initial = (
                mapping.zero_state()
                if variant["initialization"] == "zero"
                else mapping.repeated_token_state(prefix[-1])
            )
            result = run_damped_picard(
                initial,
                map_fn=mapping,
                iterations=int(config["max_solver_passes"]),
                damping=float(variant["damping"]),
                record_steps=tuple(sorted(checkpoints)),
            )
            all_rows.extend(
                solver_rows(
                    condition="S1",
                    variant=str(variant["name"]),
                    block_size=block_size,
                    result=result,
                    reference=exact,
                    map_elapsed_ns=mapping.elapsed_ns,
                )
            )

        for variant in config["anderson_variants"]:
            mapping = ContinuousTargetMap(
                torch=torch,
                model=model,
                prefix=prefix,
                block_size=block_size,
                top_k=int(variant["top_k"]),
                temperature=temperature,
            )
            initial = mapping.repeated_token_state(prefix[-1])
            result = run_anderson(
                initial,
                map_fn=mapping,
                iterations=int(config["max_solver_passes"]),
                history_size=int(variant["history_size"]),
                damping=float(variant["damping"]),
                regularization=float(
                    config["anderson"]["regularization"]
                ),
                coefficient_clip=float(
                    config["anderson"]["coefficient_clip"]
                ),
                condition_limit=float(
                    config["anderson"]["condition_limit"]
                ),
                record_steps=tuple(sorted(checkpoints)),
            )
            all_rows.extend(
                solver_rows(
                    condition="S2",
                    variant=str(variant["name"]),
                    block_size=block_size,
                    result=result,
                    reference=exact,
                    map_elapsed_ns=mapping.elapsed_ns,
                )
            )

        oracle_map = ContinuousTargetMap(
            torch=torch,
            model=model,
            prefix=prefix,
            block_size=block_size,
            top_k=int(config["projection"]["primary_top_k"]),
            temperature=temperature,
        )
        oracle_eval = oracle_map(
            oracle_map.exact_token_state(exact)
        )
        oracle_verify = verify_exact_proposal(
            exact, oracle_eval.hard_tokens
        )
        oracle_exact = oracle_verify.matching_prefix == block_size
        if not oracle_exact:
            raise RuntimeError(
                "S3 exact future-state oracle failed map alignment"
            )
        oracle_rows.append(
            {
                "condition": "S3",
                "block_size": block_size,
                "matching_prefix": oracle_verify.matching_prefix,
                "exact_committed_tokens": len(
                    oracle_verify.committed_tokens
                ),
                "target_solver_full_streams": 1,
                "target_equivalent_stream_fraction": (
                    1.0
                    + oracle_eval.projection_read_bytes
                    / parameter_bytes
                )
                / len(oracle_verify.committed_tokens),
                "future_information_used": True,
                "exact_output_match": True,
                "projection_read_bytes": (
                    oracle_eval.projection_read_bytes
                ),
                "projection_ops": oracle_eval.projection_ops,
                "cpu_elapsed_ns": oracle_map.elapsed_ns,
            }
        )

    continuous = [
        row
        for row in all_rows
        if row["condition"] in {"S1", "S2"}
    ]
    if not continuous:
        raise RuntimeError("no valid S1/S2 rows were produced")
    best = min(
        continuous,
        key=lambda row: favorable_key(row, parameter_bytes),
    )
    selected = verify_selected(
        row=best,
        prefix=prefix,
        reference=reference[: best["block_size"]],
        target_predictor=target_predictor,
        parameter_bytes=parameter_bytes,
    )

    pass4_s0 = [
        row
        for row in all_rows
        if row["condition"] == "S0" and row["solver_passes"] == 4
    ]
    pass4_s2 = [
        row
        for row in all_rows
        if row["condition"] == "S2" and row["solver_passes"] == 4
    ]
    best_s0 = max(
        pass4_s0,
        key=lambda row: (
            row["matching_prefix"],
            -row["block_size"],
        ),
    )
    best_s2 = max(
        pass4_s2,
        key=lambda row: (
            row["matching_prefix"],
            -row["block_size"],
        ),
    )

    public_rows = [
        {key: value for key, value in row.items() if key != "_proposal"}
        for row in all_rows
    ]
    return {
        "model_id": model_record["model_id"],
        "model_revision": model_record["resolved_revision"],
        "parameter_count": model_record["parameter_count"],
        "parameter_bytes": parameter_bytes,
        "layer_count": model_record["layer_count"],
        "hidden_size": model_record["hidden_size"],
        "prompt_index": prompt_index,
        "prompt_family": prompt_record["family"],
        "prompt_sha256": sha256_bytes(
            str(prompt_record["text"]).encode("utf-8")
        ),
        "input_tokens": len(prefix),
        "position_limit": position_limit,
        "baseline_reference_tokens": max_block,
        "baseline_cpu_elapsed_ns": baseline_elapsed,
        "excluded_states": exclusions,
        "solver_rows": public_rows,
        "selected_oracle_best_s1s2": selected,
        "best_s0_pass4_matching_prefix": (
            best_s0["matching_prefix"]
        ),
        "best_s2_pass4_matching_prefix": (
            best_s2["matching_prefix"]
        ),
        "s3_oracles": oracle_rows,
    }


def triangular_audit(config: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    barrier_holds = True
    for index, item in enumerate(config["triangular_controls"]):
        length = int(item["length"])
        vocab_size = int(item["vocab_size"])
        if index == 0:
            exact = tuple(range(1, length + 1))
        else:
            exact = tuple(
                ((position * 37 + 11) % (vocab_size - 1)) + 1
                for position in range(length)
            )
        chain = GatedTriangularChain(
            exact,
            vocab_size=vocab_size,
            decoy_token=int(item["decoy_token"]),
        )
        picard = run_damped_picard(
            chain.zero_state(),
            map_fn=chain.map,
            iterations=4,
            damping=1.0,
            record_steps=(1, 2, 3, 4),
        )
        anderson = run_anderson(
            chain.zero_state(),
            map_fn=chain.map,
            iterations=4,
            history_size=4,
            damping=1.0,
            regularization=1e-8,
            record_steps=(1, 2, 3, 4),
        )
        picard_prefixes = [
            matching_prefix(row.hard_tokens, exact)
            for row in picard.snapshots
        ]
        anderson_prefixes = [
            matching_prefix(row.hard_tokens, exact)
            for row in anderson.snapshots
        ]
        local_holds = all(
            prefix <= round_index
            for round_index, prefix in enumerate(
                picard_prefixes, start=1
            )
        ) and all(
            prefix <= round_index
            for round_index, prefix in enumerate(
                anderson_prefixes, start=1
            )
        )
        barrier_holds = barrier_holds and local_holds
        records.append(
            {
                "name": item["name"],
                "length": length,
                "picard_prefixes_by_round": picard_prefixes,
                "anderson_prefixes_by_round": anderson_prefixes,
                "anderson_numerical_fallbacks": (
                    anderson.numerical_fallbacks
                ),
                "one_position_per_round_barrier_observed": (
                    local_holds
                ),
            }
        )

    first = GatedTriangularChain(
        (1, 2, 3, 4, 5, 6), vocab_size=10, decoy_token=0
    )
    second = GatedTriangularChain(
        (1, 2, 7, 8, 9, 6), vocab_size=10, decoy_token=0
    )
    query = np.zeros((6, 10), dtype=np.float64)
    query[0, 1] = 1.0
    indistinguishable = triangular_transcript_indistinguishable(
        first, second, query, unresolved_position=2
    )
    return {
        "declared_interface": (
            "one synchronous black-box causal block evaluation per target "
            "round; arbitrary history arithmetic but no external future "
            "information"
        ),
        "transcript_indistinguishability_observed": indistinguishable,
        "one_position_per_round_barrier_observed": (
            barrier_holds and indistinguishable
        ),
        "records": records,
    }


def aggregate_results(
    cases: list[dict[str, Any]],
    triangular: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    selected = [case["selected_oracle_best_s1s2"] for case in cases]
    prefixes = [float(row["matching_prefix"]) for row in selected]
    fractions = [
        float(row["target_equivalent_stream_fraction"])
        for row in selected
    ]
    exact_mismatches = sum(
        not bool(row["exact_output_match"]) for row in selected
    )
    future_uses = sum(
        bool(row["future_information_used"]) for row in selected
    )
    numerical_fallbacks = sum(
        int(row["numerical_fallbacks"]) for row in selected
    )
    unhandled_numerical_failures = 0
    s0_p50 = statistics.median(
        float(case["best_s0_pass4_matching_prefix"])
        for case in cases
    )
    s2_p50 = statistics.median(
        float(case["best_s2_pass4_matching_prefix"])
        for case in cases
    )
    improvement = s2_p50 / max(1.0, s0_p50)

    model_medians: dict[str, float] = {}
    for model_id in sorted({str(case["model_id"]) for case in cases}):
        model_medians[model_id] = statistics.median(
            float(
                case["selected_oracle_best_s1s2"][
                    "matching_prefix"
                ]
            )
            for case in cases
            if case["model_id"] == model_id
        )
    model_order = [
        str(item["model_id"]) for item in config["models"]
    ]
    smallest = model_medians[model_order[0]]
    largest = model_medians[model_order[-1]]
    size_trend_pass = largest >= (
        1.0
        - float(
            config["early_gate"][
                "maximum_relative_largest_model_prefix_drop"
            ]
        )
    ) * smallest

    p50_prefix = statistics.median(prefixes)
    p90_fraction = percentile(fractions, 0.90)
    prefix_pass = p50_prefix >= float(
        config["early_gate"]["minimum_p50_matching_prefix"]
    )
    traffic_pass = p90_fraction <= float(
        config["early_gate"][
            "maximum_p90_target_equivalent_stream_fraction"
        ]
    )
    anderson_pass = improvement >= float(
        config["early_gate"][
            "minimum_anderson_improvement_over_jacobi"
        ]
    )
    exact_pass = exact_mismatches <= int(
        config["early_gate"]["exact_mismatch_limit"]
    )
    causal_pass = future_uses <= int(
        config["early_gate"][
            "deployable_future_information_limit"
        ]
    )
    numerical_pass = unhandled_numerical_failures <= int(
        config["early_gate"][
            "unhandled_numerical_failure_limit"
        ]
    )
    universal_barrier_absent = not bool(
        triangular["one_position_per_round_barrier_observed"]
    )
    survives = all(
        [
            prefix_pass,
            traffic_pass,
            anderson_pass,
            exact_pass,
            causal_pass,
            numerical_pass,
            size_trend_pass,
            universal_barrier_absent,
        ]
    )
    decision = (
        "CONTINUE_TARGET_ONLY_CONTINUOUS_FIXED_POINT_TO_COMPLETE_PHASE_C"
        if survives
        else str(config["early_gate"]["decision_on_failure"])
    )
    return {
        "MEASURED": {
            "model_count": len(
                {case["model_id"] for case in cases}
            ),
            "case_count": len(cases),
            "prompt_family_count": len(
                {case["prompt_family"] for case in cases}
            ),
            "oracle_best_s1s2_p50_matching_prefix": p50_prefix,
            "oracle_best_s1s2_max_matching_prefix": max(prefixes),
            "oracle_best_s1s2_p90_target_equivalent_stream_fraction": (
                p90_fraction
            ),
            "s0_p50_matching_prefix_after_4_passes": s0_p50,
            "s2_p50_matching_prefix_after_4_passes": s2_p50,
            "s2_improvement_over_s0": improvement,
            "selected_exact_output_mismatches": exact_mismatches,
            "selected_deployable_future_information_uses": (
                future_uses
            ),
            "selected_numerical_fallbacks": numerical_fallbacks,
            "unhandled_numerical_failures": (
                unhandled_numerical_failures
            ),
            "model_median_matching_prefix": model_medians,
            "triangular_one_position_per_round_barrier_observed": bool(
                triangular[
                    "one_position_per_round_barrier_observed"
                ]
            ),
            "triangular_transcript_indistinguishability_observed": bool(
                triangular[
                    "transcript_indistinguishability_observed"
                ]
            ),
            "s3_oracle_exact_failures": sum(
                not all(
                    bool(row["exact_output_match"])
                    for row in case["s3_oracles"]
                )
                for case in cases
            ),
            "excluded_state_count": sum(
                len(case["excluded_states"]) for case in cases
            ),
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "DERIVED": {
            "decision": decision,
            "target_only_continuous_fixed_point_survives_early_gate": (
                survives
            ),
            "prefix_gate_pass": prefix_pass,
            "traffic_gate_pass": traffic_pass,
            "anderson_improvement_gate_pass": anderson_pass,
            "exact_gate_pass": exact_pass,
            "causal_gate_pass": causal_pass,
            "numerical_gate_pass": numerical_pass,
            "model_size_trend_gate_pass": size_trend_pass,
            "universal_round_barrier_absent_gate_pass": (
                universal_barrier_absent
            ),
            "variant_selection_uses_reference": True,
            "universal_claim_scope": (
                "rejected when hidden triangular barrier is observed; "
                "empirical checkpoint rows remain average-case evidence only"
            ),
        },
    }


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
        default=ROOT / "experiments/exp_049/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_049_candidate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir
    if output.exists():
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]) % (2**32))
    cache_dir = Path(
        os.environ.get("HF_HOME", output / "hf-cache")
    )
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
    cases: list[dict[str, Any]] = []

    for model_cfg in config["models"]:
        model_path, files = resolve_snapshot(
            model_id=str(model_cfg["model_id"]),
            revision=str(model_cfg["revision"]),
            cache_dir=cache_dir,
            allow_patterns=MODEL_PATTERNS,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        parameter_count = model_parameter_count(model)
        parameter_bytes = model_parameter_bytes(model)
        model_record = {
            "model_id": model_cfg["model_id"],
            "resolved_revision": model_cfg["revision"],
            "parameter_count": parameter_count,
            "parameter_bytes": parameter_bytes,
            "layer_count": int(
                getattr(
                    model.config,
                    "num_layers",
                    getattr(model.config, "num_hidden_layers", 0),
                )
            ),
            "hidden_size": int(
                getattr(model.config, "hidden_size")
            ),
            "vocab_size": int(
                getattr(model.config, "vocab_size")
            ),
            "files": files,
        }
        manifest["models"].append(model_record)
        for prompt_index, prompt in enumerate(
            config["held_out_prompts"]
        ):
            case = run_case(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                model_record=model_record,
                prompt_record=prompt,
                prompt_index=prompt_index,
                config=config,
            )
            cases.append(case)
            print(
                json.dumps(
                    {
                        "model": case["model_id"],
                        "family": case["prompt_family"],
                        "best_prefix": case[
                            "selected_oracle_best_s1s2"
                        ]["matching_prefix"],
                        "fraction": case[
                            "selected_oracle_best_s1s2"
                        ]["target_equivalent_stream_fraction"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del model

    triangular = triangular_audit(config)
    aggregate = aggregate_results(cases, triangular, config)
    target_parameters = int(
        config["projection_target"]["target_parameters"]
    )
    baseline_parameters = int(
        config["projection_target"]["baseline_parameters"]
    )
    bits = int(config["projection_target"]["bits_per_weight"])
    multiplier = float(
        config["projection_target"]["target_traffic_multiplier"]
    )
    target_gib = target_parameters * bits / 8 / 1024**3
    baseline_gib = baseline_parameters * bits / 8 / 1024**3
    allowed_gib = baseline_gib * multiplier
    required_fraction = allowed_gib / target_gib

    summary = {
        "experiment": "EXP-049",
        "name": (
            "anderson_accelerated_continuous_block_fixed_point_gate"
        ),
        "phase": ["A", "B", "C-observation"],
        "evidence_level": "E1",
        "complete_real_operation_replacement": False,
        "phase_d_status": "NOT TESTED",
        "future_information_policy": {
            "S0_S1_S2": False,
            "S3_oracle": True,
        },
        **aggregate,
        "PROJECTED": {
            "target_q4_full_weight_gib_per_stream": target_gib,
            "baseline_q4_full_weight_gib_per_stream": baseline_gib,
            "allowed_1_2x_baseline_gib_per_token": allowed_gib,
            "required_target_equivalent_stream_fraction": (
                required_fraction
            ),
        },
        "UNVERIFIED": [
            "complete multi-cycle real operation-replacement generator",
            "causal deployable fixed variant selector",
            "physical target-weight reuse across block positions",
            "accelerator continuous projection and Anderson kernels",
            "KV and memory-transfer cost on target hardware",
            "70B and 405B behavior",
            "8 GiB peak VRAM",
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
    with (raw_dir / "cases.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")
    write_json(raw_dir / "checkpoint_manifest.json", manifest)
    write_json(raw_dir / "triangular_audit.json", triangular)
    write_json(processed_dir / "aggregate.json", aggregate)
    write_json(output / "summary.json", summary)
    write_json(
        artifacts_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
    )
    (artifacts_dir / "contract.txt").write_text(
        "EXP-049 E1 CPU reference only. S3 is future-aware. S1/S2 "
        "fixed trajectories use no future tokens. Offline best-variant "
        "selection uses the exact reference only as a favorable "
        "falsification upper bound. 405B, 8 GiB, CUDA, PCIe, SSD, TTFT, "
        "tokens/second, and physical block weight reuse are NOT TESTED.\n",
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
