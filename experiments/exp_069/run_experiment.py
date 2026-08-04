#!/usr/bin/env python3
"""Run EXP-069 pinned causal exact temporal-span necessary Gate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import time
from typing import Any, Sequence

import numpy as np

from experiments.exp_061.run_experiment import (
    MODEL_PATTERNS,
    TOKENIZER_PATTERNS,
    dump,
    dump_rows,
    git_commit,
    matrix_role,
    percentile,
    resolve_snapshot,
    sha256_file,
    sha256_tokens,
    unregistered_projection_parameters,
    write_checksums,
)
from vortex_runtime.temporal_span_replay import (
    TemporalSpanRecorder,
    canonical_float32_bytes,
    certify_temporal_span,
    dense_operation_terms,
    favorable_basis_cache_bytes,
    grouped_calls,
    q4_matrix_bytes,
    verify_fraction_witness,
)

ROOT = Path(__file__).resolve().parents[2]
EXP061_REGISTRATIONS = ROOT / "results/exp_061/raw/registration_rows.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def exp061_registration_index() -> dict[tuple[str, str], dict[str, Any]]:
    if not EXP061_REGISTRATIONS.exists():
        raise RuntimeError("frozen EXP-061 registration evidence is missing")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(EXP061_REGISTRATIONS):
        key = (str(row["model_id"]), str(row["canonical_name"]))
        if key in result:
            raise RuntimeError(f"duplicate EXP-061 registration: {key}")
        result[key] = row
    return result


def trace_sha256(vectors: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for vector in vectors:
        payload = canonical_float32_bytes(vector)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def cached_greedy(
    *,
    torch: Any,
    model: Any,
    input_ids: Any,
    attention_mask: Any | None,
    max_new_tokens: int,
    eos_token_ids: set[int],
    recorder: TemporalSpanRecorder | None,
    model_id: str,
    prompt_family: str,
) -> tuple[tuple[int, ...], bool]:
    generated: list[int] = []
    terminated = False
    with torch.inference_mode():
        if recorder is not None:
            recorder.set_context(
                model_id=model_id,
                prompt_family=prompt_family,
                phase="prefill",
                decode_step=0,
            )
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        token = output.logits[:, -1, :].argmax(dim=-1)
        for generated_index in range(max_new_tokens):
            token_id = int(token.item())
            generated.append(token_id)
            if token_id in eos_token_ids:
                terminated = True
                break
            if generated_index + 1 == max_new_tokens:
                break
            next_step = generated_index + 1
            if recorder is not None:
                recorder.set_context(
                    model_id=model_id,
                    prompt_family=prompt_family,
                    phase="first_decode" if next_step == 1 else "warm_decode",
                    decode_step=next_step,
                )
            output = model(
                input_ids=token.reshape(1, 1),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = output.past_key_values
            token = output.logits[:, -1, :].argmax(dim=-1)
    if recorder is not None:
        recorder.set_context(
            model_id=model_id,
            prompt_family=prompt_family,
            phase="inactive",
            decode_step=-1,
        )
    return tuple(generated), terminated


def control_population(primes: Sequence[int], seed: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    failures = 0

    recurrence = [
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        np.asarray([2.0, -1.0], dtype=np.float32),
    ]
    result = certify_temporal_span(recurrence, primes=primes)
    passed = result.independent_flags == (True, True, False, False)
    failures += int(not passed)
    rows.append(
        {
            "control": "low_dimensional_exact_recurrence",
            "passed": passed,
            **result.as_dict(),
        }
    )

    repeated = recurrence[:2] + [recurrence[0].copy()]
    result = certify_temporal_span(repeated, primes=primes)
    witness = verify_fraction_witness(
        repeated[:2], repeated[2], [Fraction(1), Fraction(0)]
    )
    passed = result.exact_duplicate_hits == 1 and witness
    failures += int(not passed)
    rows.append(
        {
            "control": "repeated_vector_exact_witness",
            "passed": passed,
            "witness_verified": witness,
            **result.as_dict(),
        }
    )

    affine_sources = [
        np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([3.0, 1.0, 2.0], dtype=np.float32),
    ]
    affine_target = np.asarray([2.0, 1.0, 1.0], dtype=np.float32)
    witness = verify_fraction_witness(
        affine_sources,
        affine_target,
        [Fraction(1, 2), Fraction(1, 2)],
    )
    result = certify_temporal_span(
        affine_sources + [affine_target], primes=primes
    )
    passed = witness and not result.independent_flags[-1]
    failures += int(not passed)
    rows.append(
        {
            "control": "affine_dependent_exact_witness",
            "passed": passed,
            "witness_verified": witness,
            **result.as_dict(),
        }
    )

    triangular = [np.eye(16, dtype=np.float32)[index] for index in range(16)]
    result = certify_temporal_span(triangular, primes=primes)
    passed = all(result.independent_flags) and result.maximum_rank_lower_bound == 16
    failures += int(not passed)
    rows.append(
        {
            "control": "triangular_forced_independence",
            "passed": passed,
            **result.as_dict(),
        }
    )

    rng = np.random.default_rng(seed)
    random_matrix = rng.integers(-16, 17, size=(20, 12), dtype=np.int16).astype(
        np.float32
    )
    result = certify_temporal_span(list(random_matrix), primes=primes)
    final_ranks = [trajectory[-1] for trajectory in result.rank_trajectories.values()]
    passed = (
        result.maximum_rank_lower_bound == 12
        and len(set(final_ranks)) == 1
        and result.rank_disagreement_count == 0
    )
    failures += int(not passed)
    rows.append(
        {
            "control": "random_dyadic_dimension_saturation",
            "passed": passed,
            **result.as_dict(),
        }
    )
    return rows, failures


def phase_summary(calls: Sequence[Any], phase: str) -> dict[str, Any]:
    selected = [row for row in calls if row.phase == phase]
    modules = Counter(row.module_name for row in selected)
    return {
        "phase": phase,
        "captured_vector_count": len(selected),
        "module_count": len(modules),
        "module_vector_count_minimum": min(modules.values()) if modules else 0,
        "module_vector_count_maximum": max(modules.values()) if modules else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/exp_069/config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/exp_069_candidate",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache/exp_069_huggingface",
    )
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output = arguments.output_dir
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    arguments.cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(config["torch_num_threads"]))
    os.environ.setdefault("MKL_NUM_THREADS", str(config["torch_num_threads"]))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(int(config["torch_num_threads"]))
    primes = tuple(int(value) for value in config["primes"])
    frozen_registrations = exp061_registration_index()
    tokenizer_entry = config["tokenizer"]
    tokenizer_snapshot = resolve_snapshot(
        model_id=str(tokenizer_entry["model_id"]),
        revision=str(tokenizer_entry["revision"]),
        cache_dir=arguments.cache_dir,
        allow_patterns=TOKENIZER_PATTERNS,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_snapshot, local_files_only=True
    )
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    registration_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    output_token_mismatches = 0
    registration_mismatches = 0
    rank_or_trace_mismatches = 0
    frozen_registration_hash_mismatches = 0
    started = time.perf_counter_ns()

    for model_entry in config["models"]:
        model_id = str(model_entry["model_id"])
        revision = str(model_entry["revision"])
        snapshot = resolve_snapshot(
            model_id=model_id,
            revision=revision,
            cache_dir=arguments.cache_dir,
            allow_patterns=MODEL_PATTERNS,
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        recorder = TemporalSpanRecorder.from_model(model)
        unregistered = unregistered_projection_parameters(model, recorder)
        registration_mismatches += len(unregistered)
        registration_index = {
            item.canonical_name: item for item in recorder.registrations
        }

        for item in recorder.registrations:
            frozen = frozen_registrations.get((model_id, item.canonical_name))
            if frozen is None:
                frozen_registration_hash_mismatches += 1
                frozen_hash = None
            else:
                frozen_hash = str(frozen["weight_sha256"])
                frozen_registration_hash_mismatches += int(
                    frozen_hash != item.weight_sha256
                    or list(item.weight_shape) != list(frozen["weight_shape"])
                )
            registration_rows.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "canonical_name": item.canonical_name,
                    "aliases": list(item.aliases),
                    "input_width": item.input_width,
                    "output_width": item.output_width,
                    "weight_shape": list(item.weight_shape),
                    "weight_sha256": item.weight_sha256,
                    "frozen_exp061_weight_sha256": frozen_hash,
                    "frozen_exp061_match": frozen_hash == item.weight_sha256,
                }
            )

        recorder.attach()
        model_case_start = len(case_rows)
        model_projection_start = len(projection_rows)

        for prompt in config["held_out_prompts"]:
            family = str(prompt["family"])
            text = str(prompt["text"])
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=int(config["max_input_tokens"]),
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            if int(input_ids.shape[1]) == 0:
                raise RuntimeError(f"empty prompt tokenization for {family}")

            reference, reference_terminated = cached_greedy(
                torch=torch,
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(config["max_new_tokens"]),
                eos_token_ids=eos_ids,
                recorder=None,
                model_id=model_id,
                prompt_family=family,
            )
            observed, observed_terminated = cached_greedy(
                torch=torch,
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(config["max_new_tokens"]),
                eos_token_ids=eos_ids,
                recorder=recorder,
                model_id=model_id,
                prompt_family=family,
            )
            calls = recorder.drain()
            mismatch_count = sum(
                int(left != right)
                for left, right in zip(reference, observed, strict=False)
            ) + abs(len(reference) - len(observed))
            mismatch_count += int(reference_terminated != observed_terminated)
            output_token_mismatches += mismatch_count

            warm_step_count = max(0, len(observed) - 2)
            if (
                not observed_terminated
                and warm_step_count
                < int(config["minimum_warm_decode_tokens_unless_eos"])
            ):
                rank_or_trace_mismatches += 1

            warm_groups = grouped_calls(calls, phase="warm_decode")
            case_projection_rows: list[dict[str, Any]] = []
            for module_name, registration in sorted(registration_index.items()):
                module_calls = warm_groups.get(module_name, [])
                if warm_step_count > 0 and len(module_calls) != warm_step_count:
                    rank_or_trace_mismatches += 1
                if not module_calls:
                    continue
                vectors = [row.vector for row in module_calls]
                certificate = certify_temporal_span(vectors, primes=primes)
                if certificate.rank_disagreement_count:
                    # Disagreement is not unsound, but is retained as an audit
                    # signal and disallowed by this first Gate.
                    rank_or_trace_mismatches += certificate.rank_disagreement_count

                per_call_weight_bytes = q4_matrix_bytes(
                    registration.input_width, registration.output_width
                )
                per_call_operations = dense_operation_terms(
                    registration.input_width, registration.output_width
                )
                total_weight_bytes = certificate.vector_count * per_call_weight_bytes
                total_operations = certificate.vector_count * per_call_operations
                mandatory_weight_bytes = (
                    certificate.certified_independent_count * per_call_weight_bytes
                )
                mandatory_operations = (
                    certificate.certified_independent_count * per_call_operations
                )
                duplicate_cache_read_bytes = (
                    certificate.exact_duplicate_hits
                    * registration.output_width
                    * 4
                )
                duplicate_copy_operations = (
                    certificate.exact_duplicate_hits * registration.output_width
                )
                fail_closed_full_calls = (
                    certificate.vector_count - certificate.exact_duplicate_hits
                )
                fail_closed_weight_bytes = (
                    fail_closed_full_calls * per_call_weight_bytes
                    + duplicate_cache_read_bytes
                )
                fail_closed_operations = (
                    fail_closed_full_calls * per_call_operations
                    + duplicate_copy_operations
                )
                cache_bytes = favorable_basis_cache_bytes(
                    input_width=registration.input_width,
                    output_width=registration.output_width,
                    rank_lower_bound=certificate.maximum_rank_lower_bound,
                )
                row = {
                    "model_id": model_id,
                    "revision": revision,
                    "prompt_family": family,
                    "module_name": module_name,
                    "module_aliases": list(registration.aliases),
                    "input_width": registration.input_width,
                    "output_width": registration.output_width,
                    "warm_decode_vector_count": certificate.vector_count,
                    "trace_sha256": trace_sha256(vectors),
                    "q4_matrix_bytes_per_full_call": per_call_weight_bytes,
                    "dense_operations_per_full_call": per_call_operations,
                    "dense_weight_bytes_all_calls": total_weight_bytes,
                    "dense_operations_all_calls": total_operations,
                    "mandatory_full_weight_bytes_lower_bound": mandatory_weight_bytes,
                    "mandatory_operations_lower_bound": mandatory_operations,
                    "mandatory_weight_fraction_lower_bound": (
                        mandatory_weight_bytes / total_weight_bytes
                    ),
                    "mandatory_operation_fraction_lower_bound": (
                        mandatory_operations / total_operations
                    ),
                    "verified_exact_replay_hits": certificate.exact_duplicate_hits,
                    "verified_exact_replay_hit_fraction": (
                        certificate.exact_duplicate_fraction
                    ),
                    "unverified_nonincrease_count": (
                        certificate.uncertified_count
                        - certificate.exact_duplicate_hits
                    ),
                    "fail_closed_weight_bytes": fail_closed_weight_bytes,
                    "fail_closed_operation_terms": fail_closed_operations,
                    "fail_closed_weight_fraction": (
                        fail_closed_weight_bytes / total_weight_bytes
                    ),
                    "fail_closed_operation_fraction": (
                        fail_closed_operations / total_operations
                    ),
                    "favorable_basis_cache_bytes": cache_bytes,
                    "basis_cache_to_q4_matrix_ratio": (
                        cache_bytes / per_call_weight_bytes
                    ),
                    **certificate.as_dict(),
                }
                projection_rows.append(row)
                case_projection_rows.append(row)

            if not case_projection_rows and warm_step_count > 0:
                rank_or_trace_mismatches += 1
            dense_weight_total = sum(
                int(row["dense_weight_bytes_all_calls"])
                for row in case_projection_rows
            )
            dense_operation_total = sum(
                int(row["dense_operations_all_calls"])
                for row in case_projection_rows
            )
            mandatory_weight_total = sum(
                int(row["mandatory_full_weight_bytes_lower_bound"])
                for row in case_projection_rows
            )
            mandatory_operation_total = sum(
                int(row["mandatory_operations_lower_bound"])
                for row in case_projection_rows
            )
            fail_closed_weight_total = sum(
                int(row["fail_closed_weight_bytes"])
                for row in case_projection_rows
            )
            fail_closed_operation_total = sum(
                int(row["fail_closed_operation_terms"])
                for row in case_projection_rows
            )
            replay_hits = sum(
                int(row["verified_exact_replay_hits"])
                for row in case_projection_rows
            )
            warm_vectors = sum(
                int(row["warm_decode_vector_count"])
                for row in case_projection_rows
            )
            cache_total = sum(
                int(row["favorable_basis_cache_bytes"])
                for row in case_projection_rows
            )
            q4_matrix_population = sum(
                int(row["q4_matrix_bytes_per_full_call"])
                for row in case_projection_rows
            )

            case_rows.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "prompt_family": family,
                    "prompt": text,
                    "prompt_token_count": int(input_ids.numel()),
                    "generated_token_count": len(observed),
                    "terminated_on_eos": observed_terminated,
                    "warm_decode_step_count": warm_step_count,
                    "reference_token_sha256": sha256_tokens(reference),
                    "observed_token_sha256": sha256_tokens(observed),
                    "output_token_mismatches": mismatch_count,
                    "registered_projection_count": len(registration_index),
                    "warm_projection_trace_count": len(case_projection_rows),
                    "mandatory_weight_fraction_lower_bound": (
                        mandatory_weight_total / dense_weight_total
                        if dense_weight_total
                        else 0.0
                    ),
                    "mandatory_operation_fraction_lower_bound": (
                        mandatory_operation_total / dense_operation_total
                        if dense_operation_total
                        else 0.0
                    ),
                    "fail_closed_weight_fraction": (
                        fail_closed_weight_total / dense_weight_total
                        if dense_weight_total
                        else 0.0
                    ),
                    "fail_closed_operation_fraction": (
                        fail_closed_operation_total / dense_operation_total
                        if dense_operation_total
                        else 0.0
                    ),
                    "verified_exact_replay_hits": replay_hits,
                    "verified_exact_replay_hit_fraction": (
                        replay_hits / warm_vectors if warm_vectors else 0.0
                    ),
                    "favorable_basis_cache_bytes": cache_total,
                    "q4_projection_matrix_population_bytes": q4_matrix_population,
                    "basis_cache_to_q4_projection_population_ratio": (
                        cache_total / q4_matrix_population
                        if q4_matrix_population
                        else 0.0
                    ),
                }
            )
            for phase in ("prefill", "first_decode", "warm_decode"):
                phase_rows.append(
                    {
                        "model_id": model_id,
                        "revision": revision,
                        "prompt_family": family,
                        **phase_summary(calls, phase),
                    }
                )

        missing_called = recorder.missing_called_modules()
        registration_mismatches += len(missing_called)
        recorder.detach()
        local_cases = case_rows[model_case_start:]
        local_projections = projection_rows[model_projection_start:]
        model_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "parameter_count": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "registered_projection_count": len(recorder.registrations),
                "unregistered_projection_parameters": list(unregistered),
                "uncalled_registered_projections": list(missing_called),
                "case_count": len(local_cases),
                "projection_trace_count": len(local_projections),
                "p50_mandatory_weight_fraction_lower_bound": percentile(
                    [
                        row["mandatory_weight_fraction_lower_bound"]
                        for row in local_cases
                    ],
                    0.50,
                ),
                "p90_mandatory_weight_fraction_lower_bound": percentile(
                    [
                        row["mandatory_weight_fraction_lower_bound"]
                        for row in local_cases
                    ],
                    0.90,
                ),
                "verified_exact_replay_hits": sum(
                    int(row["verified_exact_replay_hits"])
                    for row in local_cases
                ),
            }
        )
        del model

    controls, control_failures = control_population(
        primes, int(config["seed"])
    )
    rank_or_trace_mismatches += control_failures

    fractions = [
        float(row["mandatory_weight_fraction_lower_bound"])
        for row in case_rows
    ]
    operations = [
        float(row["mandatory_operation_fraction_lower_bound"])
        for row in case_rows
    ]
    family_values: dict[str, list[float]] = defaultdict(list)
    family_replays: dict[str, list[float]] = defaultdict(list)
    for row in case_rows:
        family = str(row["prompt_family"])
        family_values[family].append(
            float(row["mandatory_weight_fraction_lower_bound"])
        )
        family_replays[family].append(
            float(row["verified_exact_replay_hit_fraction"])
        )
    family_p90 = {
        family: percentile(values, 0.90)
        for family, values in sorted(family_values.items())
    }
    family_max_replay = {
        family: max(values) for family, values in sorted(family_replays.items())
    }
    model_p50 = {
        str(row["model_id"]): float(
            row["p50_mandatory_weight_fraction_lower_bound"]
        )
        for row in model_rows
    }
    best_model = min(model_p50.values())
    largest_model_id = str(config["models"][-1]["model_id"])
    largest_degradation = (
        model_p50[largest_model_id] / best_model - 1.0
        if best_model > 0
        else math.inf
    )

    gate = config["gate"]
    correctness_gate = (
        output_token_mismatches
        <= int(gate["maximum_output_token_mismatches"])
        and registration_mismatches
        <= int(gate["maximum_registration_mismatches"])
        and rank_or_trace_mismatches
        <= int(gate["maximum_rank_or_control_mismatches"])
        and frozen_registration_hash_mismatches == 0
    )
    population_gate = (
        len(model_rows) == int(gate["expected_model_count"])
        and len(case_rows) == int(gate["expected_prompt_case_count"])
        and len(family_values) == int(gate["required_prompt_family_count"])
        and all(
            int(row["warm_projection_trace_count"])
            == int(row["registered_projection_count"])
            for row in case_rows
            if int(row["warm_decode_step_count"]) > 0
        )
    )
    weight_gate = (
        percentile(fractions, 0.50)
        <= float(gate["maximum_p50_mandatory_weight_fraction"])
        and percentile(fractions, 0.90)
        <= float(gate["maximum_p90_mandatory_weight_fraction"])
        and all(
            value
            <= float(gate["maximum_family_p90_mandatory_weight_fraction"])
            for value in family_p90.values()
        )
    )
    operation_gate = (
        percentile(operations, 0.50)
        <= float(gate["maximum_p50_mandatory_operation_fraction"])
        and percentile(operations, 0.90)
        <= float(gate["maximum_p90_mandatory_operation_fraction"])
    )
    trend_gate = largest_degradation <= float(
        gate["maximum_largest_model_degradation"]
    )
    verified_replay_in_every_family = all(
        value > 0.0 for value in family_max_replay.values()
    )
    survives = all(
        (
            correctness_gate,
            population_gate,
            weight_gate,
            operation_gate,
            trend_gate,
            verified_replay_in_every_family,
        )
    )
    decision = (
        "PROMOTE_CAUSAL_EXACT_TEMPORAL_SPAN_REPLAY_TO_BITWISE_CACHE_GATE"
        if survives
        else str(config["failure_decision"])
    )

    summary = {
        "experiment": "EXP-069",
        "name": "pinned_causal_exact_temporal_span_necessary_gate",
        "phase": ["A", "B", "C-small-model-causal-trajectory"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "DERIVED": {
            "correctness_gate_pass": correctness_gate,
            "population_gate_pass": population_gate,
            "mandatory_weight_lower_bound_gate_pass": weight_gate,
            "mandatory_operation_lower_bound_gate_pass": operation_gate,
            "model_trend_gate_pass": trend_gate,
            "verified_replay_in_every_family_gate_pass": (
                verified_replay_in_every_family
            ),
            "causal_exact_temporal_span_replay_survives_gate": survives,
            "decision": decision,
            "logical_scope": (
                "a rank increase modulo any registered odd prime proves exact "
                "rational independence and therefore a mandatory full W*x pass; "
                "modular non-increase receives no replay credit without a separately "
                "verified exact coefficient witness"
            ),
            "favorable_free_grants": [
                "all uncertified non-increase calls are free in the mandatory lower bound",
                "coefficient discovery and rank metadata are excluded from the mandatory lower bound",
                "only exact duplicate witnesses are charged in fail-closed replay accounting",
            ],
            "failure_basis": (
                "certified-independent arrivals alone exceed the preregistered "
                "whole-execution weight/operation budget"
                if not weight_gate or not operation_gate
                else "insufficient verified exact replay coverage"
            ),
        },
        "MEASURED": {
            "model_count": len(model_rows),
            "case_count": len(case_rows),
            "prompt_family_count": len(family_values),
            "registration_row_count": len(registration_rows),
            "projection_trace_count": len(projection_rows),
            "output_token_mismatches": output_token_mismatches,
            "registration_mismatches": registration_mismatches,
            "rank_trace_or_control_mismatches": rank_or_trace_mismatches,
            "frozen_exp061_registration_hash_mismatches": (
                frozen_registration_hash_mismatches
            ),
            "p50_mandatory_weight_fraction_lower_bound": percentile(
                fractions, 0.50
            ),
            "p90_mandatory_weight_fraction_lower_bound": percentile(
                fractions, 0.90
            ),
            "p50_mandatory_operation_fraction_lower_bound": percentile(
                operations, 0.50
            ),
            "p90_mandatory_operation_fraction_lower_bound": percentile(
                operations, 0.90
            ),
            "minimum_mandatory_weight_fraction_lower_bound": min(fractions),
            "maximum_mandatory_weight_fraction_lower_bound": max(fractions),
            "family_p90_mandatory_weight_fraction_lower_bound": family_p90,
            "family_max_verified_exact_replay_hit_fraction": family_max_replay,
            "total_verified_exact_replay_hits": sum(
                int(row["verified_exact_replay_hits"])
                for row in case_rows
            ),
            "maximum_case_verified_exact_replay_hit_fraction": max(
                float(row["verified_exact_replay_hit_fraction"])
                for row in case_rows
            ),
            "p50_fail_closed_weight_fraction": percentile(
                [float(row["fail_closed_weight_fraction"]) for row in case_rows],
                0.50,
            ),
            "p90_fail_closed_weight_fraction": percentile(
                [float(row["fail_closed_weight_fraction"]) for row in case_rows],
                0.90,
            ),
            "p50_basis_cache_to_q4_projection_population_ratio": percentile(
                [
                    float(row["basis_cache_to_q4_projection_population_ratio"])
                    for row in case_rows
                ],
                0.50,
            ),
            "model_p50_mandatory_weight_fraction_lower_bound": model_p50,
            "largest_model_degradation_fraction": largest_degradation,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_ns": time.perf_counter_ns() - started,
        },
        "UNVERIFIED": [
            "complete exact coefficient-witness search for modular non-increases",
            "bitwise floating-point output replay equivalence",
            "physical temporal replay cache and kernel",
            "actual Transformer operation replacement",
            "405B causal trajectory rank",
            "405B execution",
            "8 GiB total runtime state",
            "CUDA PCIe SSD TTFT and tokens per second",
        ],
        "claim_boundary": {
            "exact_float32_dyadic_modular_rank": "MEASURED",
            "mandatory_full_pass_lower_bound": "MEASURED",
            "model_exact_replay_witness_search": "DUPLICATES ONLY",
            "bitwise_floating_point_replay": "NOT TESTED",
            "physical_cache_kernel": "NOT TESTED",
            "405b_execution": "NOT TESTED",
            "8_gib_vram": "NOT TESTED",
            "target_hardware": "NOT TESTED",
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(arguments.config),
            "exp061_registration_rows_sha256": sha256_file(EXP061_REGISTRATIONS),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "tokenizer_id": str(tokenizer_entry["model_id"]),
            "tokenizer_revision": str(tokenizer_entry["revision"]),
        },
    }

    dump_rows(output / "raw/registration_rows.jsonl", registration_rows)
    dump_rows(output / "raw/projection_rows.jsonl", projection_rows)
    dump_rows(output / "raw/case_rows.jsonl", case_rows)
    dump_rows(output / "raw/phase_rows.jsonl", phase_rows)
    dump_rows(output / "raw/model_rows.jsonl", model_rows)
    dump_rows(output / "raw/control_rows.jsonl", controls)
    dump(output / "summary.json", summary)
    dump(output / "processed/aggregate.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(
        str(config["evidence_ceiling"]) + "\n", encoding="utf-8"
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not correctness_gate or not population_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
