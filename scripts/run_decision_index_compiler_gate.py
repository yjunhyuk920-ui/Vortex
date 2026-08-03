from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.decision_index_compiler import (
    DecisionTrace,
    build_exact_decision_graph,
    evaluate_heldout_coverage,
    export_graph_to_compact40,
    grammar_sha256,
    graph_growth_frontier,
    load_bounded_grammar,
    replay_compiled_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a bounded exact TinyLlama decision graph and export "
            "it to the compact40 host-indexed VM."
        )
    )
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    parser.add_argument(
        "--grammar",
        type=Path,
        default=Path(
            "experiments/decision_index_compiler_grammar.json"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/decision_index_compiler_gate.json"
        ),
    )
    parser.add_argument(
        "--vm-output",
        type=Path,
        default=Path(
            "results/decision_index_compiler_compact40.vtx"
        ),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(
            "results/decision_index_compiler_manifest.json"
        ),
    )
    return parser.parse_args()


def require_transformers() -> tuple[Any, Any, str]:
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "install transformers and sentencepiece for Experiment 045"
        ) from error
    return AutoModelForCausalLM, AutoTokenizer, transformers.__version__


def load_model(
    AutoModelForCausalLM: Any,
    *,
    model_name: str,
    device: torch.device,
) -> torch.nn.Module:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
    model.to(device)
    model.eval()
    return model


def encode_chat_prompt(
    tokenizer: Any,
    *,
    system_message: str,
    user_message: str,
    device: torch.device,
) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return encoded["input_ids"].to(device)


def collect_exact_greedy_trace(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt,
    system_message: str,
    horizon: int,
    device: torch.device,
) -> DecisionTrace:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    input_ids = encode_chat_prompt(
        tokenizer,
        system_message=system_message,
        user_message=prompt.text,
        device=device,
    )
    began = time.perf_counter_ns()
    with torch.no_grad():
        prefill = model(input_ids=input_ids, use_cache=True)
        next_token = torch.argmax(
            prefill.logits[:, -1, :],
            dim=-1,
        )
        past_key_values = prefill.past_key_values

    continuation = [int(next_token[0].item())]
    control = next_token[:, None]
    forward_calls = 1
    for _step in range(1, horizon):
        with torch.no_grad():
            decoded = model(
                input_ids=control,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = torch.argmax(
                decoded.logits[:, -1, :],
                dim=-1,
            )
            past_key_values = decoded.past_key_values
        continuation.append(int(next_token[0].item()))
        control = next_token[:, None]
        forward_calls += 1
    elapsed = time.perf_counter_ns() - began

    eos_token_id = tokenizer.eos_token_id
    eos_position = None
    if eos_token_id is not None:
        for position, token_id in enumerate(continuation):
            if token_id == eos_token_id:
                eos_position = position
                break

    return DecisionTrace(
        prompt=prompt,
        prompt_token_ids=tuple(
            int(value) for value in input_ids[0].detach().cpu().tolist()
        ),
        continuation_token_ids=tuple(continuation),
        model_forward_calls=forward_calls,
        elapsed_ns=elapsed,
        eos_position=eos_position,
    )


def maximum_identical_run(tokens: tuple[int, ...]) -> int:
    if not tokens:
        return 0
    best = 1
    current = 1
    for previous, value in zip(tokens, tokens[1:]):
        if previous == value:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def trace_diagnostic(trace: DecisionTrace) -> dict:
    return {
        "prompt_id": trace.prompt.prompt_id,
        "split": trace.prompt.split,
        "prompt_tokens": len(trace.prompt_token_ids),
        "continuation_token_ids": list(trace.continuation_token_ids),
        "unique_continuation_tokens": trace.unique_continuation_tokens,
        "maximum_identical_run": maximum_identical_run(
            trace.continuation_token_ids
        ),
        "eos_position": trace.eos_position,
        "model_forward_calls": trace.model_forward_calls,
        "elapsed_ns": trace.elapsed_ns,
    }


def main() -> None:
    args = parse_args()
    grammar, prompts = load_bounded_grammar(args.grammar)
    horizons = [int(value) for value in grammar["horizons"]]
    maximum_horizon = int(grammar["maximum_horizon"])
    codebook_limit = int(grammar["vm_token_codebook_limit"])
    if max(horizons) != maximum_horizon:
        raise ValueError("maximum horizon and frontier disagree")

    AutoModelForCausalLM, AutoTokenizer, transformers_version = (
        require_transformers()
    )
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(
        AutoModelForCausalLM,
        model_name=args.model,
        device=device,
    )

    system_message = str(grammar["system_message"])
    compiled_prompts = [
        prompt for prompt in prompts if prompt.split == "compiled"
    ]
    heldout_prompts = [
        prompt for prompt in prompts if prompt.split == "heldout"
    ]
    duplicate_prompts = [
        prompt
        for prompt in prompts
        if prompt.split == "duplicate_control"
    ]

    trace_began = time.perf_counter_ns()
    unique_traces = [
        collect_exact_greedy_trace(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            system_message=system_message,
            horizon=maximum_horizon,
            device=device,
        )
        for prompt in compiled_prompts + heldout_prompts
    ]
    trace_elapsed_ns = time.perf_counter_ns() - trace_began
    trace_by_id = {
        trace.prompt.prompt_id: trace for trace in unique_traces
    }
    compiled_traces = [
        trace_by_id[prompt.prompt_id] for prompt in compiled_prompts
    ]
    heldout_traces = [
        trace_by_id[prompt.prompt_id] for prompt in heldout_prompts
    ]

    duplicate_traces: list[DecisionTrace] = []
    for duplicate_prompt in duplicate_prompts:
        source = trace_by_id[str(duplicate_prompt.duplicate_of)]
        duplicate_traces.append(
            replace(
                source,
                prompt=duplicate_prompt,
                model_forward_calls=0,
                elapsed_ns=0,
            )
        )
    graph_traces = compiled_traces + duplicate_traces

    unique_growth = graph_growth_frontier(
        compiled_traces,
        horizons,
        codebook_limit=codebook_limit,
    )
    controlled_growth = graph_growth_frontier(
        graph_traces,
        horizons,
        codebook_limit=codebook_limit,
    )

    graph_began = time.perf_counter_ns()
    graph = build_exact_decision_graph(
        graph_traces,
        horizon=maximum_horizon,
        codebook_limit=codebook_limit,
    )
    graph_elapsed_ns = time.perf_counter_ns() - graph_began

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.vm_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    export_began = time.perf_counter_ns()
    build, manifest_bytes = export_graph_to_compact40(
        graph,
        vm_path=args.vm_output,
        manifest_path=args.manifest_output,
        metadata={
            "experiment": "045_decision_index_compiler_gate",
            "model": args.model,
            "grammar": grammar["name"],
            "grammar_sha256": grammar_sha256(args.grammar),
            "transformers_version": transformers_version,
            "tokenizer_class": tokenizer.__class__.__name__,
            "model_class": model.__class__.__name__,
        },
    )
    export_elapsed_ns = time.perf_counter_ns() - export_began

    replay = replay_compiled_graph(
        graph,
        graph_traces,
        vm_path=args.vm_output,
    )
    heldout = evaluate_heldout_coverage(graph, heldout_traces)

    duplicate_reuse = []
    prompt_to_start = dict(
        zip(graph.prompt_order, graph.prompt_start_addresses)
    )
    for duplicate in duplicate_prompts:
        source_id = str(duplicate.duplicate_of)
        duplicate_reuse.append(
            {
                "duplicate_prompt_id": duplicate.prompt_id,
                "source_prompt_id": source_id,
                "same_start_address": (
                    prompt_to_start[duplicate.prompt_id]
                    == prompt_to_start[source_id]
                ),
                "start_address": prompt_to_start[duplicate.prompt_id],
            }
        )

    compiled_distinct_tokens = sorted(
        {
            token
            for trace in compiled_traces
            for token in trace.continuation_token_ids
        }
    )
    codebook_tokens = [
        tokenizer.convert_ids_to_tokens(token_id)
        for token_id in graph.token_codebook
    ]
    codebook_text = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in graph.token_codebook
    ]

    unique_final = unique_growth[-1]
    duplicate_only_reuse = (
        controlled_growth[-1].exact_duplicate_records_removed
        == maximum_horizon * len(duplicate_traces)
    )
    nontrivial_exact_reuse = (
        unique_final.exact_duplicate_records_removed > 0
    )
    implementation_passes = bool(
        len(compiled_prompts) == 8
        and len(heldout_prompts) == 4
        and len(graph.token_codebook) <= codebook_limit
        and replay.all_exact
        and all(item["same_start_address"] for item in duplicate_reuse)
        and duplicate_only_reuse
        and build.atomic_replace
        and build.temporary_file_removed
    )
    architecture_advances = bool(
        implementation_passes
        and (
            nontrivial_exact_reuse
            or heldout.coverage > 0.0
        )
    )

    compile_model_calls = sum(
        trace.model_forward_calls for trace in compiled_traces
    )
    heldout_model_calls = sum(
        trace.model_forward_calls for trace in heldout_traces
    )

    payload = {
        "experiment": "045_bounded_exact_decision_index_compiler",
        "evidence_level": (
            "E1/E2 real TinyLlama bounded-grammar compiler and exact "
            "compact40 replay"
        ),
        "model": args.model,
        "model_class": model.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "transformers_version": transformers_version,
        "grammar": {
            "name": grammar["name"],
            "sha256": grammar_sha256(args.grammar),
            "full_combinations": len(compiled_prompts) + len(heldout_prompts),
            "compiled_combinations": len(compiled_prompts),
            "heldout_combinations": len(heldout_prompts),
            "duplicate_controls": len(duplicate_prompts),
            "horizons": horizons,
            "maximum_horizon": maximum_horizon,
            "compiled_prompt_ids": [
                prompt.prompt_id for prompt in compiled_prompts
            ],
            "heldout_prompt_ids": [
                prompt.prompt_id for prompt in heldout_prompts
            ],
        },
        "build_accounting": {
            "compiled_model_forward_calls": compile_model_calls,
            "heldout_ground_truth_forward_calls": heldout_model_calls,
            "duplicate_control_forward_calls": 0,
            "trace_collection_elapsed_ns": trace_elapsed_ns,
            "graph_build_elapsed_ns": graph_elapsed_ns,
            "vm_export_elapsed_ns": export_elapsed_ns,
            "vm_file_bytes": build.file_bytes,
            "manifest_bytes": manifest_bytes,
            "graph_nodes": graph.node_count,
            "graph_starts": graph.start_count,
        },
        "growth_without_duplicate_control": [
            point.to_dict() for point in unique_growth
        ],
        "growth_with_duplicate_control": [
            point.to_dict() for point in controlled_growth
        ],
        "token_codebook": {
            "limit": codebook_limit,
            "size": len(graph.token_codebook),
            "token_ids": list(graph.token_codebook),
            "tokenizer_tokens": codebook_tokens,
            "decoded_text": codebook_text,
            "compiled_distinct_token_ids": compiled_distinct_tokens,
        },
        "duplicate_control": duplicate_reuse,
        "compiled_vm_replay": replay.to_dict(),
        "heldout_coverage": heldout.to_dict(),
        "trace_diagnostics": [
            trace_diagnostic(trace)
            for trace in compiled_traces + heldout_traces
        ],
        "scope_separation": {
            "unmodified_real_checkpoint_used": True,
            "training_or_finetuning_used": False,
            "finite_grammar_complete": implementation_passes,
            "compiled_paths_replay_without_model": replay.all_exact,
            "exact_duplicate_prompt_reuse_proven": duplicate_only_reuse,
            "semantic_or_approximate_state_merging_used": False,
            "nontrivial_exact_reuse_excluding_duplicate_prompts": (
                nontrivial_exact_reuse
            ),
            "heldout_grammar_generalization_proven": (
                heldout.coverage > 0.0
            ),
            "universal_prompt_coverage_proven": False,
            "real_405b_execution_performed": False,
            "ci_build_time_target_representative": False,
        },
        "conclusion": {
            "compiler_implementation_passes": implementation_passes,
            "architecture_advances_beyond_exact_memoization": (
                architecture_advances
            ),
            "compiled_grammar_coverage": (
                1.0 if implementation_passes else 0.0
            ),
            "heldout_coverage": heldout.coverage,
            "unique_graph_nodes_at_max_horizon": graph.node_count,
            "path_records_without_duplicate_control": (
                len(compiled_traces) * maximum_horizon
            ),
            "nontrivial_exact_nodes_removed": (
                unique_final.exact_duplicate_records_removed
            ),
            "decision": (
                "advance exact state quotienting to a broader grammar"
                if architecture_advances
                else (
                    "accept bounded compiler implementation but reject "
                    "exact-prefix memoization as a general execution "
                    "mechanism"
                    if implementation_passes
                    else "compiler Gate failed and requires repair"
                )
            ),
            "fixed_target_status": (
                "unsolved: bounded TinyLlama trace compilation does not "
                "establish arbitrary prompt coverage, 405B construction, "
                "or target hardware performance"
            ),
        },
        "next_obligation": (
            "if exact-prefix reuse and held-out coverage remain zero, test "
            "certified state equivalence or adaptive on-demand compilation; "
            "do not scale raw prefix enumeration as a universal runtime"
        ),
    }

    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
