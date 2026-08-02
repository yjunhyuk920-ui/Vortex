from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_hot_candidate_coverage import DEFAULT_WIDTHS, last_logits
from scripts.run_oracle_block_shared_adjoint import (
    DEFAULT_BUILD_PROMPTS,
    encode_prompt,
    require_transformers,
)
from vortex_runtime.candidate_coverage import (
    CandidateCoverageRow,
    coverage_at_k,
    token_rank,
    top1_margin,
)
from vortex_runtime.local_affine_dictionary import (
    LocalAffineDictionaryLinearModule,
    build_local_affine_dictionaries,
    quantize_local_affine_dictionary,
    replace_with_local_affine_dictionary_modules,
)
from vortex_runtime.rank_frontier import rank_budget_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a routed local affine response-capsule dictionary from fixed "
            "disjoint prompts plus the exact user prompt, quantize it, and "
            "measure exact-token coverage on unseen continuation positions."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--local-rank", type=int, required=True)
    parser.add_argument("--capsule-bits", type=int, required=True)
    parser.add_argument("--rank-rtol", type=float, default=1e-6)
    parser.add_argument("--kmeans-iterations", type=int, default=12)
    parser.add_argument("--build-prompt", action="append", dest="build_prompts")
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument("--suffix", action="append", dest="suffixes")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local_affine_dictionary_point.json"),
    )
    return parser.parse_args()


def capture_exact_prompt_dataset(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    replacements: Mapping[str, LocalAffineDictionaryLinearModule],
    prompts: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[int]]:
    input_rows: dict[str, list[torch.Tensor]] = {
        name: [] for name in replacements
    }
    output_rows: dict[str, list[torch.Tensor]] = {
        name: [] for name in replacements
    }
    handles: list[Any] = []
    for name, module in replacements.items():
        def pre_hook(
            _module: torch.nn.Module,
            hook_args: tuple[torch.Tensor, ...],
            *,
            key: str = name,
        ) -> None:
            tensor = hook_args[0].detach().to("cpu", torch.float32)
            input_rows[key].append(tensor.reshape(-1, tensor.shape[-1]))

        def output_hook(
            _module: torch.nn.Module,
            _hook_args: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            key: str = name,
        ) -> None:
            tensor = output.detach().to("cpu", torch.float32)
            output_rows[key].append(tensor.reshape(-1, tensor.shape[-1]))

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(output_hook))

    token_counts: list[int] = []
    for module in replacements.values():
        module.set_mode("exact")
    try:
        for prompt in prompts:
            encoded = encode_prompt(tokenizer, prompt, device)
            token_counts.append(int(encoded["input_ids"].shape[-1]))
            with torch.inference_mode():
                model(
                    **encoded,
                    use_cache=False,
                    return_dict=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    captured_inputs = {
        name: torch.cat(rows, dim=0) for name, rows in input_rows.items()
    }
    captured_outputs = {
        name: torch.cat(rows, dim=0) for name, rows in output_rows.items()
    }
    return captured_inputs, captured_outputs, token_counts


def relative_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    reference32 = reference.detach().to("cpu", torch.float32)
    estimate32 = estimate.detach().to("cpu", torch.float32)
    numerator = torch.linalg.vector_norm(reference32 - estimate32)
    denominator = torch.linalg.vector_norm(reference32)
    return float((numerator / torch.clamp(denominator, min=1e-12)).item())


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.clusters <= 0 or args.local_rank < 0:
        raise SystemExit("tokens/clusters must be positive and local rank non-negative")
    if not 2 <= args.capsule_bits <= 16:
        raise SystemExit("capsule bits must be between 2 and 16")
    if args.rank_rtol < 0 or args.kmeans_iterations <= 0:
        raise SystemExit("rank tolerance and k-means iterations are invalid")

    stored_equivalent_rank = args.clusters * (args.local_rank + 1)
    active_equivalent_rank = args.clusters + args.local_rank + 1
    stored_budget = rank_budget_point(
        stored_equivalent_rank,
        capsule_bits=args.capsule_bits,
    )
    active_budget = rank_budget_point(
        active_equivalent_rank,
        capsule_bits=args.capsule_bits,
    )
    budget_pass = (
        stored_budget.memory_pass
        and active_budget.traffic_pass
        and active_budget.compute_pass
    )
    if not budget_pass:
        raise SystemExit(
            "local dictionary point violates the fixed 405B envelope: "
            f"stored={stored_budget.to_dict()} active={active_budget.to_dict()}"
        )

    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    suffixes = tuple(args.suffixes or ("self_attn.o_proj", "mlp.down_proj"))
    replacements = replace_with_local_affine_dictionary_modules(
        model,
        suffixes=suffixes,
    )
    if not replacements:
        raise RuntimeError(f"no modules matched {suffixes}")

    build_prompts = list(args.build_prompts or DEFAULT_BUILD_PROMPTS)
    capture_prompts = [*build_prompts, args.eval_prompt]
    started = time.perf_counter()
    captured_inputs, captured_outputs, prompt_token_counts = (
        capture_exact_prompt_dataset(
            model=model,
            tokenizer=tokenizer,
            device=device,
            replacements=replacements,
            prompts=capture_prompts,
        )
    )
    minimum_vectors = min(
        int(tensor.shape[0]) for tensor in captured_inputs.values()
    )
    if args.clusters > minimum_vectors:
        raise RuntimeError(
            f"cluster count {args.clusters} exceeds minimum captured vectors "
            f"{minimum_vectors}"
        )

    build_stats = build_local_affine_dictionaries(
        replacements,
        captured_inputs=captured_inputs,
        captured_outputs=captured_outputs,
        clusters=args.clusters,
        local_rank=args.local_rank,
        rank_rtol=args.rank_rtol,
        kmeans_iterations=args.kmeans_iterations,
    )
    quantization: dict[str, dict[str, int | float]] = {}
    for name, module in replacements.items():
        assert module.dictionary is not None
        quantization[name] = quantize_local_affine_dictionary(
            module.dictionary,
            bits=args.capsule_bits,
        ).to_dict()

    post_quantization_errors: dict[str, float] = {}
    for name, module in replacements.items():
        module.set_mode("dictionary")
        input_tensor = captured_inputs[name].to(
            device=module.exact.weight.device,
            dtype=module.exact.weight.dtype,
        )
        with torch.inference_mode():
            estimate = module(input_tensor)
        post_quantization_errors[name] = relative_error(
            captured_outputs[name],
            estimate,
        )

    encoded = encode_prompt(tokenizer, args.eval_prompt, device)
    prefix = encoded["input_ids"]
    rows: list[CandidateCoverageRow] = []
    for position in range(args.tokens):
        for module in replacements.values():
            module.set_mode("exact")
        exact_logits = last_logits(model, prefix)
        exact_token = int(torch.argmax(exact_logits).item())

        for module in replacements.values():
            module.set_mode("dictionary")
        hot_logits = last_logits(model, prefix)
        hot_token = int(torch.argmax(hot_logits).item())
        rows.append(
            CandidateCoverageRow(
                position=position,
                exact_token=exact_token,
                hot_token=hot_token,
                exact_token_rank=token_rank(hot_logits, exact_token),
                hot_top1_margin=top1_margin(hot_logits),
                exact_logit_gap_from_hot_top1=float(
                    (hot_logits[hot_token] - hot_logits[exact_token]).item()
                ),
            )
        )
        prefix = torch.cat(
            (
                prefix,
                torch.tensor([[exact_token]], dtype=torch.long, device=device),
            ),
            dim=-1,
        )

    exact_ranks = [row.exact_token_rank for row in rows]
    coverage = coverage_at_k(rows, DEFAULT_WIDTHS)
    first_divergence = next((row for row in rows if not row.exact_match), None)
    first_rank = (
        None if first_divergence is None else first_divergence.exact_token_rank
    )
    coverage_pass = (
        (first_rank is None or first_rank <= 32)
        and coverage["32"] >= 0.95
    )
    dictionary_pass = budget_pass and coverage_pass

    actual_stored_columns = [
        stats.stored_response_columns for stats in build_stats.values()
    ]
    actual_active_columns = [
        stats.active_rank_maximum + stats.routing_centroid_columns
        for stats in build_stats.values()
    ]
    result = {
        "evidence_level": "E1 causal routed local affine dictionary",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "clusters": args.clusters,
        "local_rank": args.local_rank,
        "capsule_bits": args.capsule_bits,
        "rank_rtol": args.rank_rtol,
        "kmeans_iterations": args.kmeans_iterations,
        "build_prompts": build_prompts,
        "eval_prompt": args.eval_prompt,
        "capture_prompt_token_counts": prompt_token_counts,
        "captured_vectors_per_module": minimum_vectors,
        "causal_contract": (
            "Dictionary centroids, local bases, and response images use only "
            "fixed disjoint prompt prefills plus the exact user prompt prefill. "
            "No continuation target, continuation gradient, or continuation "
            "activation is available before the dictionary is frozen."
        ),
        "budget_contract": (
            "Stored rank is K*(local_rank+1). Active traffic/compute conservatively "
            "charges all K input centroids for routing plus one selected affine "
            "centroid and local basis, for K+local_rank+1 equivalent columns."
        ),
        "stored_equivalent_rank": stored_equivalent_rank,
        "active_equivalent_rank": active_equivalent_rank,
        "stored_budget": stored_budget.to_dict(),
        "active_budget": active_budget.to_dict(),
        "actual_stored_response_columns": {
            "minimum": min(actual_stored_columns),
            "maximum": max(actual_stored_columns),
            "mean": sum(actual_stored_columns) / len(actual_stored_columns),
        },
        "actual_active_equivalent_columns": {
            "minimum": min(actual_active_columns),
            "maximum": max(actual_active_columns),
            "mean": sum(actual_active_columns) / len(actual_active_columns),
        },
        "build_statistics": {
            name: stats.to_dict() for name, stats in build_stats.items()
        },
        "quantization": quantization,
        "post_quantization_training_reconstruction": {
            "maximum_module_output_relative_error": max(
                post_quantization_errors.values()
            ),
            "mean_module_output_relative_error": sum(
                post_quantization_errors.values()
            ) / len(post_quantization_errors),
            "per_module": post_quantization_errors,
        },
        "evaluated_unseen_tokens": len(rows),
        "same_context_contract": (
            "Exact and dictionary paths receive the same authoritative exact "
            "prefix at every unseen continuation position."
        ),
        "exact_top1_match_rate": sum(row.exact_match for row in rows) / len(rows),
        "coverage_at_k": coverage,
        "rank_statistics": {
            "minimum": min(exact_ranks),
            "maximum": max(exact_ranks),
            "mean": sum(exact_ranks) / len(exact_ranks),
        },
        "first_divergence": (
            None
            if first_divergence is None
            else {
                "position": first_divergence.position,
                "exact_token": first_divergence.exact_token,
                "hot_token": first_divergence.hot_token,
                "exact_token_rank": first_divergence.exact_token_rank,
                "hot_top1_margin": first_divergence.hot_top1_margin,
                "exact_logit_gap_from_hot_top1": (
                    first_divergence.exact_logit_gap_from_hot_top1
                ),
            }
        ),
        "rows": [row.__dict__ | {"exact_match": row.exact_match} for row in rows],
        "decision_rule": (
            "advance only when stored memory, active routing traffic/compute, "
            "first-divergence rank <=32, and top-32 coverage >=0.95 all pass"
        ),
        "decision": (
            "advance routed local affine dictionary"
            if dictionary_pass
            else "reject tested routed local affine dictionary point"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                "clusters": args.clusters,
                "local_rank": args.local_rank,
                "bits": args.capsule_bits,
                "stored_equivalent_rank": stored_equivalent_rank,
                "active_equivalent_rank": active_equivalent_rank,
                "post_quant_training_error": result[
                    "post_quantization_training_reconstruction"
                ],
                "exact_top1_match_rate": result["exact_top1_match_rate"],
                "top32_coverage": result["coverage_at_k"]["32"],
                "rank_statistics": result["rank_statistics"],
                "first_divergence": result["first_divergence"],
                "decision": result["decision"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
