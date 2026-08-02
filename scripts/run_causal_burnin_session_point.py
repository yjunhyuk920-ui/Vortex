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
from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.burnin_budget import (
    amortization_frontier,
    minimum_amortization_horizon,
)
from vortex_runtime.candidate_coverage import (
    CandidateCoverageRow,
    coverage_at_k,
    token_rank,
    top1_margin,
)
from vortex_runtime.capsule_quantization import fake_quantize_response_capsules
from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
    replace_with_decision_tile_modules,
)
from vortex_runtime.feasibility import default_gate0_report
from vortex_runtime.rank_frontier import rank_budget_point
from vortex_runtime.session_prefill_basis import compile_session_response_bases


DEFAULT_HORIZONS = (256, 512, 1024, 2048, 4096)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a small exact causal burn-in, compile a local response "
            "basis from the exact prompt plus already-generated burn-in module "
            "inputs/outputs, quantize it, and evaluate only unseen continuation."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--burnin-tokens", type=int, required=True)
    parser.add_argument("--eval-tokens", type=int, default=32)
    parser.add_argument("--rank-limit", type=int, default=72)
    parser.add_argument("--capsule-bits", type=int, default=8)
    parser.add_argument("--rank-rtol", type=float, default=1e-6)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument("--suffix", action="append", dest="suffixes")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("causal_burnin_session_point.json"),
    )
    return parser.parse_args()


def clear_capsules(
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
    *,
    max_rank: int,
) -> None:
    for module in replacements.values():
        module.atlas.input_basis = torch.empty(
            (module.exact.in_features, 0),
            dtype=module.atlas.basis_dtype,
        )
        module.atlas.output_image = torch.empty(
            (module.exact.out_features, 0),
            dtype=module.atlas.basis_dtype,
        )
        module.atlas.max_rank = max_rank


def capture_exact_prefix_io(
    *,
    model: torch.nn.Module,
    replacements: Mapping[str, DecisionResidualTileAtlasLinearModule],
    input_ids: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    captured_inputs: dict[str, torch.Tensor] = {}
    captured_outputs: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    for name, module in replacements.items():
        def pre_hook(
            _module: torch.nn.Module,
            hook_args: tuple[torch.Tensor, ...],
            *,
            key: str = name,
        ) -> None:
            captured_inputs[key] = hook_args[0].detach().to("cpu")

        def output_hook(
            _module: torch.nn.Module,
            _hook_args: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            key: str = name,
        ) -> None:
            captured_outputs[key] = output.detach().to("cpu")

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(output_hook))

    for module in replacements.values():
        module.set_mode("exact")
    try:
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    return captured_inputs, captured_outputs


def relative_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    reference32 = reference.detach().to("cpu", torch.float32)
    estimate32 = estimate.detach().to("cpu", torch.float32)
    numerator = torch.linalg.vector_norm(reference32 - estimate32)
    denominator = torch.linalg.vector_norm(reference32)
    return float((numerator / torch.clamp(denominator, min=1e-12)).item())


def main() -> None:
    args = parse_args()
    if args.burnin_tokens < 0:
        raise SystemExit("burn-in token count must be non-negative")
    if args.eval_tokens <= 0 or args.rank_limit <= 0:
        raise SystemExit("evaluation tokens and rank limit must be positive")
    if not 2 <= args.capsule_bits <= 16:
        raise SystemExit("capsule bits must be between 2 and 16")
    if args.rank_rtol < 0:
        raise SystemExit("rank tolerance must be non-negative")

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
    replacements = replace_with_decision_tile_modules(
        model,
        suffixes=suffixes,
        max_rank=args.rank_limit,
    )
    if not replacements:
        raise RuntimeError(f"no modules matched {suffixes}")

    encoded = encode_prompt(tokenizer, args.eval_prompt, device)
    prefix = encoded["input_ids"]
    prompt_tokens = int(prefix.shape[-1])
    burnin_token_ids: list[int] = []
    started = time.perf_counter()

    for _ in range(args.burnin_tokens):
        for module in replacements.values():
            module.set_mode("exact")
        logits = last_logits(model, prefix)
        token = int(torch.argmax(logits).item())
        burnin_token_ids.append(token)
        prefix = torch.cat(
            (
                prefix,
                torch.tensor([[token]], dtype=torch.long, device=device),
            ),
            dim=-1,
        )

    captured_inputs, captured_outputs = capture_exact_prefix_io(
        model=model,
        replacements=replacements,
        input_ids=prefix,
    )
    clear_capsules(replacements, max_rank=args.rank_limit)
    compile_stats = compile_session_response_bases(
        replacements,
        captured_inputs=captured_inputs,
        captured_outputs=captured_outputs,
        max_rank=args.rank_limit,
        rank_rtol=args.rank_rtol,
    )
    compiled_ranks = [item.compiled_rank for item in compile_stats.values()]
    quantization, per_module_quantization = fake_quantize_response_capsules(
        replacements,
        bits=args.capsule_bits,
    )

    post_quantization_errors: dict[str, float] = {}
    for name, module in replacements.items():
        module.set_mode("project")
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

    rows: list[CandidateCoverageRow] = []
    for position in range(args.eval_tokens):
        for module in replacements.values():
            module.set_mode("exact")
        exact_logits = last_logits(model, prefix)
        exact_token = int(torch.argmax(exact_logits).item())

        for module in replacements.values():
            module.set_mode("project")
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

    first_divergence = next((row for row in rows if not row.exact_match), None)
    exact_ranks = [row.exact_token_rank for row in rows]
    coverage = coverage_at_k(rows, DEFAULT_WIDTHS)
    maximum_compiled_rank = max(compiled_ranks)
    hot_budget = rank_budget_point(
        maximum_compiled_rank,
        capsule_bits=args.capsule_bits,
    )
    target_report = default_gate0_report(1.0)
    cold_traffic = float(target_report["traffic"]["cold_full_repair_gib"])
    cold_compute = float(target_report["compute"]["cold_full_repair_gflop"])
    amortization = amortization_frontier(
        exact_burnin_tokens=args.burnin_tokens,
        horizons=DEFAULT_HORIZONS,
        hot_traffic_gib_per_token=hot_budget.hot_traffic_gib_per_token,
        cold_exact_traffic_gib_per_token=cold_traffic,
        traffic_limit_gib_per_token=hot_budget.traffic_limit_gib_per_token,
        hot_compute_gflop_per_token=hot_budget.hot_compute_gflop_per_token,
        cold_exact_compute_gflop_per_token=cold_compute,
        compute_limit_gflop_per_token=hot_budget.compute_limit_gflop_per_token,
    )
    minimum_traffic_horizon = minimum_amortization_horizon(
        exact_burnin_tokens=args.burnin_tokens,
        hot_cost_per_token=hot_budget.hot_traffic_gib_per_token,
        cold_exact_cost_per_token=cold_traffic,
        cost_limit_per_token=hot_budget.traffic_limit_gib_per_token,
    )
    minimum_compute_horizon = minimum_amortization_horizon(
        exact_burnin_tokens=args.burnin_tokens,
        hot_cost_per_token=hot_budget.hot_compute_gflop_per_token,
        cold_exact_cost_per_token=cold_compute,
        cost_limit_per_token=hot_budget.compute_limit_gflop_per_token,
    )
    first_rank = (
        None if first_divergence is None else first_divergence.exact_token_rank
    )
    coverage_pass = (
        (first_rank is None or first_rank <= 32)
        and coverage["32"] >= 0.95
    )
    warm_candidate_pass = hot_budget.pass_all and coverage_pass
    horizon_4096 = next(
        item for item in amortization if item.horizon_tokens == 4096
    )

    result = {
        "evidence_level": "E1 causal exact-burnin local trajectory capsule",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "prompt_tokens": prompt_tokens,
        "exact_burnin_tokens": args.burnin_tokens,
        "burnin_token_ids": burnin_token_ids,
        "evaluated_unseen_tokens": len(rows),
        "rank_limit": args.rank_limit,
        "capsule_bits": args.capsule_bits,
        "rank_rtol": args.rank_rtol,
        "causal_contract": (
            "Burn-in tokens are generated one by one by the exact model before "
            "compilation. The capsule receives only exact prompt-plus-burn-in "
            "module inputs and outputs. Every evaluated continuation token is "
            "unseen when the capsule is frozen and is used only for measurement."
        ),
        "compiled_rank_statistics": {
            "minimum": min(compiled_ranks),
            "maximum": maximum_compiled_rank,
            "mean": sum(compiled_ranks) / len(compiled_ranks),
        },
        "prompt_burnin_reconstruction_before_quantization": {
            "maximum_input_relative_error": max(
                item.input_reconstruction_relative_error
                for item in compile_stats.values()
            ),
            "maximum_output_relative_error": max(
                item.output_reconstruction_relative_error
                for item in compile_stats.values()
            ),
        },
        "prompt_burnin_reconstruction_after_quantization": {
            "maximum_module_output_relative_error": max(
                post_quantization_errors.values()
            ),
            "mean_module_output_relative_error": sum(
                post_quantization_errors.values()
            ) / len(post_quantization_errors),
            "per_module": post_quantization_errors,
        },
        "quantization": {
            "aggregate": quantization.to_dict(),
            "per_module": per_module_quantization,
        },
        "hot_budget": hot_budget.to_dict(),
        "startup_exact_cost": {
            "full_exact_traffic_gib_per_burnin_token": cold_traffic,
            "full_exact_compute_gflop_per_burnin_token": cold_compute,
            "minimum_traffic_amortization_horizon": minimum_traffic_horizon,
            "minimum_compute_amortization_horizon": minimum_compute_horizon,
            "frontier": [item.to_dict() for item in amortization],
            "horizon_4096_pass": horizon_4096.pass_all,
        },
        "same_context_contract": (
            "Exact and frozen local-capsule paths receive the same authoritative "
            "exact prefix at every unseen continuation position."
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
            "advance warm decode only when the frozen local capsule fits the "
            "405B hot memory/traffic/compute gate, first-divergence exact-token "
            "rank is at most 32, and top-32 coverage is at least 0.95; startup "
            "exact cost is reported separately over finite horizons"
        ),
        "warm_decode_candidate_pass": warm_candidate_pass,
        "full_session_4096_pass": warm_candidate_pass and horizon_4096.pass_all,
        "decision": (
            "advance causal burnin local trajectory capsule"
            if warm_candidate_pass
            else "reject tested causal burnin local trajectory point"
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
                "burnin_tokens": args.burnin_tokens,
                "compiled_rank_statistics": result[
                    "compiled_rank_statistics"
                ],
                "post_quant_reconstruction": result[
                    "prompt_burnin_reconstruction_after_quantization"
                ],
                "exact_top1_match_rate": result["exact_top1_match_rate"],
                "top32_coverage": result["coverage_at_k"]["32"],
                "rank_statistics": result["rank_statistics"],
                "first_divergence": result["first_divergence"],
                "startup_exact_cost": result["startup_exact_cost"],
                "warm_decode_candidate_pass": warm_candidate_pass,
                "decision": result["decision"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
