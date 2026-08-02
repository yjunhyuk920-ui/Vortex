from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_oracle_block_shared_adjoint import encode_prompt, require_transformers
from vortex_runtime.candidate_coverage import token_rank
from vortex_runtime.feasibility import default_specs
from vortex_runtime.layer_precision_oracle import (
    LayerPrecisionEffect,
    build_layer_precision_effect,
    precision_module_groups,
    rank_precision_effects,
    unique_weight_elements,
)
from vortex_runtime.precision_consensus import progressive_refinement_budget
from vortex_runtime.progressive_precision import (
    fake_quantize_full_rank_modules,
    symmetric_per_row_fake_quantize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use exact future labels only as an oracle to test whether a small "
            "Q6-to-Q8 decoder-layer subset can recover full-rank Q6 errors."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--base-bits", type=int, default=6)
    parser.add_argument("--upgrade-bits", type=int, default=8)
    parser.add_argument("--layers-per-group", type=int, default=2)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--runtime-refinement-fraction", type=float, default=0.25)
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(model_name: str, device: torch.device) -> nn.Module:
    AutoModelForCausalLM, _ = require_transformers()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model


def greedy_tokens(
    *,
    model: nn.Module,
    encoded: dict[str, torch.Tensor],
    count: int,
) -> torch.Tensor:
    generated: list[torch.Tensor] = []
    with torch.inference_mode():
        output = model(**encoded, use_cache=True, return_dict=True)
        current = torch.argmax(output.logits[:, -1, :], dim=-1).reshape(-1, 1)
        generated.append(current.reshape(-1))
        past = output.past_key_values
        for _ in range(1, count):
            output = model(
                input_ids=current,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = output.past_key_values
            current = torch.argmax(output.logits[:, -1, :], dim=-1).reshape(-1, 1)
            generated.append(current.reshape(-1))
    return torch.stack(generated, dim=1).to("cpu")


def teacher_forced_logits(
    *,
    model: nn.Module,
    encoded: dict[str, torch.Tensor],
    exact_tokens: torch.Tensor,
) -> torch.Tensor:
    input_ids = encoded["input_ids"]
    tokens = exact_tokens.to(input_ids.device)
    teacher_ids = (
        torch.cat((input_ids, tokens[:, :-1]), dim=1)
        if tokens.shape[1] > 1
        else input_ids
    )
    kwargs: dict[str, Any] = {
        "input_ids": teacher_ids,
        "use_cache": False,
        "return_dict": True,
    }
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        extension = torch.ones(
            (attention_mask.shape[0], tokens.shape[1] - 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        kwargs["attention_mask"] = torch.cat((attention_mask, extension), dim=1)
    with torch.inference_mode():
        output = model(**kwargs)
    start = input_ids.shape[1] - 1
    end = start + tokens.shape[1]
    return output.logits[:, start:end, :].detach().to("cpu", torch.float32)


def top1_tokens(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def error_positions(logits: torch.Tensor, exact_tokens: torch.Tensor) -> list[int]:
    predicted = top1_tokens(logits)
    exact = exact_tokens.to("cpu")
    return [
        position
        for position in range(exact.shape[1])
        if int(predicted[0, position].item()) != int(exact[0, position].item())
    ]


def exact_gap(logits: torch.Tensor, exact_tokens: torch.Tensor, position: int) -> float:
    position_logits = logits[0, position]
    exact_token = int(exact_tokens[0, position].item())
    top = int(torch.argmax(position_logits).item())
    return float((position_logits[top] - position_logits[exact_token]).item())


def module_maps(model: nn.Module) -> dict[str, nn.Module]:
    return dict(model.named_modules())


def apply_group_precision(
    *,
    target_model: nn.Module,
    source_model: nn.Module,
    module_names: Iterable[str],
    bits: int,
    row_chunk: int,
    keep_applied: bool,
) -> tuple[dict[str, torch.Tensor], int]:
    target_modules = module_maps(target_model)
    source_modules = module_maps(source_model)
    backups: dict[str, torch.Tensor] = {}
    seen: set[tuple[int, int]] = set()
    elements = 0
    with torch.no_grad():
        for name in module_names:
            target_module = target_modules[name]
            source_module = source_modules[name]
            if not isinstance(target_module, (nn.Linear, nn.Embedding)):
                continue
            if not isinstance(source_module, (nn.Linear, nn.Embedding)):
                raise RuntimeError(f"source module type mismatch for {name}")
            target_weight = target_module.weight
            identity = (
                target_weight.untyped_storage().data_ptr(),
                target_weight.storage_offset(),
            )
            if identity in seen:
                continue
            seen.add(identity)
            if not keep_applied:
                backups[name] = target_weight.detach().clone()
            restored, _ = symmetric_per_row_fake_quantize(
                source_module.weight,
                bits=bits,
                source_bits=16,
                name=f"{name}.weight",
                row_chunk=row_chunk,
            )
            target_weight.copy_(
                restored.to(device=target_weight.device, dtype=target_weight.dtype)
            )
            elements += target_weight.numel()
            del restored
    return backups, elements


def restore_group(model: nn.Module, backups: dict[str, torch.Tensor]) -> None:
    modules = module_maps(model)
    with torch.no_grad():
        for name, backup in backups.items():
            module = modules[name]
            if not isinstance(module, (nn.Linear, nn.Embedding)):
                raise RuntimeError(f"restore module type mismatch for {name}")
            module.weight.copy_(backup.to(module.weight.device, module.weight.dtype))


def evaluate_effect(
    *,
    group: str,
    module_names: tuple[str, ...],
    mixed_logits: torch.Tensor,
    base_logits: torch.Tensor,
    exact_tokens: torch.Tensor,
    group_elements: int,
    total_elements: int,
    residual_bits: int,
) -> LayerPrecisionEffect:
    base_errors = set(error_positions(base_logits, exact_tokens))
    mixed_errors = set(error_positions(mixed_logits, exact_tokens))
    corrected = len(base_errors - mixed_errors)
    introduced = len(mixed_errors - base_errors)
    exact_tokens_count = exact_tokens.shape[1] - len(mixed_errors)
    gap_reduction = sum(
        exact_gap(base_logits, exact_tokens, position)
        - exact_gap(mixed_logits, exact_tokens, position)
        for position in base_errors
    )
    return build_layer_precision_effect(
        group=group,
        module_names=module_names,
        unique_elements=group_elements,
        total_unique_elements=total_elements,
        residual_bits=residual_bits,
        corrected_base_errors=corrected,
        introduced_errors=introduced,
        exact_top1_tokens=exact_tokens_count,
        total_tokens=exact_tokens.shape[1],
        exact_gap_reduction=gap_reduction,
    )


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.layers_per_group <= 0 or args.row_chunk <= 0:
        raise SystemExit("tokens, layer grouping, and row chunk must be positive")
    if not 2 <= args.base_bits < args.upgrade_bits < 16:
        raise SystemExit("expected 2 <= base_bits < upgrade_bits < 16")
    if not 0 <= args.runtime_refinement_fraction <= 1:
        raise SystemExit("runtime refinement fraction must be in [0,1]")

    _, AutoTokenizer = require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(args.device)
    encoded = encode_prompt(tokenizer, args.eval_prompt, device)
    started = time.perf_counter()

    exact_model = load_model(args.model, device)
    exact_tokens = greedy_tokens(
        model=exact_model,
        encoded=encoded,
        count=args.tokens,
    )
    exact_logits = teacher_forced_logits(
        model=exact_model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )
    if error_positions(exact_logits, exact_tokens):
        raise RuntimeError("exact teacher-forced path disagrees with greedy tokens")

    base_model = load_model(args.model, device)
    base_precision, _ = fake_quantize_full_rank_modules(
        base_model,
        bits=args.base_bits,
        source_bits=16,
        row_chunk=args.row_chunk,
    )
    base_logits = teacher_forced_logits(
        model=base_model,
        encoded=encoded,
        exact_tokens=exact_tokens,
    )
    base_errors = error_positions(base_logits, exact_tokens)
    groups = precision_module_groups(
        base_model,
        layers_per_group=args.layers_per_group,
    )
    total_elements = unique_weight_elements(base_model)
    residual_bits = args.upgrade_bits - args.base_bits

    effects: list[LayerPrecisionEffect] = []
    for group, names in groups.items():
        backups, group_elements = apply_group_precision(
            target_model=base_model,
            source_model=exact_model,
            module_names=names,
            bits=args.upgrade_bits,
            row_chunk=args.row_chunk,
            keep_applied=False,
        )
        mixed_logits = teacher_forced_logits(
            model=base_model,
            encoded=encoded,
            exact_tokens=exact_tokens,
        )
        effect = evaluate_effect(
            group=group,
            module_names=names,
            mixed_logits=mixed_logits,
            base_logits=base_logits,
            exact_tokens=exact_tokens,
            group_elements=group_elements,
            total_elements=total_elements,
            residual_bits=residual_bits,
        )
        effects.append(effect)
        restore_group(base_model, backups)
        del mixed_logits, backups
        gc.collect()

    ranked = rank_precision_effects(effects)
    cumulative: list[dict[str, object]] = []
    cumulative_elements = 0
    exact_reached = False
    target, baseline = default_specs()
    for order, effect in enumerate(ranked, start=1):
        _, applied_elements = apply_group_precision(
            target_model=base_model,
            source_model=exact_model,
            module_names=effect.module_names,
            bits=args.upgrade_bits,
            row_chunk=args.row_chunk,
            keep_applied=True,
        )
        cumulative_elements += applied_elements
        logits = teacher_forced_logits(
            model=base_model,
            encoded=encoded,
            exact_tokens=exact_tokens,
        )
        errors = error_positions(logits, exact_tokens)
        layer_fraction = cumulative_elements / total_elements
        budget = progressive_refinement_budget(
            target=target,
            baseline=baseline,
            block_positions=4096,
            refinement_fraction=args.runtime_refinement_fraction,
            refined_layer_fraction=layer_fraction,
            consensus_bits=args.base_bits,
            residual_bits=residual_bits,
            consensus_effective_tops=120.0,
            residual_effective_tops=320.0,
        )
        cumulative.append(
            {
                "order": order,
                "added_group": effect.group,
                "groups": [item.group for item in ranked[:order]],
                "layer_fraction": layer_fraction,
                "residual_gib_405b": budget.residual_weight_gib,
                "exact_top1_tokens": args.tokens - len(errors),
                "exact_top1_rate": (args.tokens - len(errors)) / args.tokens,
                "remaining_error_positions": errors,
                "maximum_exact_token_rank": max(
                    token_rank(logits[0, position], int(exact_tokens[0, position].item()))
                    for position in range(args.tokens)
                ),
                "budget": budget.to_dict(),
            }
        )
        del logits
        if not errors:
            exact_reached = True
            break

    minimum_exact = cumulative[-1] if exact_reached else None
    qualifies = bool(
        minimum_exact is not None
        and minimum_exact["budget"]["ideal_pass"]
        and minimum_exact["budget"]["required_overlap_fraction"] <= 0.95
    )
    payload = {
        "evidence_level": "E1 exact-future Q6-to-Q8 layer-local oracle",
        "model": args.model,
        "prompt": args.eval_prompt,
        "tokens": args.tokens,
        "base_bits": args.base_bits,
        "upgrade_bits": args.upgrade_bits,
        "layers_per_group": args.layers_per_group,
        "base_precision": base_precision.to_dict(),
        "base_error_positions": base_errors,
        "base_exact_top1_rate": (args.tokens - len(base_errors)) / args.tokens,
        "groups": {group: list(names) for group, names in groups.items()},
        "individual_effects_ranked": [effect.to_dict() for effect in ranked],
        "cumulative_frontier": cumulative,
        "minimum_exact_cumulative_point": minimum_exact,
        "qualifies": qualifies,
        "decision": (
            "advance layer-local progressive precision to causal selection"
            if qualifies
            else "reject tested layer-local precision point"
        ),
        "oracle_warning": (
            "Group ranking uses exact future continuation labels. It proves only "
            "whether a budget-compatible corrective layer subset exists; it is "
            "not a runtime selector."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "base_errors": base_errors,
                "ranked_groups": [effect.group for effect in ranked],
                "minimum_exact": minimum_exact,
                "qualifies": qualifies,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
