from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
    replace_with_decision_tile_modules,
    score_adjoint_residual_tiles,
)


DEFAULT_BUILD_PROMPTS = [
    "Explain why the sky appears blue and why sunsets appear red.",
    "Write Python code for merge sort and explain its complexity.",
    "한국어로 PLM BOM 변경 검증 절차와 오류 처리 방법을 설명해줘.",
    "Solve a probability problem using conditional probability step by step.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank exact residual weight tiles by their first-order influence "
            "on the exact-target versus approximate-competitor logit margin."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--max-rank", type=int, default=32)
    parser.add_argument("--row-tile", type=int, default=128)
    parser.add_argument("--col-tile", type=int, default=128)
    parser.add_argument(
        "--build-prompt",
        action="append",
        dest="build_prompts",
    )
    parser.add_argument(
        "--eval-prompt",
        default="한국어로 정렬 알고리즘의 시간 복잡도를 설명해줘.",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        dest="suffixes",
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("oracle_adjoint_tile_repair.json"),
    )
    return parser.parse_args()


def require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("pip install transformers sentencepiece") from exc
    return AutoModelForCausalLM, AutoTokenizer


def encode_prompt(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> list[int]:
    encoded = encode_prompt(tokenizer, prompt, device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return output[0].detach().to("cpu").tolist()


def model_parameter_bytes(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        pointer = parameter.untyped_storage().data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        total += parameter.numel() * parameter.element_size()
    return total


def configure_top_tiles(
    replacements: dict[str, DecisionResidualTileAtlasLinearModule],
    candidates: list[dict[str, Any]],
    count: int,
    row_tile: int,
    col_tile: int,
) -> int:
    selected: dict[str, list[tuple[int, int]]] = {
        name: [] for name in replacements
    }
    for item in candidates[:count]:
        selected[str(item["module"])].append(
            (int(item["row_tile"]), int(item["col_tile"]))
        )
    for name, module in replacements.items():
        module.configure_residual_tile_repair(
            row_tile=row_tile,
            col_tile=col_tile,
            tile_indices=selected[name],
        )
        module.set_mode("project_residual_repair")
    return sum(
        module.selected_residual_repair_bytes
        for module in replacements.values()
    )


def prefix_metrics(
    *,
    count: int,
    exact_bytes: int,
    full_model_bytes: int,
    exact_match: bool,
) -> dict[str, int | float | bool | None]:
    return {
        "selected_tiles": count,
        "exact_weight_bytes_per_token": exact_bytes,
        "full_model_repair_fraction_per_token": exact_bytes / full_model_bytes,
        "zero_exact_repair": exact_bytes == 0,
        "tokens_per_full_repair_equivalent": (
            None if exact_bytes == 0 else full_model_bytes / exact_bytes
        ),
        "exact_sequence_match": exact_match,
    }


def count_within_bytes(cumulative: list[int], maximum: float) -> int:
    count = 0
    for index, value in enumerate(cumulative, start=1):
        if value > maximum:
            break
        count = index
    return count


def capture_adjoint_tiles(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    eval_prompt: str,
    exact_sequence: list[int],
    replacements: dict[str, DecisionResidualTileAtlasLinearModule],
    device: torch.device,
    row_tile: int,
    col_tile: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in replacements.values():
        module.set_mode("project")

    captured_inputs: dict[str, torch.Tensor] = {}
    captured_outputs: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in replacements.items():
        def pre_hook(
            _module: torch.nn.Module,
            args: tuple[torch.Tensor, ...],
            *,
            key: str = name,
        ) -> None:
            captured_inputs[key] = args[0]

        def output_hook(
            _module: torch.nn.Module,
            _args: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            key: str = name,
        ) -> None:
            output.retain_grad()
            captured_outputs[key] = output

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(output_hook))

    try:
        encoded = encode_prompt(tokenizer, eval_prompt, device)
        prompt_ids = encoded["input_ids"][0].detach().to("cpu").tolist()
        if exact_sequence[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("generated exact sequence does not preserve prompt prefix")
        teacher_ids = torch.tensor(
            [exact_sequence[:-1]],
            dtype=torch.long,
            device=device,
        )
        embeddings = model.get_input_embeddings()(teacher_ids).detach()
        embeddings.requires_grad_(True)
        attention_mask = torch.ones(
            teacher_ids.shape,
            dtype=torch.long,
            device=device,
        )
        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits

        margin_terms: list[torch.Tensor] = []
        margin_rows: list[dict[str, int | float]] = []
        generated_count = len(exact_sequence) - len(prompt_ids)
        for offset in range(generated_count):
            position = len(prompt_ids) - 1 + offset
            target = int(exact_sequence[len(prompt_ids) + offset])
            row = logits[0, position]
            mask = torch.ones_like(row, dtype=torch.bool)
            mask[target] = False
            competitor = int(torch.argmax(row.masked_fill(~mask, float("-inf"))).item())
            margin = row[target] - row[competitor]
            margin_terms.append(margin)
            margin_rows.append(
                {
                    "position": position,
                    "target_token": target,
                    "competitor_token": competitor,
                    "approximate_margin": float(margin.detach().item()),
                }
            )
        if not margin_terms:
            raise RuntimeError("exact sequence contains no generated targets")
        objective = torch.stack(margin_terms).sum()
        objective.backward()
    finally:
        for handle in handles:
            handle.remove()

    candidates: list[dict[str, Any]] = []
    missing_gradients: list[str] = []
    signed_total = 0.0
    for name, module in replacements.items():
        captured_input = captured_inputs.get(name)
        captured_output = captured_outputs.get(name)
        if captured_input is None or captured_output is None:
            raise RuntimeError(f"missing capture for {name}")
        gradient = captured_output.grad
        if gradient is None:
            missing_gradients.append(name)
            continue
        tiles = score_adjoint_residual_tiles(
            module,
            input_tensor=captured_input,
            output_gradient=gradient,
            row_tile=row_tile,
            col_tile=col_tile,
        )
        for tile in tiles:
            signed_total += float(tile["signed_margin_contribution"])
            candidates.append({"module": name, **tile})

    metadata = {
        "teacher_sequence_tokens": len(exact_sequence),
        "generated_targets": len(margin_rows),
        "approximate_margin_sum": float(objective.detach().item()),
        "margin_rows": margin_rows,
        "missing_gradient_modules": missing_gradients,
        "signed_full_residual_linearized_contribution": signed_total,
    }
    return candidates, metadata


def evaluate_ranking(
    *,
    name: str,
    candidates: list[dict[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    eval_prompt: str,
    exact_tokens: list[int],
    replacements: dict[str, DecisionResidualTileAtlasLinearModule],
    device: torch.device,
    max_new_tokens: int,
    row_tile: int,
    col_tile: int,
    full_model_bytes: int,
) -> dict[str, Any]:
    cumulative: list[int] = []
    total = 0
    for item in candidates:
        total += int(item["weight_bytes"])
        cumulative.append(total)

    gate_minimum = 491.29915997929805
    promotion_threshold = 600.0
    rejection_threshold = 300.0
    max_gate_bytes = full_model_bytes / gate_minimum
    max_promotion_bytes = full_model_bytes / promotion_threshold
    max_rejection_bytes = full_model_bytes / rejection_threshold
    promotion_count = count_within_bytes(cumulative, max_promotion_bytes)
    gate_count = count_within_bytes(cumulative, max_gate_bytes)
    rejection_count = count_within_bytes(cumulative, max_rejection_bytes)

    probe_counts = {0, 1, 2, 4, 8, 16, 32, 64}
    probe_counts.update(range(4, rejection_count + 1, 4))
    probe_counts.update({promotion_count, gate_count, rejection_count})
    value = 128
    while value < len(candidates):
        probe_counts.add(value)
        value *= 2
    probe_counts.add(len(candidates))
    ordered = sorted(count for count in probe_counts if 0 <= count <= len(candidates))

    tested: list[dict[str, int | float | bool | None]] = []
    first_match = None
    gate_match = None
    for count in ordered:
        exact_bytes = configure_top_tiles(
            replacements,
            candidates,
            count,
            row_tile,
            col_tile,
        )
        tokens = generate(
            model,
            tokenizer,
            eval_prompt,
            device,
            max_new_tokens,
        )
        item = prefix_metrics(
            count=count,
            exact_bytes=exact_bytes,
            full_model_bytes=full_model_bytes,
            exact_match=tokens == exact_tokens,
        )
        tested.append(item)
        if item["exact_sequence_match"] and first_match is None:
            first_match = item
        if (
            item["exact_sequence_match"]
            and exact_bytes <= max_gate_bytes
            and gate_match is None
        ):
            gate_match = item

    return {
        "ranking": name,
        "candidate_tiles": len(candidates),
        "promotion_budget_tile_count": promotion_count,
        "gate_budget_tile_count": gate_count,
        "rejection_budget_tile_count": rejection_count,
        "gate_budget_match": gate_match,
        "first_repair_match": first_match,
        "tested_prefixes": tested,
        "top_tiles": candidates[:64],
    }


def main() -> None:
    args = parse_args()
    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    suffixes = tuple(
        args.suffixes or ("self_attn.o_proj", "mlp.down_proj")
    )
    replacements = replace_with_decision_tile_modules(
        model,
        suffixes=suffixes,
        max_rank=args.max_rank,
    )
    if not replacements:
        raise RuntimeError(f"no modules matched {suffixes}")

    for module in replacements.values():
        module.set_mode("exact")
    exact_tokens = generate(
        model,
        tokenizer,
        args.eval_prompt,
        device,
        args.max_new_tokens,
    )

    build_prompts = args.build_prompts or DEFAULT_BUILD_PROMPTS
    for module in replacements.values():
        module.set_mode("learn_exact")
    for prompt in build_prompts:
        generate(model, tokenizer, prompt, device, args.max_new_tokens)

    ranks = {name: module.atlas.rank for name, module in replacements.items()}
    capsule_bytes = sum(
        module.atlas.capsule_bytes for module in replacements.values()
    )
    full_model_bytes = model_parameter_bytes(model)
    managed_weight_bytes = sum(
        module.logical_weight_bytes for module in replacements.values()
    )

    started = time.perf_counter()
    candidates, adjoint_metadata = capture_adjoint_tiles(
        model=model,
        tokenizer=tokenizer,
        eval_prompt=args.eval_prompt,
        exact_sequence=exact_tokens,
        replacements=replacements,
        device=device,
        row_tile=args.row_tile,
        col_tile=args.col_tile,
    )

    positive = sorted(
        candidates,
        key=lambda item: (
            float(item["positive_contribution_per_byte"]),
            float(item["signed_margin_contribution"]),
        ),
        reverse=True,
    )
    absolute = sorted(
        candidates,
        key=lambda item: (
            float(item["absolute_contribution_per_byte"]),
            float(item["absolute_margin_contribution"]),
        ),
        reverse=True,
    )

    positive_result = evaluate_ranking(
        name="positive-signed-margin",
        candidates=positive,
        model=model,
        tokenizer=tokenizer,
        eval_prompt=args.eval_prompt,
        exact_tokens=exact_tokens,
        replacements=replacements,
        device=device,
        max_new_tokens=args.max_new_tokens,
        row_tile=args.row_tile,
        col_tile=args.col_tile,
        full_model_bytes=full_model_bytes,
    )
    absolute_result = evaluate_ranking(
        name="absolute-margin-influence",
        candidates=absolute,
        model=model,
        tokenizer=tokenizer,
        eval_prompt=args.eval_prompt,
        exact_tokens=exact_tokens,
        replacements=replacements,
        device=device,
        max_new_tokens=args.max_new_tokens,
        row_tile=args.row_tile,
        col_tile=args.col_tile,
        full_model_bytes=full_model_bytes,
    )

    gate_match = positive_result["gate_budget_match"] or absolute_result[
        "gate_budget_match"
    ]
    first_matches = [
        item
        for item in (
            positive_result["first_repair_match"],
            absolute_result["first_repair_match"],
        )
        if item is not None
    ]
    best_first_match = max(
        first_matches,
        key=lambda item: float(item["tokens_per_full_repair_equivalent"]),
        default=None,
    )
    if gate_match is not None:
        decision = "adjoint tile oracle meets Gate 0 byte envelope"
    elif best_first_match is not None:
        decision = "adjoint tile oracle restores output only below Gate 0 efficiency"
    else:
        decision = "adjoint tile oracle did not restore the sequence"

    result = {
        "evidence_level": "E1 exact-target adjoint tile oracle",
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "managed_suffixes": list(suffixes),
        "matched_modules": len(replacements),
        "max_rank": args.max_rank,
        "row_tile": args.row_tile,
        "col_tile": args.col_tile,
        "build_prompts": build_prompts,
        "eval_prompt": args.eval_prompt,
        "max_new_tokens": args.max_new_tokens,
        "rank_min": min(ranks.values()),
        "rank_max": max(ranks.values()),
        "rank_mean": sum(ranks.values()) / len(ranks),
        "capsule_bytes": capsule_bytes,
        "full_model_weight_bytes": full_model_bytes,
        "managed_weight_bytes": managed_weight_bytes,
        "adjoint": adjoint_metadata,
        "positive_ranking": positive_result,
        "absolute_ranking": absolute_result,
        "gate_budget_match": gate_match,
        "best_first_repair_match": best_first_match,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "scope_note": (
            "Extremely optimistic oracle: exact generated target tokens define "
            "the margin, exact evaluation activations and gradients rank tiles, "
            "and nested prefixes are tested. A deployable runtime would need to "
            "predict this information without the exact target sequence."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                "rank_min": result["rank_min"],
                "rank_max": result["rank_max"],
                "missing_gradient_modules": adjoint_metadata[
                    "missing_gradient_modules"
                ],
                "gate_budget_match": gate_match,
                "best_first_repair_match": best_first_match,
                "decision": decision,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
