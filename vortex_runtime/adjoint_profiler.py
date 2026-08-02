from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
    score_adjoint_residual_tiles,
)


@dataclass(frozen=True)
class MarginRow:
    position: int
    target_token: int
    competitor_token: int
    approximate_margin: float


@dataclass(frozen=True)
class AdjointProfile:
    candidates: list[dict[str, Any]]
    teacher_sequence_tokens: int
    generated_targets: int
    approximate_margin_sum: float
    margin_rows: list[MarginRow]
    missing_gradient_modules: list[str]
    non_differentiable_modules: list[str]
    signed_full_residual_linearized_contribution: float
    teacher_source: str

    def metadata(self) -> dict[str, Any]:
        return {
            "teacher_sequence_tokens": self.teacher_sequence_tokens,
            "generated_targets": self.generated_targets,
            "approximate_margin_sum": self.approximate_margin_sum,
            "margin_rows": [row.__dict__ for row in self.margin_rows],
            "missing_gradient_modules": self.missing_gradient_modules,
            "non_differentiable_modules": self.non_differentiable_modules,
            "signed_full_residual_linearized_contribution": (
                self.signed_full_residual_linearized_contribution
            ),
            "teacher_source": self.teacher_source,
        }


def profile_teacher_sequence_margin_tiles(
    *,
    model: nn.Module,
    tokenizer: Any,
    eval_prompt: str,
    teacher_sequence: list[int],
    teacher_source: str,
    replacements: dict[str, DecisionResidualTileAtlasLinearModule],
    device: torch.device,
    row_tile: int,
    col_tile: int,
    encode_prompt: Callable[[Any, str, torch.device], dict[str, torch.Tensor]],
) -> AdjointProfile:
    """Profile first-order margin influence for a supplied teacher sequence.

    The teacher may be the exact target continuation for an optimistic oracle or
    the hot path's own proposal for a target-independent selector. Gradients are
    requested from the scalar proposed-token margin objective to every captured
    projection output. Exact evaluation output is not required by this function.
    """

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in replacements.values():
        module.set_mode("project")

    captured_inputs: dict[str, torch.Tensor] = {}
    captured_outputs: dict[str, torch.Tensor] = {}
    non_differentiable: list[str] = []
    handles: list[Any] = []

    for name, module in replacements.items():
        def pre_hook(
            _module: nn.Module,
            args: tuple[torch.Tensor, ...],
            *,
            key: str = name,
        ) -> None:
            captured_inputs[key] = args[0].detach()

        def output_hook(
            _module: nn.Module,
            _args: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            key: str = name,
        ) -> None:
            captured_outputs[key] = output
            if not output.requires_grad:
                non_differentiable.append(key)

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(output_hook))

    try:
        encoded = encode_prompt(tokenizer, eval_prompt, device)
        prompt_ids = encoded["input_ids"][0].detach().cpu().tolist()
        if teacher_sequence[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("teacher sequence does not preserve prompt prefix")
        if len(teacher_sequence) <= len(prompt_ids):
            raise RuntimeError("teacher sequence contains no generated targets")

        teacher_ids = torch.tensor(
            [teacher_sequence[:-1]],
            dtype=torch.long,
            device=device,
        )
        embeddings = model.get_input_embeddings()(teacher_ids).detach()
        embeddings.requires_grad_(True)
        attention_mask = torch.ones_like(teacher_ids)
        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits

        margin_terms: list[torch.Tensor] = []
        margin_rows: list[MarginRow] = []
        generated_count = len(teacher_sequence) - len(prompt_ids)
        for offset in range(generated_count):
            position = len(prompt_ids) - 1 + offset
            target = int(teacher_sequence[len(prompt_ids) + offset])
            row = logits[0, position]
            competitor_scores = row.clone()
            competitor_scores[target] = float("-inf")
            competitor = int(torch.argmax(competitor_scores).item())
            margin = row[target] - row[competitor]
            margin_terms.append(margin)
            margin_rows.append(
                MarginRow(
                    position=position,
                    target_token=target,
                    competitor_token=competitor,
                    approximate_margin=float(margin.detach().item()),
                )
            )

        objective = torch.stack(margin_terms).sum()
        if not objective.requires_grad:
            raise RuntimeError("margin objective is not differentiable")

        ordered_names = [
            name for name in replacements if name in captured_outputs
        ]
        differentiable_names = [
            name
            for name in ordered_names
            if captured_outputs[name].requires_grad
        ]
        output_tensors = [captured_outputs[name] for name in differentiable_names]
        gradients = torch.autograd.grad(
            objective,
            output_tensors,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        gradient_map = dict(zip(differentiable_names, gradients))
    finally:
        for handle in handles:
            handle.remove()

    candidates: list[dict[str, Any]] = []
    missing: list[str] = []
    signed_total = 0.0
    for name, module in replacements.items():
        captured_input = captured_inputs.get(name)
        captured_output = captured_outputs.get(name)
        if captured_input is None or captured_output is None:
            raise RuntimeError(f"missing forward capture for {name}")
        gradient = gradient_map.get(name)
        if gradient is None:
            missing.append(name)
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

    return AdjointProfile(
        candidates=candidates,
        teacher_sequence_tokens=len(teacher_sequence),
        generated_targets=len(margin_rows),
        approximate_margin_sum=float(objective.detach().item()),
        margin_rows=margin_rows,
        missing_gradient_modules=missing,
        non_differentiable_modules=sorted(set(non_differentiable)),
        signed_full_residual_linearized_contribution=signed_total,
        teacher_source=teacher_source,
    )


def profile_exact_target_margin_tiles(
    *,
    model: nn.Module,
    tokenizer: Any,
    eval_prompt: str,
    exact_sequence: list[int],
    replacements: dict[str, DecisionResidualTileAtlasLinearModule],
    device: torch.device,
    row_tile: int,
    col_tile: int,
    encode_prompt: Callable[[Any, str, torch.device], dict[str, torch.Tensor]],
) -> AdjointProfile:
    """Compatibility wrapper for the exact-target optimistic oracle."""

    return profile_teacher_sequence_margin_tiles(
        model=model,
        tokenizer=tokenizer,
        eval_prompt=eval_prompt,
        teacher_sequence=exact_sequence,
        teacher_source="exact_target_continuation",
        replacements=replacements,
        device=device,
        row_tile=row_tile,
        col_tile=col_tile,
        encode_prompt=encode_prompt,
    )
