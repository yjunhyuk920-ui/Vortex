from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Callable

import torch
from torch import nn

from vortex_runtime.adjoint_profiler import MarginRow
from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
)


@dataclass(frozen=True)
class MarginBoundProfile:
    candidates: list[dict[str, Any]]
    teacher_sequence_tokens: int
    generated_targets: int
    approximate_margin_sum: float
    margin_rows: list[MarginRow]
    missing_gradient_modules: list[str]
    metadata_bytes: int

    def metadata(self) -> dict[str, Any]:
        return {
            "teacher_sequence_tokens": self.teacher_sequence_tokens,
            "generated_targets": self.generated_targets,
            "approximate_margin_sum": self.approximate_margin_sum,
            "margin_rows": [row.__dict__ for row in self.margin_rows],
            "missing_gradient_modules": self.missing_gradient_modules,
            "metadata_bytes": self.metadata_bytes,
            "selector_contract": (
                "tile score uses precomputed ||W_tile||_F and runtime proposal "
                "input-residual/output-gradient energies; no exact target token "
                "or full runtime weight scan"
            ),
        }


def cauchy_margin_bound(
    *,
    weight_energy: float,
    gradient_energy: float,
    residual_energy: float,
) -> float:
    if min(weight_energy, gradient_energy, residual_energy) < 0:
        raise ValueError("energies must be non-negative")
    return sqrt(weight_energy * gradient_energy * residual_energy)


def _tile_energy(
    tensor: torch.Tensor,
    *,
    width: int,
    tile: int,
) -> torch.Tensor:
    flat = tensor.reshape(-1, width).detach().to("cpu", torch.float32)
    tile_count = (width + tile - 1) // tile
    padded_width = tile_count * tile
    padded = torch.zeros((flat.shape[0], padded_width), dtype=torch.float32)
    padded[:, :width] = flat
    return padded.reshape(flat.shape[0], tile_count, tile).square().sum(
        dim=(0, 2)
    )


def profile_proposal_margin_bounds(
    *,
    model: nn.Module,
    tokenizer: Any,
    eval_prompt: str,
    proposal_sequence: list[int],
    replacements: dict[str, DecisionResidualTileAtlasLinearModule],
    device: torch.device,
    row_tile: int,
    col_tile: int,
    encode_prompt: Callable[[Any, str, torch.device], dict[str, torch.Tensor]],
) -> MarginBoundProfile:
    """Rank residual tiles with a target-independent first-order upper bound.

    For a residual tile, proposal input residual ``R``, and proposal-margin
    adjoint ``G``:

    ``|<W_tile, G.T @ R>| <= ||W_tile||_F ||G||_F ||R||_F``.

    ``||W_tile||_F`` is precomputed once while producing the runtime format.
    Runtime selection reads one scalar per tile and computes row/column energy
    from the hot proposal. It does not read every exact weight tile.
    """

    if row_tile <= 0 or col_tile <= 0:
        raise ValueError("tile dimensions must be positive")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in replacements.values():
        module.reset_residual_tile_profile(
            row_tile=row_tile,
            col_tile=col_tile,
        )
        module.set_mode("project")

    captured_inputs: dict[str, torch.Tensor] = {}
    captured_outputs: dict[str, torch.Tensor] = {}
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

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(output_hook))

    try:
        encoded = encode_prompt(tokenizer, eval_prompt, device)
        prompt_ids = encoded["input_ids"][0].detach().cpu().tolist()
        if proposal_sequence[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("proposal sequence does not preserve prompt prefix")
        if len(proposal_sequence) <= len(prompt_ids):
            raise RuntimeError("proposal sequence contains no generated targets")

        teacher_ids = torch.tensor(
            [proposal_sequence[:-1]],
            dtype=torch.long,
            device=device,
        )
        embeddings = model.get_input_embeddings()(teacher_ids).detach()
        embeddings.requires_grad_(True)
        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=torch.ones_like(teacher_ids),
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits

        margin_terms: list[torch.Tensor] = []
        margin_rows: list[MarginRow] = []
        generated_count = len(proposal_sequence) - len(prompt_ids)
        for offset in range(generated_count):
            position = len(prompt_ids) - 1 + offset
            proposed = int(proposal_sequence[len(prompt_ids) + offset])
            row = logits[0, position]
            competitors = row.clone()
            competitors[proposed] = float("-inf")
            competitor = int(torch.argmax(competitors).item())
            margin = row[proposed] - row[competitor]
            margin_terms.append(margin)
            margin_rows.append(
                MarginRow(
                    position=position,
                    target_token=proposed,
                    competitor_token=competitor,
                    approximate_margin=float(margin.detach().item()),
                )
            )

        objective = torch.stack(margin_terms).sum()
        ordered_names = [
            name
            for name in replacements
            if name in captured_outputs and captured_outputs[name].requires_grad
        ]
        gradients = torch.autograd.grad(
            objective,
            [captured_outputs[name] for name in ordered_names],
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        gradient_map = dict(zip(ordered_names, gradients))
    finally:
        for handle in handles:
            handle.remove()

    candidates: list[dict[str, Any]] = []
    missing: list[str] = []
    metadata_bytes = 0
    for name, module in replacements.items():
        captured_input = captured_inputs.get(name)
        gradient = gradient_map.get(name)
        weight_energy = module.weight_tile_energy
        if captured_input is None:
            raise RuntimeError(f"missing captured input for {name}")
        if gradient is None:
            missing.append(name)
            continue
        if weight_energy is None:
            raise RuntimeError(f"missing precomputed weight energy for {name}")

        residual = module._input_residual(captured_input)
        residual_energy = _tile_energy(
            residual,
            width=module.exact.in_features,
            tile=col_tile,
        )
        gradient_energy = _tile_energy(
            gradient,
            width=module.exact.out_features,
            tile=row_tile,
        )
        metadata_bytes += weight_energy.numel() * weight_energy.element_size()

        row_tiles, col_tiles = weight_energy.shape
        for row_index in range(row_tiles):
            row_start = row_index * row_tile
            row_end = min(row_start + row_tile, module.exact.out_features)
            for col_index in range(col_tiles):
                col_start = col_index * col_tile
                col_end = min(col_start + col_tile, module.exact.in_features)
                tile_bytes = (
                    (row_end - row_start)
                    * (col_end - col_start)
                    * module.exact.weight.element_size()
                )
                weight_sq = float(weight_energy[row_index, col_index].item())
                grad_sq = float(gradient_energy[row_index].item())
                residual_sq = float(residual_energy[col_index].item())
                bound = cauchy_margin_bound(
                    weight_energy=weight_sq,
                    gradient_energy=grad_sq,
                    residual_energy=residual_sq,
                )
                candidates.append(
                    {
                        "module": name,
                        "row_tile": row_index,
                        "col_tile": col_index,
                        "row_start": row_start,
                        "row_end": row_end,
                        "col_start": col_start,
                        "col_end": col_end,
                        "weight_bytes": tile_bytes,
                        "margin_bound": bound,
                        "bound_per_byte": bound / max(1, tile_bytes),
                        "weight_energy": weight_sq,
                        "gradient_energy": grad_sq,
                        "input_residual_energy": residual_sq,
                    }
                )

    return MarginBoundProfile(
        candidates=candidates,
        teacher_sequence_tokens=len(proposal_sequence),
        generated_targets=len(margin_rows),
        approximate_margin_sum=float(objective.detach().item()),
        margin_rows=margin_rows,
        missing_gradient_modules=missing,
        metadata_bytes=metadata_bytes,
    )
