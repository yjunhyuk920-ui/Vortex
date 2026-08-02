from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class MLPDecisionTrace:
    prompt: str
    prompt_tokens: int
    winner_token: int
    competitor_token: int
    exact_margin: float
    activations: list[torch.Tensor]
    output_duals: list[torch.Tensor]


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def collect_one_step_mlp_decision_trace(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
) -> MLPDecisionTrace:
    root = getattr(model, "model", None)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise ValueError("expected a Llama-style model.model.layers stack")

    encoded = encode_prompt(tokenizer, prompt, device)
    with torch.no_grad():
        prefill = model(**encoded, use_cache=True)
        first_token = torch.argmax(prefill.logits[:, -1, :], dim=-1)

    captures: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles: list[Any] = []
    for layer_index, layer in enumerate(layers):
        def hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            index: int = layer_index,
        ) -> None:
            if not inputs or not isinstance(output, torch.Tensor):
                raise RuntimeError("unexpected Llama MLP hook signature")
            captures[index] = (inputs[0], output)

        handles.append(layer.mlp.register_forward_hook(hook))

    model.zero_grad(set_to_none=True)
    decode = model(
        input_ids=first_token[:, None],
        past_key_values=prefill.past_key_values,
        use_cache=False,
    )
    for handle in handles:
        handle.remove()

    logits = decode.logits[:, -1, :]
    values, indices = torch.topk(logits, k=2, dim=-1)
    margin = values[0, 0] - values[0, 1]
    outputs = [captures[index][1] for index in range(len(layers))]
    gradients = torch.autograd.grad(margin, outputs, allow_unused=False)
    return MLPDecisionTrace(
        prompt=prompt,
        prompt_tokens=int(encoded["input_ids"].shape[1]),
        winner_token=int(indices[0, 0].item()),
        competitor_token=int(indices[0, 1].item()),
        exact_margin=float(margin.detach().item()),
        activations=[
            captures[index][0].detach()[0, -1].to("cpu", torch.float32).contiguous()
            for index in range(len(layers))
        ],
        output_duals=[
            gradient.detach()[0, -1].to("cpu", torch.float32).contiguous()
            for gradient in gradients
        ],
    )
