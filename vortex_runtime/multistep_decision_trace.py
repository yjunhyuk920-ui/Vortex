from __future__ import annotations

import copy
from dataclasses import dataclass
import gc
from typing import Any

import torch


@dataclass
class CausalMLPDecisionTrace:
    prompt: str
    prompt_tokens: int
    decode_step: int
    routing_hidden: torch.Tensor
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


def _llama_layers(model: torch.nn.Module) -> Any:
    root = getattr(model, "model", None)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise ValueError("expected a Llama-style model.model.layers stack")
    return layers


def collect_causal_multistep_mlp_traces(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    steps: int,
    device: torch.device,
) -> list[CausalMLPDecisionTrace]:
    """Collect exact current MLP inputs/duals with a causal prior-token router key.

    The routing key for step `t` is the final hidden state produced by the
    previously completed token. Current MLP activations, current output duals,
    current logits, and future tokens are never used by the router key.

    A gradient-enabled decode obtains the exact top-one-versus-runner-up MLP
    output duals. A separate no-grad decode of the same token advances the KV
    cache without retaining the previous step's autograd graph. The gradient
    decode receives a deep copy of the cache so mutable cache implementations
    cannot change the progression state.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    layers = _llama_layers(model)
    model.eval()
    encoded = encode_prompt(tokenizer, prompt, device)

    with torch.no_grad():
        prefill = model(
            **encoded,
            use_cache=True,
            output_hidden_states=True,
        )
        next_input = torch.argmax(prefill.logits[:, -1, :], dim=-1)[:, None]
        past_key_values = prefill.past_key_values
        routing_hidden = (
            prefill.hidden_states[-1][0, -1]
            .detach()
            .to("cpu", torch.float32)
            .contiguous()
        )
    prompt_tokens = int(encoded["input_ids"].shape[1])
    traces: list[CausalMLPDecisionTrace] = []

    for decode_step in range(steps):
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
        gradient_cache = copy.deepcopy(past_key_values)
        try:
            decode = model(
                input_ids=next_input,
                past_key_values=gradient_cache,
                use_cache=False,
            )
        finally:
            for handle in handles:
                handle.remove()

        if len(captures) != len(layers):
            raise RuntimeError("not every decoder MLP produced a trace")
        logits = decode.logits[:, -1, :]
        values, indices = torch.topk(logits, k=2, dim=-1)
        margin = values[0, 0] - values[0, 1]
        outputs = [captures[index][1] for index in range(len(layers))]
        gradients = torch.autograd.grad(
            margin,
            outputs,
            allow_unused=False,
            retain_graph=False,
        )
        winner = int(indices[0, 0].item())
        competitor = int(indices[0, 1].item())
        traces.append(
            CausalMLPDecisionTrace(
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                decode_step=decode_step,
                routing_hidden=routing_hidden.clone(),
                winner_token=winner,
                competitor_token=competitor,
                exact_margin=float(margin.detach().item()),
                activations=[
                    captures[index][0]
                    .detach()[0, -1]
                    .to("cpu", torch.float32)
                    .contiguous()
                    for index in range(len(layers))
                ],
                output_duals=[
                    gradient.detach()[0, -1]
                    .to("cpu", torch.float32)
                    .contiguous()
                    for gradient in gradients
                ],
            )
        )

        winner_tensor = indices[:, 0].detach()[:, None]
        del (
            gradients,
            outputs,
            captures,
            decode,
            logits,
            values,
            margin,
            gradient_cache,
        )
        model.zero_grad(set_to_none=True)
        gc.collect()

        with torch.no_grad():
            refresh = model(
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            refresh_winner = torch.argmax(refresh.logits[:, -1, :], dim=-1)[:, None]
            if not torch.equal(refresh_winner, winner_tensor):
                raise RuntimeError("gradient and no-grad decode winners diverged")
            past_key_values = refresh.past_key_values
            routing_hidden = (
                refresh.hidden_states[-1][0, -1]
                .detach()
                .to("cpu", torch.float32)
                .contiguous()
            )
            next_input = winner_tensor.to(device)
        del refresh
        gc.collect()

    return traces
