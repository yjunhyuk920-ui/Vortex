from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PromptContinuationTrace:
    prompt: str
    prompt_token_ids: torch.Tensor
    prompt_hidden_states: torch.Tensor
    first_generated_token: int
    continuation_token_ids: torch.Tensor
    continuation_hidden_states: torch.Tensor

    @property
    def prompt_tokens(self) -> int:
        return int(self.prompt_token_ids.numel())

    @property
    def continuation_steps(self) -> int:
        return int(self.continuation_hidden_states.shape[0])


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def collect_prompt_continuation_trace(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    continuation_steps: int,
    device: torch.device,
) -> PromptContinuationTrace:
    """Collect exact prompt states and a temporally disjoint greedy continuation.

    The prompt hidden trajectory is the only build input for the decision
    program. Continuation tokens and states are returned solely as evaluation
    evidence. Manual decode intentionally continues after EOS so a trivial EOS
    loop can be detected and rejected by the experiment.
    """

    if continuation_steps <= 0:
        raise ValueError("continuation_steps must be positive")
    encoded = encode_prompt(tokenizer, prompt, device)
    model.eval()
    with torch.no_grad():
        prefill = model(
            **encoded,
            use_cache=True,
            output_hidden_states=True,
        )
        prompt_hidden = (
            prefill.hidden_states[-1][0]
            .detach()
            .to("cpu", torch.float32)
            .contiguous()
        )
        prompt_tokens = encoded["input_ids"][0].detach().to("cpu", torch.long)
        first_token_tensor = torch.argmax(prefill.logits[:, -1, :], dim=-1)
        first_token = int(first_token_tensor[0].item())
        past_key_values = prefill.past_key_values
        control = first_token_tensor[:, None]

    continuation_tokens = [first_token]
    continuation_hidden: list[torch.Tensor] = []
    for _ in range(continuation_steps):
        with torch.no_grad():
            decode = model(
                input_ids=control,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            hidden = (
                decode.hidden_states[-1][0, -1]
                .detach()
                .to("cpu", torch.float32)
                .contiguous()
            )
            next_token_tensor = torch.argmax(decode.logits[:, -1, :], dim=-1)
            next_token = int(next_token_tensor[0].item())
            past_key_values = decode.past_key_values
            control = next_token_tensor[:, None]
        continuation_hidden.append(hidden)
        continuation_tokens.append(next_token)

    return PromptContinuationTrace(
        prompt=prompt,
        prompt_token_ids=prompt_tokens.contiguous(),
        prompt_hidden_states=prompt_hidden,
        first_generated_token=first_token,
        continuation_token_ids=torch.tensor(
            continuation_tokens,
            dtype=torch.long,
        ),
        continuation_hidden_states=torch.stack(continuation_hidden, dim=0),
    )
